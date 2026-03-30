"""무결성 보장 파이프라인 (Zero Hallucination Pipeline).

4단계 파이프라인: Planner → Map-Reduce STORM → CoVe 검증 → Deterministic Sanitizer.
Plan-and-Execute, Map-Reduce, CoVe, Sanitizer를 결합하여
LLM 출력의 환각을 아키텍처 레벨에서 결정론적으로 통제한다.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from harness.base import HarnessConfig, HarnessState
from harness.models import ExecutionPlan, parse_questions
from harness.sanitizer import DeterministicSanitizer, LocalChunkDB

# ── 상수 ──

MAX_CORRECTIONS = 3
STORM_PERSONAS = ("비판적 검토자", "도메인 전문가", "반론 제기자")

# 모든 LLM 호출에 공통으로 삽입되는 주제 보존 규칙
TOPIC_PRESERVATION_RULE = (
    "[중요 규칙] 사용자가 명시한 주제, 고유명사, 제품명은 절대 변경하거나 "
    "다른 것으로 대체하지 마라. 해당 주제에 대한 정보가 없으면 "
    "'정보를 찾을 수 없다'고 명확히 밝혀라. "
    "추측으로 다른 주제를 대입하는 것은 금지한다."
)


# ── 상태 스키마 ──


class ZeroHallucinationState(HarnessState):
    """무결성 보장 파이프라인 상태.

    HarnessState를 확장하여 실행 계획, 워커 결과, 초안,
    검증 질문/답변, sanitizer 오류, 교정 횟수를 추가한다.
    """

    term_context: str  # 용어 확인 결과 (Term Resolver 출력)
    execution_plan: ExecutionPlan | None
    worker_results: Annotated[list[dict], operator.add]
    draft: str | None
    verification_questions: list[str]
    verification_answers: Annotated[list[dict], operator.add]
    sanitizer_errors: list[str]
    correction_count: int


# ── 노드 함수 팩토리 ──
#
# 각 노드 함수는 build_zero_hallucination_pipeline() 내부에서
# 모델 인스턴스를 클로저로 캡처하여 생성된다.
# 아래는 독립적으로 테스트 가능하도록 모델을 인자로 받는 팩토리 함수이다.


def make_term_resolver_node(resolver_model: BaseChatModel, knowledge_base=None):
    """0단계: 쿼리 내 고유명사·전문 용어를 사전 확인한다.

    조사를 시작하기 전에 모르는 용어를 먼저 파악하여,
    이후 모든 노드에 정확한 컨텍스트를 제공한다.
    """

    def term_resolver_node(state: dict[str, Any]) -> dict[str, Any]:
        user_request = state["messages"][-1].content

        # 1. 지식 베이스에서 관련 정보 검색
        kb_context = ""
        if knowledge_base is not None and knowledge_base.is_loaded:
            chunks = knowledge_base.search(user_request, top_k=5)
            if chunks:
                kb_context = "지식 베이스 검색 결과:\n" + "\n---\n".join(
                    f"[{c.source}] {c.text[:300]}" for c in chunks
                )

        # 2. LLM에게 용어 확인 요청 (간결한 결과 요구)
        prompt = (
            "아래 요청에 등장하는 고유명사를 각각 한 줄로 정리하라.\n\n"
            "형식: 용어 → 정식 명칭 (개발사/제조사, 카테고리)\n"
            "규칙:\n"
            "- 참고 자료가 있으면 반드시 그 내용에 기반하라.\n"
            "- 확실하지 않으면 '확인 필요'로 표시하고 추측하지 마라.\n"
            "- 다른 제품/게임으로 대체하지 마라.\n"
            "- 500자 이내로 간결하게 답하라.\n"
        )
        if kb_context:
            prompt += f"\n{kb_context}\n"
        prompt += f"\n요청: {user_request}"

        result = resolver_model.invoke(prompt)
        term_context = result.content

        return {"term_context": term_context}

    return term_resolver_node


def make_planner_node(planner_model: BaseChatModel):
    """1단계: 사용자 요청을 구조화된 실행 계획으로 변환."""

    def planner_node(state: dict[str, Any]) -> dict[str, Any]:
        user_request = state["messages"][-1].content
        term_context = state.get("term_context", "")
        # structured output 보호: term_context가 너무 길면 축약
        if len(term_context) > 800:
            term_context = term_context[:800] + "..."

        plan = planner_model.with_structured_output(ExecutionPlan).invoke(
            f"{TOPIC_PRESERVATION_RULE}\n\n"
            f"용어 확인 결과:\n{term_context}\n\n"
            f"다음 요청을 독립적인 하위 작업으로 분할하라. "
            f"요청에 언급된 고유명사(게임명, 제품명, 기술명 등)를 "
            f"그대로 유지하라:\n{user_request}"
        )
        return {
            "execution_plan": plan,
            "plan": [t.model_dump() for t in plan.sub_tasks],
        }

    return planner_node


def make_storm_worker(worker_model: BaseChatModel, knowledge_base=None):
    """2단계: STORM 마이크로 루프 워커. 다중 페르소나 탐색 + 압축.

    Args:
        worker_model: 워커 LLM 모델.
        knowledge_base: LocalKnowledgeBase 인스턴스 (선택). 제공되면 작업과
            관련된 자료를 검색하여 프롬프트 컨텍스트에 삽입한다.
    """

    def storm_worker(state: dict[str, Any]) -> dict[str, Any]:
        task = state["task"]
        task_desc = task.description if hasattr(task, "description") else str(task)
        task_id = task.id if hasattr(task, "id") else "unknown"

        # 원본 쿼리 + 용어 컨텍스트 추출
        original_query = ""
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1] if isinstance(msgs, list) else msgs
            original_query = last.content if hasattr(last, "content") else str(last)
        term_context = state.get("term_context", "")

        # 지식 베이스에서 관련 자료 검색 (로컬 모델 컨텍스트 보호)
        context = ""
        if knowledge_base is not None and knowledge_base.is_loaded:
            context = knowledge_base.get_context_for_task(task_desc, max_chars=800)

        # 배경 컨텍스트 구성 (한 번만, 간결하게 — 로컬 모델 컨텍스트 보호)
        bg = f"{TOPIC_PRESERVATION_RULE}\n"
        if term_context:
            bg += f"용어: {term_context[:300]}\n"
        if context:
            bg += f"\n{context[:800]}\n"

        findings: list[str] = []
        for persona in STORM_PERSONAS:
            # 이전 발견은 마지막 것만 전달 (컨텍스트 절약)
            prev = findings[-1][:300] if findings else "없음"
            prompt = (
                f"{bg}\n"
                f"[{persona}] 작업: {task_desc}\n"
                f"이전 발견 요약: {prev}"
            )
            result = worker_model.invoke(prompt)
            findings.append(result.content)

        # 압축
        all_findings = "\n---\n".join(f[:500] for f in findings)
        compress_prompt = (
            f"{TOPIC_PRESERVATION_RULE}\n"
            f"핵심 사실만 200토큰 이내로 압축:\n{all_findings}"
        )
        compressed = worker_model.invoke(compress_prompt)
        return {
            "worker_results": [{"task_id": task_id, "data": compressed.content}]
        }

    return storm_worker


def make_synthesis_node(synthesizer_model: BaseChatModel, knowledge_base=None):
    """초안 합성: 워커 결과를 종합 보고서로 병합.

    Args:
        synthesizer_model: 합성 LLM 모델.
        knowledge_base: LocalKnowledgeBase 인스턴스 (선택). 제공되면
            참고 자료에 기반하여 인용을 포함하도록 지시한다.
    """

    def synthesis_node(state: dict[str, Any]) -> dict[str, Any]:
        # 원본 쿼리 추출
        original_query = ""
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1] if isinstance(msgs, list) else msgs
            original_query = last.content if hasattr(last, "content") else str(last)

        results_text = "\n".join(
            f"[{r['task_id']}]: {r['data']}" for r in state["worker_results"]
        )
        prompt = (
            f"{TOPIC_PRESERVATION_RULE}\n"
            f"원본 요청: {original_query}\n\n"
            f"다음 조사 데이터를 종합하여 보고서를 작성하라.\n"
            f"규칙:\n"
            f"- 인용 태그([출처:...], [task_...] 등)는 사용하지 마라.\n"
            f"- 사실적 정보만 자연스러운 문장으로 작성하라.\n"
            f"- 원본 요청의 주제와 다른 내용은 무시하라.\n"
            f"- 1500자 이내로 간결하게 작성하라.\n\n"
            f"{results_text}"
        )
        # 최대 2회 시도 (thinking 모델이 빈 응답을 반환하는 경우 대비)
        for attempt in range(2):
            draft = synthesizer_model.invoke(prompt)
            if draft.content.strip():
                return {"draft": draft.content}
            # 재시도: 더 직접적인 프롬프트
            prompt = (
                f"다음 데이터를 한국어 보고서로 작성하라. "
                f"반드시 텍스트 내용을 출력하라:\n{results_text[:2000]}"
            )
        return {"draft": draft.content}

    return synthesis_node


def make_cove_plan_node(verifier_model: BaseChatModel, max_questions: int = 5):
    """3단계-1: 초안에서 검증 질문 목록을 생성."""

    def cove_plan_node(state: dict[str, Any]) -> dict[str, Any]:
        draft = state.get("draft", "")
        # 빈 초안이면 검증 건너뛰기
        if not draft or not draft.strip():
            return {"verification_questions": []}

        try:
            questions_text = verifier_model.invoke(
                f"다음 초안의 핵심 사실적 주장을 검증할 질문을 최대 {max_questions}개만 생성하라.\n"
                f"규칙:\n"
                f"- 구체적 날짜가 언급된 경우 반드시 '해당 날짜에 실제로 그 이벤트가 있었는가?' 질문을 포함\n"
                f"- 특정 버전/모델 번호와 이벤트의 연관이 정확한지 확인\n"
                f"- 가장 중요한 수치와 사실만 검증\n\n"
                f"{draft[:3000]}"
            )
            questions = parse_questions(questions_text.content)[:max_questions]
        except Exception:
            # 400 에러 등 발생 시 기본 질문으로 대체
            questions = []
        return {"verification_questions": questions}

    return cove_plan_node


def make_factored_verifier(verifier_model: BaseChatModel):
    """3단계-2: 초안 컨텍스트 없이 독립적으로 검증 (편향 격리)."""

    def factored_verifier(state: dict[str, Any]) -> dict[str, Any]:
        term_ctx = state.get("term_context", "")
        prompt = ""
        if term_ctx:
            prompt += f"참고 용어 정보:\n{term_ctx[:300]}\n\n"
        prompt += f"다음 질문에 사실에 기반하여 답변:\n{state['question']}"
        answer = verifier_model.invoke(prompt)

        answer_text = answer.content.strip()
        # 빈 답변이면 UNVERIFIED로 표시하여 cross_check에서 처리 가능하게 함
        if not answer_text:
            answer_text = "[UNVERIFIED] 이 질문에 대한 검증 답변을 생성하지 못했음. 해당 사실을 재확인 필요."

        return {
            "verification_answers": [{
                "question": state["question"],
                "answer": answer_text,
            }]
        }

    return factored_verifier


def make_cross_check_node(verifier_model: BaseChatModel):
    """3단계-3: 검증 답변과 초안을 교차 대조."""

    def cross_check_node(state: dict[str, Any]) -> dict[str, Any]:
        original_draft = state.get("draft", "")

        # 원본 쿼리 추출 (첫 번째 HumanMessage 사용)
        original_query = ""
        msgs = state.get("messages", [])
        for m in msgs:
            if hasattr(m, "type") and m.type == "human":
                original_query = m.content
                break
        if not original_query and msgs:
            last = msgs[0] if isinstance(msgs, list) else msgs
            original_query = last.content if hasattr(last, "content") else str(last)

        # 검증 답변을 텍스트로 변환 (dict 리스트 → 읽기 쉬운 형식)
        va = state.get("verification_answers", [])
        va_text = "\n".join(
            f"Q: {a.get('question', '')}\nA: {a.get('answer', '')[:300]}"
            for a in va
        ) if va else "검증 결과 없음"

        # UNVERIFIED 답변이 있으면 추가 경고 삽입
        unverified_warning = ""
        if va:
            unverified = [a for a in va if "[UNVERIFIED]" in a.get("answer", "")]
            if unverified:
                questions = ", ".join(a.get("question", "")[:60] for a in unverified)
                unverified_warning = (
                    f"\n[경고] 다음 질문에 대해 검증이 불가능했다: {questions}\n"
                    f"해당 사실에 관련된 구체적 날짜, 버전 번호, 이벤트명이 정확한지 "
                    f"확인할 수 없으므로, 확실하지 않은 구체적 날짜/버전은 "
                    f"제거하거나 '정확한 날짜 확인 필요' 등으로 표시하라.\n"
                )

        verified_draft = verifier_model.invoke(
            f"{TOPIC_PRESERVATION_RULE}\n"
            f"원본 요청: {original_query}\n\n"
            f"초안:\n{original_draft}\n\n"
            f"검증 결과:\n{va_text}\n"
            f"{unverified_warning}\n"
            f"불일치 사항이 있으면 초안을 수정하라. "
            f"특히 원본 요청의 주제와 다른 주제로 대체된 부분이 있으면 반드시 수정하라. "
            f"검증되지 않은 구체적 날짜나 버전 번호는 제거하거나 불확실성을 명시하라. "
            f"수정된 초안 전체를 출력하라."
        )
        # 빈/축소 응답 방어: LLM이 빈 내용이나 극단적으로 짧은 응답을 반환하면
        # 원본 초안을 보존한다. (thinking 모델이 응답을 <think> 안에만 넣는 경우)
        result = verified_draft.content.strip()
        if not result or len(result) < len(original_draft) * 0.3:
            return {"draft": original_draft}
        return {"draft": result}

    return cross_check_node


def make_sanitizer_node(sanitizer: DeterministicSanitizer):
    """4단계-1: 결정론적 검증 (LLM 미사용)."""

    def sanitizer_node(state: dict[str, Any]) -> dict[str, Any]:
        draft = state.get("draft", "")
        errors = sanitizer.validate_as_strings(draft) if draft else []
        current_count = state.get("correction_count", 0)
        if errors:
            return {
                "sanitizer_errors": errors,
                "error_log": [f"SANITIZER_FAIL: {len(errors)} errors"],
                "correction_count": current_count + 1,
            }
        return {
            "sanitizer_errors": [],
            "correction_count": current_count,
        }

    return sanitizer_node


def make_self_correction_node(synthesizer_model: BaseChatModel):
    """4단계-2: Sanitizer 오류 기반 자기 교정."""

    def self_correction_node(state: dict[str, Any]) -> dict[str, Any]:
        errors = state["sanitizer_errors"]
        draft = state.get("draft", "")

        # 에러가 너무 많으면 인용 형식 문제 → 인용 제거 + 본문 보존
        if len(errors) > 10:
            error_sample = "\n".join(f"- {e}" for e in errors[:5])
            corrected = synthesizer_model.invoke(
                f"초안에서 검증되지 않은 인용구([출처:...], [task_...] 등)를 "
                f"모두 제거하고, 본문 내용은 최대한 보존하여 다시 작성하라. "
                f"오류 예시:\n{error_sample}\n\n"
                f"현재 초안:\n{draft[:3000]}"
            )
        else:
            error_msg = "\n".join(f"- {e}" for e in errors)
            corrected = synthesizer_model.invoke(
                f"다음 오류를 수정하여 초안을 다시 작성하라. "
                f"본문 내용은 보존하고 잘못된 부분만 수정:\n"
                f"{error_msg}\n\n현재 초안:\n{draft[:3000]}"
            )
        return {"draft": corrected.content, "sanitizer_errors": []}

    return self_correction_node


def finalize_node(state: dict[str, Any]) -> dict[str, Any]:
    """최종 출력: 초안을 AIMessage로 래핑."""
    return {"messages": [AIMessage(content=state.get("draft", ""))]}


# ── 라우팅 함수 ──


def dispatch_storm_workers(state: dict[str, Any]) -> list[Send]:
    """실행 계획의 하위 작업별로 STORM 워커를 동적 스폰한다."""
    plan = state["execution_plan"]
    if not plan or not plan.sub_tasks:
        return [Send("synthesis", {})]
    # 원본 메시지와 용어 컨텍스트를 워커에 전달
    msgs = state.get("messages", [])
    term_ctx = state.get("term_context", "")
    return [
        Send("storm_worker", {
            "task": task, "messages": msgs, "term_context": term_ctx,
        })
        for task in plan.sub_tasks
    ]


def dispatch_verifiers(state: dict[str, Any]) -> list[Send]:
    """검증 질문별로 독립 검증기를 동적 스폰한다."""
    questions = state.get("verification_questions", [])
    if not questions:
        return [Send("cross_check", {})]
    # term_context를 검증기에도 전달하여 용어 혼동 방지
    term_ctx = state.get("term_context", "")
    return [
        Send("factored_verifier", {
            "question": q, "draft_context": None, "term_context": term_ctx,
        })
        for q in questions
    ]


def route_after_sanitizer(state: dict[str, Any]) -> str:
    """Sanitizer 결과에 따라 자기 교정 또는 최종 출력으로 분기."""
    if state.get("sanitizer_errors") and state.get("correction_count", 0) < MAX_CORRECTIONS:
        return "self_correct"
    return "finalize"


# ── 그래프 빌더 ──


def build_zero_hallucination_pipeline(
    planner_model: BaseChatModel,
    worker_model: BaseChatModel,
    verifier_model: BaseChatModel,
    synthesizer_model: BaseChatModel,
    chunk_db: LocalChunkDB,
    config: HarnessConfig | None = None,
    max_questions: int = 5,
    knowledge_base=None,
) -> StateGraph:
    """무결성 보장 파이프라인 그래프를 생성한다.

    Args:
        planner_model: 실행 계획 생성 모델.
        worker_model: STORM 워커 모델.
        verifier_model: CoVe 검증 모델.
        synthesizer_model: 초안 합성/교정 모델.
        chunk_db: 인용 검증용 로컬 청크 DB.
        config: 하네스 설정 (선택).
        max_questions: CoVe 검증 질문 최대 수.
        knowledge_base: LocalKnowledgeBase 인스턴스 (선택).
            제공되면 Worker와 Synthesizer가 자료 컨텍스트를 활용한다.

    Returns:
        컴파일 가능한 StateGraph.
    """
    config = config or HarnessConfig()
    sanitizer = DeterministicSanitizer(chunk_db=chunk_db)

    graph = StateGraph(ZeroHallucinationState)

    # 노드 등록
    graph.add_node("term_resolver", make_term_resolver_node(
        planner_model, knowledge_base,
    ))
    graph.add_node("planner", make_planner_node(planner_model))
    graph.add_node("storm_worker", make_storm_worker(worker_model, knowledge_base))
    graph.add_node("synthesis", make_synthesis_node(synthesizer_model, knowledge_base))
    graph.add_node("cove_plan", make_cove_plan_node(verifier_model, max_questions))
    graph.add_node("factored_verifier", make_factored_verifier(verifier_model))
    graph.add_node("cross_check", make_cross_check_node(verifier_model))
    graph.add_node("sanitizer", make_sanitizer_node(sanitizer))
    graph.add_node("self_correct", make_self_correction_node(synthesizer_model))
    graph.add_node("finalize", finalize_node)

    # 엣지 연결
    graph.set_entry_point("term_resolver")
    graph.add_edge("term_resolver", "planner")
    graph.add_conditional_edges("planner", dispatch_storm_workers)
    graph.add_edge("storm_worker", "synthesis")
    graph.add_edge("synthesis", "cove_plan")
    graph.add_conditional_edges("cove_plan", dispatch_verifiers)
    graph.add_edge("factored_verifier", "cross_check")
    graph.add_edge("cross_check", "sanitizer")
    graph.add_conditional_edges("sanitizer", route_after_sanitizer, {
        "self_correct": "self_correct",
        "finalize": "finalize",
    })
    graph.add_edge("self_correct", "sanitizer")
    graph.add_edge("finalize", END)

    return graph
