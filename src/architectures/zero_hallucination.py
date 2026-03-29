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


def make_planner_node(planner_model: BaseChatModel):
    """1단계: 사용자 요청을 구조화된 실행 계획으로 변환."""

    def planner_node(state: dict[str, Any]) -> dict[str, Any]:
        user_request = state["messages"][-1].content
        plan = planner_model.with_structured_output(ExecutionPlan).invoke(
            f"{TOPIC_PRESERVATION_RULE}\n\n"
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

        # 원본 쿼리 추출 (주제 보존용)
        original_query = ""
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1] if isinstance(msgs, list) else msgs
            original_query = last.content if hasattr(last, "content") else str(last)

        # 지식 베이스에서 관련 자료 검색
        context = ""
        if knowledge_base is not None and knowledge_base.is_loaded:
            context = knowledge_base.get_context_for_task(task_desc)

        findings: list[str] = []
        for persona in STORM_PERSONAS:
            prompt = (
                f"{TOPIC_PRESERVATION_RULE}\n"
                f"원본 요청: {original_query}\n\n"
                f"[{persona}] 작업: {task_desc}\n이전 발견: {findings}"
            )
            if context:
                prompt = f"{context}\n\n{prompt}"
            result = worker_model.invoke(prompt)
            findings.append(result.content)

        # 압축: 핵심 사실만 추출
        compress_prompt = (
            f"{TOPIC_PRESERVATION_RULE}\n"
            f"원본 요청: {original_query}\n\n"
            f"핵심 사실만 200토큰 이내로 압축:\n{''.join(findings)}"
        )
        if context:
            compress_prompt = (
                f"다음 참고 자료의 내용에 기반하여 압축하라. "
                f"자료에 없는 내용은 포함하지 마라.\n{context}\n\n"
                f"{compress_prompt}"
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
        source_instruction = ""
        if knowledge_base is not None and knowledge_base.is_loaded:
            source_instruction = (
                "참고 자료에서 확인된 내용만 포함하고, "
                "각 주장에 [출처: 파일명] 형식으로 출처를 명시하라. "
                "자료에 없는 내용은 작성하지 마라.\n\n"
            )
        else:
            source_instruction = "모든 주장에 반드시 [출처]를 명시:\n"

        draft = synthesizer_model.invoke(
            f"{TOPIC_PRESERVATION_RULE}\n"
            f"원본 요청: {original_query}\n\n"
            f"다음 데이터를 종합하여 보고서 초안을 작성하라. "
            f"워커 데이터에서 원본 요청의 주제와 다른 내용이 있으면 무시하라. "
            f"{source_instruction}{results_text}"
        )
        return {"draft": draft.content}

    return synthesis_node


def make_cove_plan_node(verifier_model: BaseChatModel, max_questions: int = 5):
    """3단계-1: 초안에서 검증 질문 목록을 생성."""

    def cove_plan_node(state: dict[str, Any]) -> dict[str, Any]:
        questions_text = verifier_model.invoke(
            f"다음 초안의 핵심 사실적 주장을 검증할 질문을 최대 {max_questions}개만 생성하라. "
            f"가장 중요한 수치와 사실만 검증:\n"
            f"{state['draft']}"
        )
        questions = parse_questions(questions_text.content)[:max_questions]
        return {"verification_questions": questions}

    return cove_plan_node


def make_factored_verifier(verifier_model: BaseChatModel):
    """3단계-2: 초안 컨텍스트 없이 독립적으로 검증 (편향 격리)."""

    def factored_verifier(state: dict[str, Any]) -> dict[str, Any]:
        answer = verifier_model.invoke(
            f"다음 질문에 사실에 기반하여 답변:\n{state['question']}"
        )
        return {
            "verification_answers": [{
                "question": state["question"],
                "answer": answer.content,
            }]
        }

    return factored_verifier


def make_cross_check_node(verifier_model: BaseChatModel):
    """3단계-3: 검증 답변과 초안을 교차 대조."""

    def cross_check_node(state: dict[str, Any]) -> dict[str, Any]:
        # 원본 쿼리 추출
        original_query = ""
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1] if isinstance(msgs, list) else msgs
            original_query = last.content if hasattr(last, "content") else str(last)

        verified_draft = verifier_model.invoke(
            f"{TOPIC_PRESERVATION_RULE}\n"
            f"원본 요청: {original_query}\n\n"
            f"초안:\n{state['draft']}\n\n"
            f"검증 결과:\n{state['verification_answers']}\n\n"
            f"불일치 사항이 있으면 초안을 수정하라. "
            f"특히 원본 요청의 주제와 다른 주제로 대체된 부분이 있으면 반드시 수정하라."
        )
        return {"draft": verified_draft.content}

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
        error_msg = "\n".join(f"오류: {e}" for e in state["sanitizer_errors"])
        corrected = synthesizer_model.invoke(
            f"다음 오류를 수정하여 초안을 다시 작성:\n{error_msg}\n"
            f"현재 초안:\n{state['draft']}"
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
    # 원본 메시지를 워커에 전달하여 주제 보존
    msgs = state.get("messages", [])
    return [
        Send("storm_worker", {"task": task, "messages": msgs})
        for task in plan.sub_tasks
    ]


def dispatch_verifiers(state: dict[str, Any]) -> list[Send]:
    """검증 질문별로 독립 검증기를 동적 스폰한다."""
    questions = state.get("verification_questions", [])
    if not questions:
        return [Send("cross_check", {})]
    return [
        Send("factored_verifier", {"question": q, "draft_context": None})
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
    graph.set_entry_point("planner")
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
