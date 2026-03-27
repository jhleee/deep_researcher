"""LM Studio 실제 LLM을 사용한 E2E 통합 테스트.

로컬 LM Studio 서버의 qwen/qwen3.5-9b 모델로
Zero Hallucination Pipeline 전체를 실행한다.

LLM 호출 수 최적화:
- Planner는 FakeModel로 고정 (subtask 2개) → 워커 호출 8회로 제한
- CoVe 검증 질문은 LLM이 생성하되, 자연스러운 수(3~5개)를 기대
- 전체 E2E 예상 호출: ~15회 (약 5~8분)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from architectures.zero_hallucination import (
    build_zero_hallucination_pipeline,
    make_cove_plan_node,
    make_cross_check_node,
    make_factored_verifier,
    make_planner_node,
    make_sanitizer_node,
    make_self_correction_node,
    make_storm_worker,
    make_synthesis_node,
)
from harness.lmstudio import ThinkingModelWrapper, create_lmstudio_llm
from harness.models import ExecutionPlan, FakeStructuredChatModel, SubTask, parse_questions
from harness.sanitizer import DeterministicSanitizer, LocalChunkDB

DATASETS_DIR = Path(__file__).parent / "datasets"


def _load_topics() -> dict:
    return json.loads((DATASETS_DIR / "research_topics.json").read_text(encoding="utf-8"))


def _is_lmstudio_available() -> bool:
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://169.254.83.107:1234/v1/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_lmstudio_available(),
    reason="LM Studio 서버 미연결 (http://169.254.83.107:1234)",
)


@pytest.fixture(scope="module")
def llm() -> ThinkingModelWrapper:
    return ThinkingModelWrapper(create_lmstudio_llm(temperature=0.3))


@pytest.fixture(scope="module")
def topics_data() -> dict:
    return _load_topics()


def _make_chunk_db(topic: dict, tmp_path: Path) -> LocalChunkDB:
    db = LocalChunkDB(db_path=tmp_path / f"{topic['id']}_chunks.json")
    for chunk in topic["chunks"]:
        db.add_chunk(chunk["text"], source=chunk["source"])
    for url in topic["urls"]:
        db.add_url(url)
    for num in topic["numbers"]:
        db.add_number(str(num))
    db.save()
    return db


def _make_zh_initial_state(query: str) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "plan": [],
        "artifacts": [],
        "metadata": {},
        "error_log": [],
        "iteration_count": 0,
        "execution_plan": None,
        "worker_results": [],
        "draft": None,
        "verification_questions": [],
        "verification_answers": [],
        "sanitizer_errors": [],
        "correction_count": 0,
    }


# ── 개별 노드 실제 LLM 테스트 ──


class TestPlannerNodeLive:
    """Planner 노드: 실제 LLM으로 실행 계획 생성."""

    def test_generates_valid_plan(self, llm):
        planner = make_planner_node(llm)
        state = {"messages": [HumanMessage(content="기후변화가 한국 경제에 미치는 영향을 분석하라")]}
        result = planner(state)

        assert "execution_plan" in result
        plan = result["execution_plan"]
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.goal) > 0
        assert len(plan.sub_tasks) >= 2
        print(f"\n[Planner] Goal: {plan.goal}")
        for t in plan.sub_tasks:
            print(f"  {t.id}: {t.description}")


class TestStormWorkerLive:
    """STORM 워커: 실제 LLM으로 다중 페르소나 탐색."""

    def test_produces_compressed_result(self, llm):
        worker = make_storm_worker(llm)
        task = SubTask(id="t1", description="기후변화가 한국 농업에 미치는 영향 분석")
        result = worker({"task": task})

        assert "worker_results" in result
        assert len(result["worker_results"]) == 1
        data = result["worker_results"][0]
        assert data["task_id"] == "t1"
        assert len(data["data"]) > 10
        print(f"\n[Worker] Compressed ({len(data['data'])} chars): {data['data'][:200]}...")


class TestSynthesisNodeLive:
    """Synthesis 노드: 실제 LLM으로 워커 결과 종합."""

    def test_generates_draft(self, llm):
        synthesis = make_synthesis_node(llm)
        state = {
            "worker_results": [
                {"task_id": "t1", "data": "한국의 탄소 배출량은 2023년 기준 약 6.1억 톤이다. (환경부 2023)"},
                {"task_id": "t2", "data": "기후변화로 인한 경제적 손실은 연간 약 7조 원이다. (한국환경연구원)"},
            ]
        }
        result = synthesis(state)

        assert "draft" in result
        assert len(result["draft"]) > 50
        print(f"\n[Synthesis] Draft ({len(result['draft'])} chars): {result['draft'][:300]}...")


class TestCovePlanNodeLive:
    """CoVe Plan 노드: 실제 LLM으로 검증 질문 생성."""

    def test_generates_verification_questions(self, llm):
        cove_plan = make_cove_plan_node(llm)
        state = {
            "draft": (
                "한국의 탄소 배출량은 2023년 기준 약 6.1억 톤이다. "
                "기후변화로 인한 경제적 손실은 연간 약 7조 원으로 추산된다. "
                "한국은 2050 탄소중립을 선언하였다."
            )
        }
        result = cove_plan(state)

        questions = result["verification_questions"]
        assert len(questions) >= 1
        print(f"\n[CoVe] Questions ({len(questions)}):")
        for q in questions[:5]:
            print(f"  - {q[:100]}")


class TestCrossCheckNodeLive:
    """Cross Check 노드: 검증 결과와 초안 교차 대조."""

    def test_updates_draft_with_verification(self, llm):
        cross_check = make_cross_check_node(llm)
        result = cross_check({
            "draft": "한국의 탄소 배출량은 약 10억 톤이다. 경제 손실은 50조 원이다.",
            "verification_answers": [
                {"question": "탄소 배출량이 10억 톤인가?", "answer": "아니요, 6.1억 톤입니다."},
                {"question": "경제 손실이 50조 원인가?", "answer": "아니요, 약 7조 원입니다."},
            ],
        })

        assert "draft" in result
        assert len(result["draft"]) > 10
        print(f"\n[CrossCheck] Updated draft: {result['draft'][:300]}...")


class TestSelfCorrectionNodeLive:
    """Self Correction 노드: sanitizer 오류 기반 초안 교정."""

    def test_corrects_errors(self, llm):
        node = make_self_correction_node(llm)
        result = node({
            "draft": "한국의 탄소 배출량은 약 10억 톤이다. [출처: 가짜 출처] 경제 손실은 $500이다.",
            "sanitizer_errors": [
                "[FAKE_CITATION] '가짜 출처'는 원본에 존재하지 않음.",
                "[NUMBER_UNVERIFIED] 수치 '500'의 출처를 확인할 수 없음.",
            ],
        })

        assert "draft" in result
        assert result["sanitizer_errors"] == []
        print(f"\n[SelfCorrection] Corrected: {result['draft'][:300]}...")


# ── 전체 파이프라인 E2E 테스트 ──


class TestFullPipelineLive:
    """전체 Zero Hallucination Pipeline E2E.

    Planner는 FakeModel로 고정하여 호출 횟수를 제어한다.
    나머지 노드(Worker, Synthesis, Verifier, CrossCheck, Sanitizer, SelfCorrect)는 실제 LLM.
    """

    def _build_fixed_plan_pipeline(
        self, llm, chunk_db, subtask_count: int = 2, query: str = "분석 요청"
    ):
        """Planner를 고정하고 나머지는 실제 LLM을 사용하는 파이프라인."""
        subtasks = [
            SubTask(id=f"t{i+1}", description=f"하위작업 {i+1}")
            for i in range(subtask_count)
        ]
        plan = ExecutionPlan(goal=query, sub_tasks=subtasks)
        planner_model = FakeStructuredChatModel(responses=[plan.model_dump_json()])

        return build_zero_hallucination_pipeline(
            planner_model=planner_model,
            worker_model=llm,
            verifier_model=llm,
            synthesizer_model=llm,
            chunk_db=chunk_db,
            max_questions=3,
        )

    def test_climate_e2e(self, llm, tmp_path, topics_data):
        """기후변화 주제 E2E: 2 subtask → 탐색 → 합성 → 검증 → 정제."""
        topic = next(t for t in topics_data["topics"] if t["id"] == "climate_economy")
        chunk_db = _make_chunk_db(topic, tmp_path)

        # 구체적 subtask 지정
        subtasks = [
            SubTask(id="t1", description="한국의 탄소 배출 현황과 기후변화 추세 분석"),
            SubTask(id="t2", description="기후변화로 인한 경제적 손실과 영향 분석"),
        ]
        plan = ExecutionPlan(goal=topic["query"], sub_tasks=subtasks)
        planner_model = FakeStructuredChatModel(responses=[plan.model_dump_json()])

        graph = build_zero_hallucination_pipeline(
            planner_model=planner_model,
            worker_model=llm,
            verifier_model=llm,
            synthesizer_model=llm,
            chunk_db=chunk_db,
            max_questions=3,
        )

        compiled = graph.compile(checkpointer=MemorySaver())
        result = compiled.invoke(
            _make_zh_initial_state(topic["query"]),
            config={"configurable": {"thread_id": "live-climate-e2e"}},
        )

        # 최종 AIMessage 존재
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert len(last_msg.content) > 50

        # 워커 결과 수집됨
        assert len(result["worker_results"]) >= 2

        # 초안 존재
        assert result["draft"] is not None

        print(f"\n{'='*60}")
        print(f"[E2E Climate] Workers: {len(result['worker_results'])}")
        print(f"[E2E Climate] Corrections: {result['correction_count']}")
        print(f"[E2E Climate] Sanitizer Errors: {result['sanitizer_errors']}")
        print(f"[E2E Climate] Error Log: {result['error_log']}")
        print(f"[E2E Climate] Final ({len(last_msg.content)} chars):")
        print(last_msg.content[:500])
        print(f"{'='*60}")

    def test_semiconductor_e2e(self, llm, tmp_path, topics_data):
        """반도체 주제 E2E."""
        topic = next(t for t in topics_data["topics"] if t["id"] == "semiconductor")
        chunk_db = _make_chunk_db(topic, tmp_path)

        subtasks = [
            SubTask(id="t1", description="삼성전자와 SK하이닉스의 글로벌 경쟁력 분석"),
            SubTask(id="t2", description="미국 CHIPS Act와 한국 반도체 정책 비교"),
        ]
        plan = ExecutionPlan(goal=topic["query"], sub_tasks=subtasks)
        planner_model = FakeStructuredChatModel(responses=[plan.model_dump_json()])

        graph = build_zero_hallucination_pipeline(
            planner_model=planner_model,
            worker_model=llm,
            verifier_model=llm,
            synthesizer_model=llm,
            chunk_db=chunk_db,
            max_questions=3,
        )

        compiled = graph.compile(checkpointer=MemorySaver())
        result = compiled.invoke(
            _make_zh_initial_state(topic["query"]),
            config={"configurable": {"thread_id": "live-semi-e2e"}},
        )

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert len(last_msg.content) > 50

        print(f"\n{'='*60}")
        print(f"[E2E Semi] Workers: {len(result['worker_results'])}")
        print(f"[E2E Semi] Corrections: {result['correction_count']}")
        print(f"[E2E Semi] Final ({len(last_msg.content)} chars):")
        print(last_msg.content[:500])
        print(f"{'='*60}")

    def test_full_live_planner_e2e(self, llm, tmp_path, topics_data):
        """Planner도 실제 LLM을 사용하는 완전 E2E (바이오테크, 가장 작은 주제)."""
        topic = next(t for t in topics_data["topics"] if t["id"] == "biotech")
        chunk_db = _make_chunk_db(topic, tmp_path)

        graph = build_zero_hallucination_pipeline(
            planner_model=llm,
            worker_model=llm,
            verifier_model=llm,
            synthesizer_model=llm,
            chunk_db=chunk_db,
            max_questions=3,
        )

        compiled = graph.compile(checkpointer=MemorySaver())
        result = compiled.invoke(
            _make_zh_initial_state(topic["query"]),
            config={"configurable": {"thread_id": "live-full-e2e"}},
        )

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert len(last_msg.content) > 50
        assert result["execution_plan"] is not None

        print(f"\n{'='*60}")
        print(f"[Full E2E] Plan: {result['execution_plan'].goal}")
        print(f"[Full E2E] Subtasks: {len(result['execution_plan'].sub_tasks)}")
        print(f"[Full E2E] Workers: {len(result['worker_results'])}")
        print(f"[Full E2E] Corrections: {result['correction_count']}")
        print(f"[Full E2E] Final ({len(last_msg.content)} chars):")
        print(last_msg.content[:500])
        print(f"{'='*60}")
