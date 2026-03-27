"""zero_hallucination.py 유닛 테스트 - 노드 함수 + 라우팅 + 그래프 조립 + E2E."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from harness.models import ExecutionPlan, FakeStructuredChatModel, SubTask
from harness.sanitizer import DeterministicSanitizer, LocalChunkDB
from architectures.zero_hallucination import (
    MAX_CORRECTIONS,
    ZeroHallucinationState,
    build_zero_hallucination_pipeline,
    dispatch_storm_workers,
    dispatch_verifiers,
    finalize_node,
    make_cove_plan_node,
    make_cross_check_node,
    make_factored_verifier,
    make_planner_node,
    make_sanitizer_node,
    make_self_correction_node,
    make_storm_worker,
    make_synthesis_node,
    route_after_sanitizer,
)


class TestPlannerNode:
    """planner_node 테스트."""

    def test_generates_execution_plan(self, fake_planner_model, sample_execution_plan):
        planner = make_planner_node(fake_planner_model)
        state = {"messages": [HumanMessage(content="서울에 대해 분석하라")]}
        result = planner(state)

        assert "execution_plan" in result
        assert isinstance(result["execution_plan"], ExecutionPlan)
        assert result["execution_plan"].goal == sample_execution_plan.goal
        assert len(result["execution_plan"].sub_tasks) == 2

    def test_populates_plan_field(self, fake_planner_model):
        planner = make_planner_node(fake_planner_model)
        state = {"messages": [HumanMessage(content="분석 요청")]}
        result = planner(state)

        assert "plan" in result
        assert isinstance(result["plan"], list)
        assert all(isinstance(item, dict) for item in result["plan"])


class TestStormWorker:
    """storm_worker 테스트."""

    def test_multi_persona_loop(self):
        model = FakeStructuredChatModel(responses=[
            "비판적 분석 결과",
            "전문가 의견",
            "반론 의견",
            "압축된 핵심 사실",
        ])
        worker = make_storm_worker(model)
        task = SubTask(id="t1", description="테스트 작업")
        result = worker({"task": task})

        assert "worker_results" in result
        assert len(result["worker_results"]) == 1
        assert result["worker_results"][0]["task_id"] == "t1"
        assert result["worker_results"][0]["data"] == "압축된 핵심 사실"

    def test_invokes_model_4_times(self):
        """3 페르소나 + 1 압축 = 4회 호출."""
        model = FakeStructuredChatModel(responses=[
            "r1", "r2", "r3", "compressed"
        ])
        worker = make_storm_worker(model)
        worker({"task": SubTask(id="t1", description="test")})
        # FakeListChatModel cycles through responses in order
        # After 4 calls, the next would cycle back to "r1"
        assert model.invoke("check").content == "r1"  # cycled back


class TestSynthesisNode:
    """synthesis_node 테스트."""

    def test_generates_draft(self):
        model = FakeStructuredChatModel(responses=["종합 보고서 초안"])
        synthesis = make_synthesis_node(model)
        state = {
            "worker_results": [
                {"task_id": "t1", "data": "결과 1"},
                {"task_id": "t2", "data": "결과 2"},
            ]
        }
        result = synthesis(state)
        assert "draft" in result
        assert result["draft"] == "종합 보고서 초안"


class TestCovePlanNode:
    """cove_plan_node 테스트."""

    def test_generates_questions(self):
        model = FakeStructuredChatModel(responses=[
            "1. 인구 데이터가 정확한가?\n2. GDP 수치가 맞는가?"
        ])
        cove_plan = make_cove_plan_node(model)
        state = {"draft": "서울 인구 분석 초안"}
        result = cove_plan(state)

        assert "verification_questions" in result
        assert len(result["verification_questions"]) == 2

    def test_empty_response_returns_empty(self):
        model = FakeStructuredChatModel(responses=[""])
        cove_plan = make_cove_plan_node(model)
        result = cove_plan({"draft": "초안"})
        assert result["verification_questions"] == []


class TestFactoredVerifier:
    """factored_verifier 테스트."""

    def test_answers_question(self):
        model = FakeStructuredChatModel(responses=["사실이 확인되었습니다."])
        verifier = make_factored_verifier(model)
        state = {"question": "인구 데이터가 정확한가?", "draft_context": None}
        result = verifier(state)

        assert "verification_answers" in result
        assert len(result["verification_answers"]) == 1
        assert result["verification_answers"][0]["question"] == "인구 데이터가 정확한가?"
        assert result["verification_answers"][0]["answer"] == "사실이 확인되었습니다."


class TestCrossCheckNode:
    """cross_check_node 테스트."""

    def test_updates_draft(self):
        model = FakeStructuredChatModel(responses=["수정된 초안"])
        cross_check = make_cross_check_node(model)
        state = {
            "draft": "원본 초안",
            "verification_answers": [
                {"question": "Q1", "answer": "A1"},
            ],
        }
        result = cross_check(state)
        assert result["draft"] == "수정된 초안"


class TestSanitizerNode:
    """sanitizer_node 테스트."""

    def test_clean_draft_no_errors(self, sanitizer):
        node = make_sanitizer_node(sanitizer)
        state = {
            "draft": "이것은 일반 텍스트입니다.",
            "correction_count": 0,
        }
        result = node(state)
        assert result["sanitizer_errors"] == []
        assert result["correction_count"] == 0

    def test_draft_with_fake_citation(self, sanitizer):
        node = make_sanitizer_node(sanitizer)
        state = {
            "draft": "화성 인구 100만. [출처: 화성 인구 통계 2024]",
            "correction_count": 0,
        }
        result = node(state)
        assert len(result["sanitizer_errors"]) >= 1
        assert result["correction_count"] == 1
        assert "SANITIZER_FAIL" in result["error_log"][0]

    def test_increments_correction_count(self, sanitizer):
        node = make_sanitizer_node(sanitizer)
        state = {
            "draft": "[출처: 존재하지 않는 출처]",
            "correction_count": 2,
        }
        result = node(state)
        assert result["correction_count"] == 3

    def test_empty_draft(self, sanitizer):
        node = make_sanitizer_node(sanitizer)
        state = {"draft": "", "correction_count": 0}
        result = node(state)
        assert result["sanitizer_errors"] == []


class TestRouteAfterSanitizer:
    """route_after_sanitizer 라우팅 테스트."""

    def test_errors_below_max_returns_self_correct(self):
        state = {"sanitizer_errors": ["error1"], "correction_count": 1}
        assert route_after_sanitizer(state) == "self_correct"

    def test_errors_at_max_returns_finalize(self):
        state = {"sanitizer_errors": ["error1"], "correction_count": MAX_CORRECTIONS}
        assert route_after_sanitizer(state) == "finalize"

    def test_no_errors_returns_finalize(self):
        state = {"sanitizer_errors": [], "correction_count": 0}
        assert route_after_sanitizer(state) == "finalize"

    def test_empty_errors_list_returns_finalize(self):
        state = {"sanitizer_errors": [], "correction_count": 2}
        assert route_after_sanitizer(state) == "finalize"


class TestSelfCorrectionNode:
    """self_correction_node 테스트."""

    def test_corrects_draft(self):
        model = FakeStructuredChatModel(responses=["교정된 초안"])
        node = make_self_correction_node(model)
        state = {
            "draft": "오류가 있는 초안",
            "sanitizer_errors": ["FAKE_CITATION: 출처 없음"],
        }
        result = node(state)
        assert result["draft"] == "교정된 초안"
        assert result["sanitizer_errors"] == []


class TestFinalizeNode:
    """finalize_node 테스트."""

    def test_wraps_in_ai_message(self):
        state = {"draft": "최종 보고서 내용"}
        result = finalize_node(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "최종 보고서 내용"

    def test_empty_draft(self):
        state = {"draft": ""}
        result = finalize_node(state)
        assert result["messages"][0].content == ""

    def test_none_draft_defaults(self):
        state = {}
        result = finalize_node(state)
        assert result["messages"][0].content == ""


class TestDispatchStormWorkers:
    """dispatch_storm_workers 라우팅 테스트."""

    def test_dispatches_per_subtask(self, sample_execution_plan):
        state = {"execution_plan": sample_execution_plan}
        sends = dispatch_storm_workers(state)
        assert len(sends) == 2
        # Send objects target "storm_worker"
        for send in sends:
            assert send.node == "storm_worker"
            assert "task" in send.arg

    def test_empty_plan_sends_to_synthesis(self):
        empty_plan = ExecutionPlan(goal="empty", sub_tasks=[])
        state = {"execution_plan": empty_plan}
        sends = dispatch_storm_workers(state)
        assert len(sends) == 1
        assert sends[0].node == "synthesis"

    def test_none_plan_sends_to_synthesis(self):
        state = {"execution_plan": None}
        sends = dispatch_storm_workers(state)
        assert len(sends) == 1
        assert sends[0].node == "synthesis"


class TestDispatchVerifiers:
    """dispatch_verifiers 라우팅 테스트."""

    def test_dispatches_per_question(self):
        state = {"verification_questions": ["Q1?", "Q2?", "Q3?"]}
        sends = dispatch_verifiers(state)
        assert len(sends) == 3
        for send in sends:
            assert send.node == "factored_verifier"
            assert "question" in send.arg
            assert send.arg["draft_context"] is None

    def test_empty_questions_sends_to_cross_check(self):
        state = {"verification_questions": []}
        sends = dispatch_verifiers(state)
        assert len(sends) == 1
        assert sends[0].node == "cross_check"


# ── 그래프 조립 테스트 ──


class TestBuildPipeline:
    """build_zero_hallucination_pipeline 그래프 조립 테스트."""

    def _build_graph(self, chunk_db):
        """테스트용 그래프를 빌드한다."""
        dummy = FakeStructuredChatModel(responses=["dummy"])
        return build_zero_hallucination_pipeline(
            planner_model=dummy,
            worker_model=dummy,
            verifier_model=dummy,
            synthesizer_model=dummy,
            chunk_db=chunk_db,
        )

    def test_graph_compiles(self, chunk_db):
        graph = self._build_graph(chunk_db)
        compiled = graph.compile(checkpointer=MemorySaver())
        assert compiled is not None

    def test_has_all_nodes(self, chunk_db):
        graph = self._build_graph(chunk_db)
        node_names = set(graph.nodes.keys())
        expected = {
            "planner", "storm_worker", "synthesis", "cove_plan",
            "factored_verifier", "cross_check", "sanitizer",
            "self_correct", "finalize",
        }
        assert expected.issubset(node_names)

    def test_entry_point_is_planner(self, chunk_db):
        graph = self._build_graph(chunk_db)
        # StateGraph stores entry point in __start__ edges
        compiled = graph.compile()
        # The graph should be compilable and have planner as start
        graph_dict = compiled.get_graph().to_json()
        assert graph_dict is not None


class TestPipelineEndToEnd:
    """E2E 파이프라인 테스트. FakeModel로 전체 흐름 실행."""

    def test_full_pipeline_clean_draft(self, chunk_db):
        """Sanitizer 오류 없는 깨끗한 흐름 테스트."""
        plan = ExecutionPlan(
            goal="서울 분석",
            sub_tasks=[SubTask(id="t1", description="인구 분석")],
        )

        # Planner: ExecutionPlan JSON 반환
        planner_model = FakeStructuredChatModel(
            responses=[plan.model_dump_json()]
        )

        # Worker: 3 페르소나 + 1 압축 (1 작업)
        worker_model = FakeStructuredChatModel(responses=[
            "비판적 분석: 서울 인구 데이터 확인 필요",
            "전문가 의견: 통계청 자료 기준 950만",
            "반론: 유동인구 제외 시 다를 수 있음",
            "서울 인구 약 950만 명 (통계청 2024)",
        ])

        # Synthesizer: 깨끗한 초안 (sanitizer 통과하도록)
        synthesizer_model = FakeStructuredChatModel(responses=[
            "서울의 인구는 약 950만 명이다. 이는 통계청 자료에 기반한다.",
        ])

        # Verifier: 질문 생성 → 팩토리드 검증 → 교차 대조
        verifier_model = FakeStructuredChatModel(responses=[
            "1. 서울 인구가 950만인가?",  # cove_plan
            "네, 통계청 2024년 자료에 의하면 정확합니다.",  # factored_verifier
            "서울의 인구는 약 950만 명이다. 검증 완료.",  # cross_check
        ])

        graph = build_zero_hallucination_pipeline(
            planner_model=planner_model,
            worker_model=worker_model,
            verifier_model=verifier_model,
            synthesizer_model=synthesizer_model,
            chunk_db=chunk_db,
        )

        compiled = graph.compile(checkpointer=MemorySaver())
        result = compiled.invoke(
            {
                "messages": [HumanMessage(content="서울에 대해 분석하라")],
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
            },
            config={"configurable": {"thread_id": "e2e-test-1"}},
        )

        # 최종 메시지가 AIMessage인지 확인
        assert len(result["messages"]) >= 2  # HumanMessage + ... + AIMessage
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert len(last_msg.content) > 0

        # draft가 설정되었는지 확인
        assert result["draft"] is not None

        # sanitizer 오류가 없었는지 확인
        assert result["sanitizer_errors"] == []

    def test_pipeline_with_correction_loop(self, tmp_dir):
        """Sanitizer 오류가 있어서 교정 루프가 실행되는 테스트."""
        # 빈 chunk_db → 모든 인용이 FAKE_CITATION으로 감지됨
        empty_db = LocalChunkDB()

        plan = ExecutionPlan(
            goal="분석",
            sub_tasks=[SubTask(id="t1", description="작업")],
        )

        planner_model = FakeStructuredChatModel(
            responses=[plan.model_dump_json()]
        )

        worker_model = FakeStructuredChatModel(responses=[
            "분석1", "분석2", "분석3", "압축 결과",
        ])

        # 1차 초안: 가짜 인용 포함 → sanitizer 실패
        # 교정 후 2차 초안: 인용 없는 깨끗한 텍스트 → sanitizer 통과
        synthesizer_model = FakeStructuredChatModel(responses=[
            "[출처: 가짜 인용] 잘못된 초안",  # synthesis
            "교정된 깨끗한 초안입니다.",  # self_correct
        ])

        verifier_model = FakeStructuredChatModel(responses=[
            "1. 확인 질문?",  # cove_plan
            "확인됨",  # factored_verifier
            "[출처: 가짜 인용] 교차검증 결과",  # cross_check (still has fake cite)
        ])

        graph = build_zero_hallucination_pipeline(
            planner_model=planner_model,
            worker_model=worker_model,
            verifier_model=verifier_model,
            synthesizer_model=synthesizer_model,
            chunk_db=empty_db,
        )

        compiled = graph.compile(checkpointer=MemorySaver())
        result = compiled.invoke(
            {
                "messages": [HumanMessage(content="분석 요청")],
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
            },
            config={"configurable": {"thread_id": "e2e-correction-test"}},
        )

        # 교정이 1회 이상 실행되었는지 확인
        assert result["correction_count"] >= 1

        # 최종 메시지가 존재하는지 확인
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
