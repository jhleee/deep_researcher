"""연구 시나리오 기반 통합 테스트.

다양한 연구 주제, 참고자료, 엣지 케이스를 통해
에이전트 하네스 파이프라인의 전체 동작을 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from architectures.zero_hallucination import (
    MAX_CORRECTIONS,
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
from harness.guardrails import InputGuardrail, OutputGuardrail
from harness.models import ExecutionPlan, FakeStructuredChatModel, SubTask
from harness.sanitizer import DeterministicSanitizer, LocalChunkDB


# ── 데이터 로딩 ──

DATASETS_DIR = Path(__file__).parent / "datasets"


def _load_topics() -> dict:
    return json.loads((DATASETS_DIR / "research_topics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def topics_data() -> dict:
    return _load_topics()


def _make_chunk_db(topic: dict, tmp_path: Path) -> LocalChunkDB:
    """연구 주제 데이터로 LocalChunkDB를 생성한다."""
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
    """ZeroHallucinationState 초기 상태를 생성한다."""
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


# ── 주제별 Sanitizer 검증 테스트 ──


class TestSanitizerWithResearchData:
    """다양한 연구 주제의 참고자료를 이용한 Sanitizer 검증."""

    @pytest.fixture(params=["climate_economy", "ai_ethics", "semiconductor",
                            "urban_transport", "biotech"])
    def topic_with_db(self, request, tmp_path, topics_data):
        topic = next(t for t in topics_data["topics"] if t["id"] == request.param)
        db = _make_chunk_db(topic, tmp_path)
        return topic, DeterministicSanitizer(chunk_db=db)

    def test_clean_draft_passes(self, topic_with_db):
        """정확한 인용을 포함한 초안은 오류 없이 통과한다."""
        topic, sanitizer = topic_with_db
        errors = sanitizer.validate(topic["drafts"]["clean"])
        assert errors == [], f"[{topic['id']}] 깨끗한 초안에 오류 감지: {errors}"

    def test_fake_citation_detected(self, topic_with_db):
        """존재하지 않는 인용은 FAKE_CITATION으로 감지된다."""
        topic, sanitizer = topic_with_db
        # fuzzy_match 초안은 의도적으로 통과하도록 설계됨 → 제외
        skip_keys = {"clean", "fuzzy_match"}
        error_drafts = {k: v for k, v in topic["drafts"].items() if k not in skip_keys}
        if not error_drafts:
            pytest.skip(f"[{topic['id']}] 오류 초안 없음")
        for draft_name, draft_text in error_drafts.items():
            errors = sanitizer.validate(draft_text)
            assert len(errors) > 0, (
                f"[{topic['id']}/{draft_name}] 오류가 감지되지 않음"
            )

    def test_valid_url_passes(self, topic_with_db):
        """등록된 URL은 오류로 감지되지 않는다."""
        topic, sanitizer = topic_with_db
        valid_url = topic["urls"][0]
        draft = f"참고: {valid_url}"
        errors = [e for e in sanitizer.validate(draft) if e.error_type == "URL_INVALID"]
        assert len(errors) == 0

    def test_invalid_url_detected(self, topic_with_db):
        """등록되지 않은 URL은 URL_INVALID로 감지된다."""
        topic, sanitizer = topic_with_db
        draft = "참고: https://totally-fake-url.invalid/page"
        errors = [e for e in sanitizer.validate(draft) if e.error_type == "URL_INVALID"]
        assert len(errors) == 1

    def test_valid_number_passes(self, topic_with_db):
        """등록된 수치는 오류로 감지되지 않는다."""
        topic, sanitizer = topic_with_db
        # NUMBER_PATTERN은 N% 또는 $N 형태만 매칭
        first_num = topic["numbers"][0]
        draft = f"비율은 약 {first_num}%이다."
        errors = [e for e in sanitizer.validate(draft) if e.error_type == "NUMBER_UNVERIFIED"]
        assert len(errors) == 0

    def test_unverified_number_detected(self, topic_with_db):
        """등록되지 않은 수치는 NUMBER_UNVERIFIED로 감지된다."""
        topic, sanitizer = topic_with_db
        draft = "성장률은 약 99.99%에 달한다."
        errors = [e for e in sanitizer.validate(draft) if e.error_type == "NUMBER_UNVERIFIED"]
        assert len(errors) == 1


# ── 주제별 Planner 노드 테스트 ──


class TestPlannerWithTopics:
    """다양한 연구 질문에 대한 Planner 노드 동작 검증."""

    @pytest.mark.parametrize("topic_id", [
        "climate_economy", "ai_ethics", "semiconductor",
        "urban_transport", "biotech",
    ])
    def test_plan_generates_subtasks(self, topic_id, topics_data):
        """각 연구 주제에 대해 적절한 수의 하위 작업을 생성한다."""
        topic = next(t for t in topics_data["topics"] if t["id"] == topic_id)
        subtask_count = topic["expected_subtasks_min"]

        # 주제에 맞는 실행 계획 생성
        subtasks = [
            SubTask(id=f"t{i+1}", description=f"하위작업 {i+1}")
            for i in range(subtask_count)
        ]
        plan = ExecutionPlan(goal=topic["query"], sub_tasks=subtasks)
        model = FakeStructuredChatModel(responses=[plan.model_dump_json()])
        planner = make_planner_node(model)

        result = planner({"messages": [HumanMessage(content=topic["query"])]})
        assert isinstance(result["execution_plan"], ExecutionPlan)
        assert len(result["execution_plan"].sub_tasks) >= subtask_count

    def test_complex_plan_with_dependencies(self, topics_data):
        """의존성 있는 작업 구조를 올바르게 계획한다."""
        scenario = topics_data["multi_task_scenarios"][2]  # dependent_tasks
        subtasks = [
            SubTask(
                id=st["id"],
                description=st["description"],
                dependencies=st["dependencies"],
            )
            for st in scenario["subtasks"]
        ]
        plan = ExecutionPlan(goal="복합 분석", sub_tasks=subtasks)
        model = FakeStructuredChatModel(responses=[plan.model_dump_json()])
        planner = make_planner_node(model)

        result = planner({"messages": [HumanMessage(content="복합 분석 요청")]})
        tasks = result["execution_plan"].sub_tasks
        assert tasks[1].dependencies == ["t1"]
        assert tasks[2].dependencies == ["t2"]


# ── STORM 워커 다중 주제 테스트 ──


class TestStormWorkerMultiTopic:
    """다양한 주제에 대한 STORM 워커 동작 검증."""

    @pytest.mark.parametrize("task_desc,task_id", [
        ("기후변화의 경제적 영향 분석", "climate_t1"),
        ("AI 규제 현황 비교", "ai_t1"),
        ("반도체 공급망 분석", "semi_t1"),
        ("서울 교통 정체 원인 파악", "transport_t1"),
        ("mRNA 백신 기술 동향", "bio_t1"),
    ])
    def test_worker_processes_task(self, task_desc, task_id):
        """각 주제의 작업을 3 페르소나로 탐색하고 압축한다."""
        model = FakeStructuredChatModel(responses=[
            f"[비판적 검토자] {task_desc}: 데이터 신뢰성 검토 결과",
            f"[도메인 전문가] {task_desc}: 전문 분석 의견",
            f"[반론 제기자] {task_desc}: 대안적 시각",
            f"핵심 사실: {task_desc} 압축 결과 200토큰 이내",
        ])
        worker = make_storm_worker(model)
        result = worker({"task": SubTask(id=task_id, description=task_desc)})

        assert len(result["worker_results"]) == 1
        assert result["worker_results"][0]["task_id"] == task_id
        assert "압축" in result["worker_results"][0]["data"]

    def test_multiple_workers_aggregate(self):
        """5개 작업의 워커 결과가 정확히 집계된다."""
        all_results = []
        for i in range(5):
            model = FakeStructuredChatModel(responses=[
                f"비판{i}", f"전문{i}", f"반론{i}", f"압축결과_{i}",
            ])
            worker = make_storm_worker(model)
            result = worker({
                "task": SubTask(id=f"t{i}", description=f"작업_{i}")
            })
            all_results.extend(result["worker_results"])

        assert len(all_results) == 5
        task_ids = {r["task_id"] for r in all_results}
        assert task_ids == {f"t{i}" for i in range(5)}


# ── Send API 디스패치 테스트 ──


class TestDispatchScenarios:
    """다양한 작업 구조에 대한 Send API 디스패치 검증."""

    def test_single_task_dispatch(self, topics_data):
        scenario = topics_data["multi_task_scenarios"][0]
        plan = ExecutionPlan(
            goal="단일 작업",
            sub_tasks=[SubTask(id="t1", description="작업 1")],
        )
        sends = dispatch_storm_workers({"execution_plan": plan})
        assert len(sends) == scenario["expected_worker_dispatches"]
        assert sends[0].node == "storm_worker"

    def test_five_task_parallel_dispatch(self, topics_data):
        scenario = topics_data["multi_task_scenarios"][1]
        plan = ExecutionPlan(
            goal="5개 작업",
            sub_tasks=[SubTask(id=f"t{i}", description=f"작업 {i}")
                       for i in range(scenario["subtask_count"])],
        )
        sends = dispatch_storm_workers({"execution_plan": plan})
        assert len(sends) == scenario["expected_worker_dispatches"]
        assert all(s.node == "storm_worker" for s in sends)

    def test_many_verification_questions_dispatch(self):
        """많은 검증 질문이 병렬 디스패치된다."""
        questions = [f"사실 {i}이 정확한가?" for i in range(10)]
        sends = dispatch_verifiers({"verification_questions": questions})
        assert len(sends) == 10
        assert all(s.node == "factored_verifier" for s in sends)
        assert all(s.arg["draft_context"] is None for s in sends)


# ── CoVe 검증 다중 질문 테스트 ──


class TestCoveVerificationMultiQuestion:
    """다양한 검증 질문 패턴에 대한 CoVe 파이프라인 동작."""

    def test_numeric_claim_verification(self):
        """수치적 주장에 대한 검증 질문 생성."""
        model = FakeStructuredChatModel(responses=[
            "1. 탄소 배출량이 정확히 6.1억 톤인가?\n"
            "2. 경제 손실 7조 원의 산출 근거는?\n"
            "3. 40% 감축 목표의 기준 연도는?"
        ])
        cove_plan = make_cove_plan_node(model)
        result = cove_plan({"draft": "기후변화 분석 초안"})
        assert len(result["verification_questions"]) == 3

    def test_source_attribution_verification(self):
        """출처 귀속에 대한 검증 질문 생성."""
        model = FakeStructuredChatModel(responses=[
            "1. 통계청 2024 자료에 해당 수치가 실제로 존재하는가?\n"
            "2. 한국은행 발표 GDP 수치와 일치하는가?"
        ])
        cove_plan = make_cove_plan_node(model)
        result = cove_plan({"draft": "출처 기반 초안"})
        assert len(result["verification_questions"]) == 2

    def test_factored_verification_isolation(self):
        """각 검증이 초안 컨텍스트 없이 독립적으로 수행된다."""
        answers = [
            "통계청 2024 자료에 따르면 서울 인구는 949만이다.",
            "한국은행 2024년 GDP 전망은 1.7조 달러이다.",
            "농촌진흥청 보고서에 12% 감소가 기록되어 있다.",
        ]
        model = FakeStructuredChatModel(responses=answers)
        verifier = make_factored_verifier(model)

        results = []
        for i, q in enumerate([
            "서울 인구가 950만인가?",
            "GDP가 1.7조 달러인가?",
            "농업 생산성 감소가 12%인가?",
        ]):
            r = verifier({"question": q, "draft_context": None})
            results.append(r["verification_answers"][0])

        assert len(results) == 3
        assert all("answer" in r for r in results)
        assert all("question" in r for r in results)

    def test_cross_check_with_conflicting_answers(self):
        """검증 답변이 초안과 충돌할 때 초안을 수정한다."""
        model = FakeStructuredChatModel(responses=[
            "수정됨: 탄소 배출량은 6.1억 톤(검증 확인)이며 "
            "경제 손실은 약 7조 원(검증 확인)이다."
        ])
        cross_check = make_cross_check_node(model)
        result = cross_check({
            "draft": "탄소 배출량 10억 톤, 경제 손실 50조 원",
            "verification_answers": [
                {"question": "배출량?", "answer": "6.1억 톤"},
                {"question": "손실?", "answer": "7조 원"},
            ],
        })
        assert "수정" in result["draft"]


# ── Sanitizer 교정 루프 시나리오 테스트 ──


class TestCorrectionLoopScenarios:
    """교정 루프의 다양한 시나리오를 검증한다."""

    def test_route_to_self_correct_on_errors(self):
        state = {"sanitizer_errors": ["FAKE_CITATION"], "correction_count": 0}
        assert route_after_sanitizer(state) == "self_correct"

    def test_route_to_finalize_on_clean(self):
        state = {"sanitizer_errors": [], "correction_count": 0}
        assert route_after_sanitizer(state) == "finalize"

    def test_route_to_finalize_at_max(self):
        state = {"sanitizer_errors": ["error"], "correction_count": MAX_CORRECTIONS}
        assert route_after_sanitizer(state) == "finalize"

    def test_self_correction_clears_errors(self):
        model = FakeStructuredChatModel(responses=["교정 완료: 검증된 사실만 포함"])
        node = make_self_correction_node(model)
        result = node({
            "draft": "[출처: 가짜] 잘못된 내용",
            "sanitizer_errors": ["FAKE_CITATION: 출처 없음"],
        })
        assert result["sanitizer_errors"] == []
        assert "교정" in result["draft"]

    @pytest.mark.parametrize("correction_count", [0, 1, 2])
    def test_incremental_correction_count(self, correction_count, tmp_path):
        """correction_count가 매 sanitizer 오류마다 증가한다."""
        db = LocalChunkDB()
        sanitizer = DeterministicSanitizer(chunk_db=db)
        node = make_sanitizer_node(sanitizer)
        result = node({
            "draft": "[출처: 존재하지 않는 출처] 내용",
            "correction_count": correction_count,
        })
        assert result["correction_count"] == correction_count + 1


# ── Guardrail 시나리오 테스트 ──


class TestGuardrailScenarios:
    """가드레일의 다양한 입력 시나리오를 검증한다."""

    @pytest.fixture
    def input_guard(self) -> InputGuardrail:
        return InputGuardrail(strict=True)

    @pytest.fixture
    def output_guard(self) -> OutputGuardrail:
        return OutputGuardrail(strict=True)

    def test_pii_phone_masked(self, input_guard, topics_data):
        scenario = topics_data["guardrail_scenarios"][0]
        result = input_guard.check(scenario["query"])
        assert result.passed is True
        assert "010-1234-5678" not in result.sanitized_content
        assert "[REDACTED]" in result.sanitized_content

    def test_injection_blocked(self, input_guard, topics_data):
        scenario = topics_data["guardrail_scenarios"][1]
        result = input_guard.check(scenario["query"])
        assert result.passed is False
        assert result.reason == "INJECTION_DETECTED"

    def test_long_input_rejected(self, input_guard, topics_data):
        scenario = topics_data["guardrail_scenarios"][2]
        long_query = scenario["query_pattern"] * scenario["repeat_count"]
        result = input_guard.check(long_query)
        assert result.passed is False
        assert result.reason == "INPUT_TOO_LONG"

    def test_email_masked(self, input_guard, topics_data):
        scenario = topics_data["guardrail_scenarios"][3]
        result = input_guard.check(scenario["query"])
        assert result.passed is True
        assert "user@example.com" not in result.sanitized_content
        assert "[REDACTED]" in result.sanitized_content

    def test_clean_research_query_unchanged(self, input_guard):
        """정상 연구 질문은 변경 없이 통과한다."""
        query = "한국 반도체 산업의 글로벌 경쟁력을 분석하라"
        result = input_guard.check(query)
        assert result.passed is True
        assert result.sanitized_content == query

    def test_output_guard_clean_passes(self, output_guard):
        """정상 출력은 변경 없이 통과한다."""
        output = "한국의 탄소 배출량은 약 6.1억 톤이다."
        result = output_guard.check(output)
        assert result.passed is True
        assert result.sanitized_content == output

    def test_output_guard_harmful_blocked(self, output_guard):
        """유해 콘텐츠가 포함된 출력은 필터링된다."""
        output = "폭탄 만들기 방법은 다음과 같다."
        result = output_guard.check(output)
        assert result.passed is False
        assert result.reason == "HARMFUL_CONTENT"

    def test_output_guard_long_truncated(self):
        """매우 긴 출력은 잘라낸다."""
        guard = OutputGuardrail(strict=True, max_output_length=100)
        output = "가" * 200
        result = guard.check(output)
        assert result.passed is True
        assert result.reason == "OUTPUT_TRUNCATED"
        assert len(result.sanitized_content) < 200


# ── 주제별 E2E 파이프라인 테스트 ──


class TestEndToEndPerTopic:
    """각 연구 주제에 대한 전체 파이프라인 E2E 실행."""

    def _run_e2e(
        self,
        topic: dict,
        tmp_path: Path,
        *,
        draft_key: str = "clean",
        thread_id: str = "e2e",
    ) -> dict:
        """주제 데이터로 E2E 파이프라인을 실행한다."""
        chunk_db = _make_chunk_db(topic, tmp_path)

        subtasks = [
            SubTask(id=f"t{i+1}", description=f"하위작업_{i+1}")
            for i in range(topic["expected_subtasks_min"])
        ]
        plan = ExecutionPlan(goal=topic["query"], sub_tasks=subtasks)

        planner_model = FakeStructuredChatModel(responses=[plan.model_dump_json()])

        # 워커: 작업당 4 응답 (3 페르소나 + 1 압축)
        worker_responses = []
        for i in range(len(subtasks)):
            worker_responses.extend([
                f"[비판적 검토자] 작업{i} 비판 분석",
                f"[도메인 전문가] 작업{i} 전문 분석",
                f"[반론 제기자] 작업{i} 반론",
                f"작업{i} 핵심 사실 압축",
            ])
        worker_model = FakeStructuredChatModel(responses=worker_responses)

        # 합성: 깨끗한 초안 반환
        synthesizer_model = FakeStructuredChatModel(
            responses=[topic["drafts"][draft_key]]
        )

        # 검증: 간단한 질문 + 응답 + 교차검증
        verifier_model = FakeStructuredChatModel(responses=[
            "1. 주요 수치가 정확한가?",
            "네, 검증되었습니다.",
            topic["drafts"][draft_key],  # cross_check이 동일 초안 반환
        ])

        graph = build_zero_hallucination_pipeline(
            planner_model=planner_model,
            worker_model=worker_model,
            verifier_model=verifier_model,
            synthesizer_model=synthesizer_model,
            chunk_db=chunk_db,
        )

        compiled = graph.compile(checkpointer=MemorySaver())
        return compiled.invoke(
            _make_zh_initial_state(topic["query"]),
            config={"configurable": {"thread_id": thread_id}},
        )

    @pytest.mark.parametrize("topic_id", [
        "climate_economy", "ai_ethics", "semiconductor",
        "urban_transport", "biotech",
    ])
    def test_clean_pipeline_per_topic(self, topic_id, tmp_path, topics_data):
        """각 주제의 깨끗한 초안이 오류 없이 파이프라인을 통과한다."""
        topic = next(t for t in topics_data["topics"] if t["id"] == topic_id)
        result = self._run_e2e(topic, tmp_path, thread_id=f"clean-{topic_id}")

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert len(last_msg.content) > 0
        assert result["sanitizer_errors"] == []
        assert result["draft"] is not None

    def test_climate_correction_loop(self, tmp_path, topics_data):
        """기후변화 주제: 가짜 인용 → 교정 → 통과 흐름."""
        topic = next(t for t in topics_data["topics"] if t["id"] == "climate_economy")
        chunk_db = _make_chunk_db(topic, tmp_path)

        plan = ExecutionPlan(
            goal=topic["query"],
            sub_tasks=[SubTask(id="t1", description="기후 영향 분석")],
        )
        planner_model = FakeStructuredChatModel(responses=[plan.model_dump_json()])
        worker_model = FakeStructuredChatModel(
            responses=["비판", "전문", "반론", "압축"]
        )

        # 1차: 가짜 인용 포함 초안 → sanitizer 실패
        # 교정 후 2차: 깨끗한 초안
        synthesizer_model = FakeStructuredChatModel(responses=[
            topic["drafts"]["fake_citation"],   # synthesis
            topic["drafts"]["clean"],            # self_correct
        ])
        verifier_model = FakeStructuredChatModel(responses=[
            "1. 배출량 수치 확인?",
            "확인됨",
            topic["drafts"]["fake_citation"],    # cross_check
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
            _make_zh_initial_state(topic["query"]),
            config={"configurable": {"thread_id": "climate-correction"}},
        )

        assert result["correction_count"] >= 1
        assert isinstance(result["messages"][-1], AIMessage)

    def test_semiconductor_many_subtasks(self, tmp_path, topics_data):
        """반도체 주제: 3+ 하위작업으로 병렬 워커 스폰."""
        topic = next(t for t in topics_data["topics"] if t["id"] == "semiconductor")
        chunk_db = _make_chunk_db(topic, tmp_path)

        subtasks = [
            SubTask(id="t1", description="삼성전자 실적 분석"),
            SubTask(id="t2", description="SK하이닉스 HBM 경쟁력"),
            SubTask(id="t3", description="글로벌 파운드리 경쟁"),
        ]
        plan = ExecutionPlan(goal=topic["query"], sub_tasks=subtasks)

        planner_model = FakeStructuredChatModel(responses=[plan.model_dump_json()])
        worker_model = FakeStructuredChatModel(responses=[
            "비판1", "전문1", "반론1", "삼성 100조 매출",
            "비판2", "전문2", "반론2", "HBM3E 점유율 53%",
            "비판3", "전문3", "반론3", "TSMC 3nm 수율 80%",
        ])
        synthesizer_model = FakeStructuredChatModel(
            responses=[topic["drafts"]["clean"]]
        )
        verifier_model = FakeStructuredChatModel(responses=[
            "1. 삼성 매출 확인?\n2. HBM 점유율 확인?\n3. 수출액 확인?",
            "100조 원 확인", "53% 확인", "1300억 달러 확인",
            topic["drafts"]["clean"],
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
            _make_zh_initial_state(topic["query"]),
            config={"configurable": {"thread_id": "semi-many-tasks"}},
        )

        assert len(result["worker_results"]) == 3
        assert isinstance(result["messages"][-1], AIMessage)

    def test_max_corrections_forces_finalize(self, tmp_path, topics_data):
        """MAX_CORRECTIONS 도달 시 오류가 있어도 finalize로 진행."""
        topic = next(t for t in topics_data["topics"] if t["id"] == "urban_transport")
        # 빈 DB → 모든 인용이 실패
        empty_db = LocalChunkDB()

        plan = ExecutionPlan(
            goal="교통 분석",
            sub_tasks=[SubTask(id="t1", description="분석")],
        )
        planner_model = FakeStructuredChatModel(responses=[plan.model_dump_json()])
        worker_model = FakeStructuredChatModel(
            responses=["비판", "전문", "반론", "압축"]
        )

        # 매번 가짜 인용 포함 초안을 반환 → 교정해도 다시 실패
        bad_draft = "[출처: 완전히 가짜] 내용"
        synthesizer_model = FakeStructuredChatModel(responses=[
            bad_draft,  # synthesis
            bad_draft,  # self_correct 1
            bad_draft,  # self_correct 2
            bad_draft,  # self_correct 3 (도달 안 할 수 있음)
        ])
        verifier_model = FakeStructuredChatModel(responses=[
            "1. 확인?", "확인", bad_draft,
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
            _make_zh_initial_state("교통 분석 요청"),
            config={"configurable": {"thread_id": "max-corrections"}},
        )

        # MAX_CORRECTIONS에 도달했어야 함
        assert result["correction_count"] >= MAX_CORRECTIONS
        assert isinstance(result["messages"][-1], AIMessage)


# ── Sanitizer 엣지 케이스 심층 테스트 ──


class TestSanitizerEdgeCases:
    """Sanitizer의 엣지 케이스를 상세히 검증한다."""

    @pytest.fixture
    def rich_db(self, tmp_path) -> LocalChunkDB:
        """다양한 데이터가 포함된 풍부한 청크 DB."""
        db = LocalChunkDB(db_path=tmp_path / "rich.json")
        for topic in _load_topics()["topics"]:
            for chunk in topic["chunks"]:
                db.add_chunk(chunk["text"], source=chunk["source"])
            for url in topic["urls"]:
                db.add_url(url)
            for num in topic["numbers"]:
                db.add_number(str(num))
        db.save()
        return db

    @pytest.fixture
    def rich_sanitizer(self, rich_db) -> DeterministicSanitizer:
        return DeterministicSanitizer(chunk_db=rich_db)

    def test_multiple_valid_citations(self, rich_sanitizer):
        """여러 유효한 인용이 모두 통과한다."""
        draft = (
            "[출처: 한국의 탄소 배출량은 2023년 기준 약 6.1억 톤이다.] "
            "[출처: EU의 AI Act는 2024년 8월 발효되었으며 고위험 AI 시스템에 대한 규제를 포함한다.] "
            "[출처: 삼성전자의 2024년 반도체 매출은 약 100조 원으로 세계 1위이다.]"
        )
        errors = rich_sanitizer.validate(draft)
        assert errors == []

    def test_mixed_valid_and_invalid_citations(self, rich_sanitizer):
        """유효/무효 인용이 섞인 경우 무효한 것만 감지한다."""
        draft = (
            "[출처: 한국의 탄소 배출량은 2023년 기준 약 6.1억 톤이다.] "
            "[출처: 이것은 완전히 지어낸 출처입니다.] "
            "[출처: 삼성전자의 2024년 반도체 매출은 약 100조 원으로 세계 1위이다.]"
        )
        errors = rich_sanitizer.validate(draft)
        assert len(errors) == 1
        assert errors[0].error_type == "FAKE_CITATION"

    def test_ref_tag_format(self, rich_sanitizer):
        """[ref: ...] 형식도 동일하게 검증된다."""
        draft = (
            "[ref: 한국의 탄소 배출량은 2023년 기준 약 6.1억 톤이다.] "
            "[ref: 완전 가짜 출처]"
        )
        errors = rich_sanitizer.validate(draft)
        fake_errors = [e for e in errors if e.error_type == "FAKE_CITATION"]
        assert len(fake_errors) == 1

    def test_multiple_urls_mixed(self, rich_sanitizer):
        """등록/미등록 URL이 섞인 경우."""
        draft = (
            "참고: https://me.go.kr/climate-report-2023 "
            "추가: https://unknown-site.com/fake "
            "또한: https://oecd.ai/policy"
        )
        errors = rich_sanitizer.validate(draft)
        url_errors = [e for e in errors if e.error_type == "URL_INVALID"]
        assert len(url_errors) == 1

    def test_no_citations_no_errors(self, rich_sanitizer):
        """인용/URL/수치가 없는 순수 텍스트."""
        draft = "이것은 사실적 주장이 없는 일반 텍스트이다."
        errors = rich_sanitizer.validate(draft)
        assert errors == []

    def test_empty_draft(self, rich_sanitizer):
        """빈 초안은 오류 없이 통과한다."""
        assert rich_sanitizer.validate("") == []

    def test_percentage_and_dollar_numbers(self, rich_sanitizer):
        """% 와 $ 접두사 수치 모두 검증된다."""
        draft = "성장률은 18%이고, 투자액은 $170이다."
        errors = rich_sanitizer.validate(draft)
        # 18과 170은 모두 등록된 수치
        num_errors = [e for e in errors if e.error_type == "NUMBER_UNVERIFIED"]
        assert len(num_errors) == 0

    def test_unregistered_percentage(self, rich_sanitizer):
        """등록되지 않은 퍼센트 수치 감지."""
        draft = "점유율은 약 99.99%에 달한다."
        errors = rich_sanitizer.validate(draft)
        num_errors = [e for e in errors if e.error_type == "NUMBER_UNVERIFIED"]
        assert len(num_errors) == 1

    def test_fuzzy_match_slight_variation(self, rich_sanitizer):
        """원본과 약간 다른 인용이 fuzzy match로 통과한다."""
        # 원본: "한국의 탄소 배출량은 2023년 기준 약 6.1억 톤이다."
        draft = "[출처: 한국의 탄소 배출량은 2023년 기준 약 6.1억톤이다]"
        errors = rich_sanitizer.validate(draft)
        fake_errors = [e for e in errors if e.error_type == "FAKE_CITATION"]
        assert len(fake_errors) == 0  # fuzzy match로 통과해야 함

    def test_completely_different_citation_fails(self, rich_sanitizer):
        """완전히 다른 인용은 fuzzy match도 실패한다."""
        draft = "[출처: 화성에 인간이 정착한 것은 2025년이다.]"
        errors = rich_sanitizer.validate(draft)
        fake_errors = [e for e in errors if e.error_type == "FAKE_CITATION"]
        assert len(fake_errors) == 1


# ── ChunkDB 교차 주제 테스트 ──


class TestChunkDBCrossTopic:
    """여러 주제를 합친 ChunkDB에서의 검색 동작."""

    @pytest.fixture
    def merged_db(self, tmp_path) -> LocalChunkDB:
        db = LocalChunkDB(db_path=tmp_path / "merged.json")
        for topic in _load_topics()["topics"]:
            for chunk in topic["chunks"]:
                db.add_chunk(chunk["text"], source=chunk["source"])
            for url in topic["urls"]:
                db.add_url(url)
            for num in topic["numbers"]:
                db.add_number(str(num))
        db.save()
        return db

    def test_chunk_count(self, merged_db):
        """모든 주제의 청크가 정확히 합산된다."""
        total_chunks = sum(
            len(t["chunks"]) for t in _load_topics()["topics"]
        )
        assert len(merged_db._chunks) == total_chunks

    def test_cross_topic_exact_match(self, merged_db):
        """다른 주제의 청크도 exact match로 검색 가능하다."""
        # 기후변화 주제의 청크
        assert merged_db.exact_match("한국의 탄소 배출량은 2023년 기준 약 6.1억 톤이다.")
        # 반도체 주제의 청크
        assert merged_db.exact_match("SK하이닉스의 HBM3E 메모리 시장 점유율은 약 53%이다.")
        # 바이오 주제의 청크
        assert merged_db.exact_match("셀트리온의 바이오시밀러 글로벌 매출은 2024년 약 3.2조 원이다.")

    def test_url_from_different_topics(self, merged_db):
        """다른 주제의 URL도 검증 가능하다."""
        assert merged_db.url_exists("https://me.go.kr/climate-report-2023")
        assert merged_db.url_exists("https://oecd.ai/policy")
        assert merged_db.url_exists("https://motie.go.kr/semiconductor-export-2024")
        assert not merged_db.url_exists("https://nonexistent.com")

    def test_db_persistence_and_reload(self, tmp_path):
        """DB 저장 후 다시 로드했을 때 동일한 데이터."""
        db_path = tmp_path / "persist_test.json"
        db1 = LocalChunkDB(db_path=db_path)
        db1.add_chunk("테스트 청크", source="테스트")
        db1.add_url("https://test.com")
        db1.add_number("42")
        db1.save()

        db2 = LocalChunkDB(db_path=db_path)
        assert len(db2._chunks) == 1
        assert db2.exact_match("테스트 청크")
        assert db2.url_exists("https://test.com")
        assert db2.number_in_context("42")


# ── Synthesis 노드 다양한 입력 테스트 ──


class TestSynthesisVariedInputs:
    """워커 결과의 다양한 패턴에 대한 Synthesis 노드 동작."""

    def test_single_worker_result(self):
        model = FakeStructuredChatModel(responses=["단일 결과 기반 초안"])
        node = make_synthesis_node(model)
        result = node({"worker_results": [
            {"task_id": "t1", "data": "유일한 결과"},
        ]})
        assert result["draft"] == "단일 결과 기반 초안"

    def test_many_worker_results(self):
        model = FakeStructuredChatModel(responses=["종합 보고서"])
        node = make_synthesis_node(model)
        results = [
            {"task_id": f"t{i}", "data": f"결과_{i}"}
            for i in range(10)
        ]
        result = node({"worker_results": results})
        assert result["draft"] == "종합 보고서"

    def test_empty_worker_data(self):
        """워커 데이터가 빈 경우에도 합성이 진행된다."""
        model = FakeStructuredChatModel(responses=["데이터 부족 보고서"])
        node = make_synthesis_node(model)
        result = node({"worker_results": [
            {"task_id": "t1", "data": ""},
        ]})
        assert result["draft"] == "데이터 부족 보고서"


# ── Finalize 노드 다양한 입력 테스트 ──


class TestFinalizeVariedInputs:
    """finalize 노드의 다양한 입력에 대한 동작."""

    def test_long_draft(self):
        long_text = "내용 " * 1000
        result = finalize_node({"draft": long_text})
        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["messages"][0].content) == len(long_text)

    def test_draft_with_special_characters(self):
        draft = "특수문자: ~!@#$%^&*()_+{}|:\"<>? 한글 English 123"
        result = finalize_node({"draft": draft})
        assert result["messages"][0].content == draft

    def test_draft_with_citations(self):
        draft = "[출처: 원본 텍스트] 내용 [ref: another source]"
        result = finalize_node({"draft": draft})
        assert result["messages"][0].content == draft
