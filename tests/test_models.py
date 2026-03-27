"""models.py 유닛 테스트."""

import json

import pytest

from harness.models import (
    PERSONAS,
    ExecutionPlan,
    FakeStructuredChatModel,
    ResearchQuestions,
    SubTask,
    assign_persona,
    parse_questions,
)


class TestSubTask:
    """SubTask 모델 테스트."""

    def test_basic_creation(self):
        task = SubTask(id="t1", description="분석 수행")
        assert task.id == "t1"
        assert task.description == "분석 수행"

    def test_defaults(self):
        task = SubTask(id="t1", description="test")
        assert task.tool_name is None
        assert task.dependencies == []

    def test_with_all_fields(self):
        task = SubTask(
            id="t1", description="검색", tool_name="search", dependencies=["t0"]
        )
        assert task.tool_name == "search"
        assert task.dependencies == ["t0"]

    def test_json_round_trip(self):
        task = SubTask(id="t1", description="test", tool_name="tool_a")
        data = task.model_dump_json()
        restored = SubTask.model_validate_json(data)
        assert restored == task


class TestExecutionPlan:
    """ExecutionPlan 모델 테스트."""

    def test_creation(self):
        plan = ExecutionPlan(
            goal="테스트 목표",
            sub_tasks=[SubTask(id="t1", description="작업 1")],
        )
        assert plan.goal == "테스트 목표"
        assert len(plan.sub_tasks) == 1

    def test_multiple_tasks(self):
        plan = ExecutionPlan(
            goal="복합 분석",
            sub_tasks=[
                SubTask(id="t1", description="조사"),
                SubTask(id="t2", description="분석", dependencies=["t1"]),
                SubTask(id="t3", description="종합", dependencies=["t1", "t2"]),
            ],
        )
        assert len(plan.sub_tasks) == 3
        assert plan.sub_tasks[2].dependencies == ["t1", "t2"]

    def test_json_round_trip(self):
        plan = ExecutionPlan(
            goal="test",
            sub_tasks=[SubTask(id="t1", description="a")],
        )
        data = plan.model_dump_json()
        restored = ExecutionPlan.model_validate_json(data)
        assert restored == plan


class TestResearchQuestions:
    """ResearchQuestions 모델 테스트."""

    def test_creation(self):
        rq = ResearchQuestions(questions=["질문1?", "질문2?"])
        assert len(rq.questions) == 2

    def test_empty_questions(self):
        rq = ResearchQuestions(questions=[])
        assert rq.questions == []


class TestFakeStructuredChatModel:
    """FakeStructuredChatModel 테스트."""

    def test_invoke_returns_ai_message(self):
        model = FakeStructuredChatModel(responses=["hello world"])
        result = model.invoke("test input")
        assert result.content == "hello world"

    def test_with_structured_output(self):
        plan_json = json.dumps({
            "goal": "분석 수행",
            "sub_tasks": [{"id": "t1", "description": "첫 번째 작업"}],
        })
        model = FakeStructuredChatModel(responses=[plan_json])
        chain = model.with_structured_output(ExecutionPlan)
        result = chain.invoke("계획을 세워라")
        assert isinstance(result, ExecutionPlan)
        assert result.goal == "분석 수행"
        assert len(result.sub_tasks) == 1

    def test_structured_output_subtask(self):
        task_json = json.dumps({"id": "s1", "description": "테스트"})
        model = FakeStructuredChatModel(responses=[task_json])
        chain = model.with_structured_output(SubTask)
        result = chain.invoke("작업 생성")
        assert isinstance(result, SubTask)
        assert result.id == "s1"

    def test_multiple_invocations(self):
        model = FakeStructuredChatModel(responses=["first", "second", "third"])
        assert model.invoke("a").content == "first"
        assert model.invoke("b").content == "second"
        assert model.invoke("c").content == "third"


class TestParseQuestions:
    """parse_questions 테스트."""

    def test_numbered_questions(self):
        text = "1. 서울의 인구는?\n2. GDP 성장률은?\n3. 실업률은?"
        result = parse_questions(text)
        assert len(result) == 3
        assert result[0] == "서울의 인구는?"
        assert result[2] == "실업률은?"

    def test_dash_prefixed(self):
        text = "- 첫 번째 질문?\n- 두 번째 질문?"
        result = parse_questions(text)
        assert len(result) == 2

    def test_no_question_marks_fallback(self):
        text = "첫 번째 항목\n두 번째 항목"
        result = parse_questions(text)
        assert len(result) == 2
        assert result[0] == "첫 번째 항목"

    def test_empty_input(self):
        assert parse_questions("") == []
        assert parse_questions("   ") == []

    def test_mixed_format(self):
        text = "1. 이것은 질문인가?\n일반 텍스트\n2) 또 다른 질문?"
        result = parse_questions(text)
        # 물음표가 있는 줄만 추출
        assert len(result) == 2
        assert "이것은 질문인가?" in result
        assert "또 다른 질문?" in result

    def test_parenthesis_numbering(self):
        text = "1) 질문 A?\n2) 질문 B?"
        result = parse_questions(text)
        assert len(result) == 2
        assert result[0] == "질문 A?"


class TestAssignPersona:
    """assign_persona 테스트."""

    def test_critical_keyword(self):
        assert assign_persona("이 접근법의 문제점은 무엇인가?") == "비판적 검토자"
        assert assign_persona("오류 가능성을 분석하라") == "비판적 검토자"
        assert assign_persona("위험 요소를 파악하라") == "비판적 검토자"

    def test_domain_keyword(self):
        assert assign_persona("이 메커니즘의 원리는?") == "도메인 전문가"
        assert assign_persona("구조를 설명하라") == "도메인 전문가"
        assert assign_persona("개념을 정의하라") == "도메인 전문가"

    def test_default_persona(self):
        assert assign_persona("일반적인 질문입니다") == "반론 제기자"
        assert assign_persona("추가 정보를 찾아라") == "반론 제기자"

    def test_all_personas_reachable(self):
        results = {
            assign_persona("오류 분석"),
            assign_persona("개념 정의"),
            assign_persona("일반 질문"),
        }
        assert results == set(PERSONAS)
