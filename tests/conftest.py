"""공통 테스트 픽스처."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from harness.base import HarnessConfig, create_initial_state
from harness.checkpointer import JsonFileCheckpointer
from harness.guardrails import InputGuardrail, OutputGuardrail
from harness.models import ExecutionPlan, FakeStructuredChatModel, SubTask
from harness.sanitizer import DeterministicSanitizer, LocalChunkDB


@pytest.fixture
def config() -> HarnessConfig:
    """기본 테스트 설정."""
    return HarnessConfig(
        max_iterations=5,
        checkpoint_backend="memory",
        tracing_enabled=False,
        guardrail_strict=True,
    )


@pytest.fixture
def initial_state() -> dict:
    """빈 초기 상태."""
    return create_initial_state()


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """임시 디렉토리."""
    return tmp_path


@pytest.fixture
def json_checkpointer(tmp_dir: Path) -> JsonFileCheckpointer:
    """임시 디렉토리 기반 JSON 체크포인터."""
    return JsonFileCheckpointer(directory=str(tmp_dir / "checkpoints"))


@pytest.fixture
def chunk_db(tmp_dir: Path) -> LocalChunkDB:
    """테스트용 청크 DB."""
    db_path = tmp_dir / "chunks.json"
    db = LocalChunkDB(db_path=db_path)
    # 테스트 데이터 시딩
    db.add_chunk("서울의 인구는 약 950만 명이다.", source="통계청 2024")
    db.add_chunk("한국의 GDP는 2024년 기준 약 1.7조 달러이다.", source="한국은행")
    db.add_url("https://kostat.go.kr/report/2024")
    db.add_number("950")
    db.add_number("1.7")
    db.save()
    return db


@pytest.fixture
def sanitizer(chunk_db: LocalChunkDB) -> DeterministicSanitizer:
    """테스트용 Sanitizer."""
    return DeterministicSanitizer(chunk_db=chunk_db)


@pytest.fixture
def input_guard() -> InputGuardrail:
    return InputGuardrail(strict=True)


@pytest.fixture
def output_guard() -> OutputGuardrail:
    return OutputGuardrail(strict=True)


# ── Zero Hallucination Pipeline 픽스처 ──


@pytest.fixture
def sample_execution_plan() -> ExecutionPlan:
    """테스트용 실행 계획."""
    return ExecutionPlan(
        goal="테스트 분석",
        sub_tasks=[
            SubTask(id="t1", description="첫 번째 작업"),
            SubTask(id="t2", description="두 번째 작업", dependencies=["t1"]),
        ],
    )


@pytest.fixture
def fake_planner_model(sample_execution_plan: ExecutionPlan) -> FakeStructuredChatModel:
    """ExecutionPlan JSON을 반환하는 planner 모델."""
    return FakeStructuredChatModel(
        responses=[sample_execution_plan.model_dump_json()]
    )


@pytest.fixture
def fake_worker_model() -> FakeStructuredChatModel:
    """STORM 워커용 모델. 3 페르소나 응답 + 1 압축 응답 × 2 작업 = 8 응답."""
    responses = []
    for task_num in range(2):
        responses.extend([
            f"[비판적 검토자] 작업{task_num} 분석 결과",
            f"[도메인 전문가] 작업{task_num} 전문 의견",
            f"[반론 제기자] 작업{task_num} 반론",
            f"작업{task_num} 핵심 사실 압축 결과",
        ])
    return FakeStructuredChatModel(responses=responses)


@pytest.fixture
def fake_verifier_model() -> FakeStructuredChatModel:
    """CoVe 검증 모델. 질문 생성 + 팩토리드 검증 + 교차 대조."""
    return FakeStructuredChatModel(responses=[
        "1. 첫 번째 사실이 정확한가?\n2. 두 번째 사실이 정확한가?",
        "첫 번째 사실은 정확합니다.",
        "두 번째 사실은 정확합니다.",
        "교차 검증 완료. 초안이 정확합니다.",
    ])


@pytest.fixture
def fake_synthesizer_model() -> FakeStructuredChatModel:
    """초안 합성 + 교정 모델."""
    return FakeStructuredChatModel(responses=[
        "서울의 인구는 약 950만 명이다. [출처: 서울의 인구는 약 950만 명이다.]",
        "교정된 초안입니다.",
    ])


@pytest.fixture
def zh_initial_state(initial_state: dict) -> dict:
    """ZeroHallucinationState 초기값."""
    return {
        **initial_state,
        "execution_plan": None,
        "worker_results": [],
        "draft": None,
        "verification_questions": [],
        "verification_answers": [],
        "sanitizer_errors": [],
        "correction_count": 0,
    }
