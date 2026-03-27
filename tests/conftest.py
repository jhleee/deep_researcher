"""공통 테스트 픽스처."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from harness.base import HarnessConfig, create_initial_state
from harness.checkpointer import JsonFileCheckpointer
from harness.guardrails import InputGuardrail, OutputGuardrail
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
