"""공통 상태 스키마 및 하네스 설정.

모든 아키텍처가 공유하는 기본 상태 타입과 런타임 설정을 정의한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import operator

from langchain_core.messages import BaseMessage


class HarnessState(TypedDict):
    """모든 아키텍처가 공유하는 기본 상태 스키마.

    Attributes:
        messages: 대화 메시지 이력 (reducer: append).
        plan: 실행 계획 (하위 작업 목록).
        artifacts: 생성된 산출물 경로/내용 (reducer: append).
        metadata: 실행 메타데이터 (자유 형식).
        error_log: 오류 기록 (reducer: append).
        iteration_count: 현재 반복 횟수.
    """

    messages: Annotated[list[BaseMessage], operator.add]
    plan: list[dict[str, Any]]
    artifacts: Annotated[list[str], operator.add]
    metadata: dict[str, Any]
    error_log: Annotated[list[str], operator.add]
    iteration_count: int


@dataclass
class HarnessConfig:
    """하네스 런타임 설정.

    Attributes:
        max_iterations: 최대 반복 횟수 (무한루프 방지).
        max_tokens_budget: 토큰 예산 상한.
        checkpoint_backend: 체크포인터 백엔드 ("memory" | "json_file" | "sqlite" | "postgres").
        checkpoint_dir: 파일 기반 체크포인터의 저장 디렉토리.
        tracing_enabled: LangSmith 트레이싱 활성화 여부.
        human_in_the_loop: HITL 승인 게이트 활성화 여부.
        guardrail_strict: 엄격 모드 가드레일 활성화 여부.
    """

    max_iterations: int = 10
    max_tokens_budget: int = 100_000
    checkpoint_backend: str = "memory"  # 로컬 기본값: memory
    checkpoint_dir: str = "checkpoints"
    tracing_enabled: bool = False  # 로컬 기본값: 비활성
    human_in_the_loop: bool = False
    guardrail_strict: bool = True


def create_initial_state() -> dict[str, Any]:
    """빈 초기 상태 딕셔너리를 생성한다."""
    return {
        "messages": [],
        "plan": [],
        "artifacts": [],
        "metadata": {},
        "error_log": [],
        "iteration_count": 0,
    }
