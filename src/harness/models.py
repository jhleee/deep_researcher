"""공유 Pydantic 모델 및 테스트용 모델 유틸리티.

아키텍처 간 공유되는 데이터 모델과 LLM 없이 테스트 가능한
FakeStructuredChatModel을 정의한다.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


# ── Pydantic 모델 (아키텍처 간 공유) ──


class SubTask(BaseModel):
    """하위 작업 단위."""

    id: str = Field(description="고유 작업 식별자")
    description: str = Field(description="작업 설명")
    tool_name: str | None = Field(default=None, description="사용할 도구 이름")
    dependencies: list[str] = Field(default_factory=list, description="선행 작업 ID 목록")


class ExecutionPlan(BaseModel):
    """실행 계획."""

    goal: str = Field(description="최종 목표")
    sub_tasks: list[SubTask] = Field(description="하위 작업 목록")


class ResearchQuestions(BaseModel):
    """검증 질문 목록."""

    questions: list[str] = Field(description="세부 질문 목록")


# ── FakeStructuredChatModel (테스트용) ──


class FakeStructuredChatModel(FakeListChatModel):
    """with_structured_output을 지원하는 테스트용 ChatModel.

    FakeListChatModel의 responses에 JSON 문자열을 넣으면
    with_structured_output(Schema).invoke()가 Pydantic 모델 인스턴스를 반환한다.
    """

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> Any:
        """Pydantic 모델 파싱 체인을 반환한다."""

        def _parse(ai_message: Any) -> BaseModel:
            return schema.model_validate_json(ai_message.content)

        return self | RunnableLambda(_parse)


# ── 헬퍼 함수 ──


def parse_questions(text: str) -> list[str]:
    """LLM 응답 텍스트에서 질문 목록을 추출한다.

    번호 매김(1. 2.) 또는 줄바꿈 기반으로 파싱하며,
    물음표(?)로 끝나는 줄을 우선 추출한다.
    """
    if not text or not text.strip():
        return []

    lines = text.strip().split("\n")
    questions: list[str] = []

    for line in lines:
        # 번호 접두사 제거: "1. ", "1) ", "- " 등
        cleaned = re.sub(r"^[\d]+[.\)]\s*", "", line.strip())
        cleaned = re.sub(r"^[-*]\s*", "", cleaned)
        if not cleaned:
            continue
        if cleaned.endswith("?"):
            questions.append(cleaned)

    # 물음표가 없으면 비어있지 않은 줄을 질문으로 간주
    if not questions:
        questions = [
            re.sub(r"^[\d]+[.\)]\s*", "", re.sub(r"^[-*]\s*", "", line.strip()))
            for line in lines
            if line.strip()
        ]

    return questions


# 페르소나 키워드 매핑
_CRITICAL_KEYWORDS = ["오류", "문제", "위험", "반박", "한계", "실패", "결함"]
_DOMAIN_KEYWORDS = ["정의", "원리", "메커니즘", "구조", "특성", "개념", "이론"]

PERSONAS = ("비판적 검토자", "도메인 전문가", "반론 제기자")


def assign_persona(question: str) -> str:
    """질문 내용에 따라 페르소나를 할당한다.

    키워드 기반 휴리스틱:
    - 비판적 키워드 → 비판적 검토자
    - 도메인 키워드 → 도메인 전문가
    - 그 외 → 반론 제기자
    """
    for kw in _CRITICAL_KEYWORDS:
        if kw in question:
            return "비판적 검토자"
    for kw in _DOMAIN_KEYWORDS:
        if kw in question:
            return "도메인 전문가"
    return "반론 제기자"
