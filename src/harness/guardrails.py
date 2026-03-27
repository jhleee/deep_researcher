"""입출력 가드레일.

LLM 호출 전후에 적용되는 결정론적 검증 레이어.
외부 서비스 없이 정규식 및 규칙 기반으로 동작한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class GuardrailResult:
    """가드레일 검사 결과."""

    passed: bool
    sanitized_content: str
    reason: str = ""
    rejection_message: str = ""


# ── PII 패턴 ──

PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone_kr": re.compile(r"01[016789]-?\d{3,4}-?\d{4}"),
    "ssn_kr": re.compile(r"\d{6}-?[1-4]\d{6}"),
    "credit_card": re.compile(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"),
}

# ── 프롬프트 인젝션 패턴 ──

INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"(시스템\s*프롬프트|system\s*prompt).*?(무시|ignore|override|잊어)", re.IGNORECASE),
    re.compile(r"(forget|disregard|bypass)\s+(your|all)\s+(instructions|rules)", re.IGNORECASE),
    re.compile(r"(이전|위의)\s*(지시|명령|규칙).*?(무시|잊어|취소)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"ignore\s+(previous|above|all)\s+instructions", re.IGNORECASE),
]


class InputGuardrail:
    """입력 가드레일: PII 마스킹, 프롬프트 인젝션 차단, 토큰 길이 제어."""

    def __init__(
        self,
        strict: bool = True,
        max_input_length: int = 10_000,
        pii_mask: str = "[REDACTED]",
    ):
        self.strict = strict
        self.max_input_length = max_input_length
        self.pii_mask = pii_mask

    def check(self, content: str) -> GuardrailResult:
        """입력 콘텐츠를 검사하고 정제된 결과를 반환한다."""
        if not content or not content.strip():
            return GuardrailResult(
                passed=False,
                sanitized_content="",
                reason="EMPTY_INPUT",
                rejection_message="입력이 비어 있습니다. 질문을 입력해 주세요.",
            )

        # 1. 토큰 길이 제한
        if len(content) > self.max_input_length:
            return GuardrailResult(
                passed=False,
                sanitized_content="",
                reason="INPUT_TOO_LONG",
                rejection_message=f"입력이 너무 깁니다 (최대 {self.max_input_length}자).",
            )

        # 2. 프롬프트 인젝션 감지
        for pattern in INJECTION_PATTERNS:
            if pattern.search(content):
                if self.strict:
                    return GuardrailResult(
                        passed=False,
                        sanitized_content="",
                        reason="INJECTION_DETECTED",
                        rejection_message="요청을 처리할 수 없습니다.",
                    )

        # 3. PII 마스킹 (통과는 시키되 민감 정보를 마스킹)
        sanitized = content
        for pii_type, pattern in PII_PATTERNS.items():
            sanitized = pattern.sub(self.pii_mask, sanitized)

        return GuardrailResult(passed=True, sanitized_content=sanitized)


# ── 출력 필터 패턴 ──

HARMFUL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(폭탄|bomb)\s*(만들|제조|make|build)", re.IGNORECASE),
    re.compile(r"(해킹|hacking)\s*(방법|how\s*to)", re.IGNORECASE),
]


class OutputGuardrail:
    """출력 가드레일: 유해성 필터, 포맷 검증."""

    def __init__(self, strict: bool = True, max_output_length: int = 50_000):
        self.strict = strict
        self.max_output_length = max_output_length

    def check(self, content: str) -> GuardrailResult:
        """출력 콘텐츠를 검사하고 정제된 결과를 반환한다."""
        if not content:
            return GuardrailResult(passed=True, sanitized_content="")

        # 1. 유해 콘텐츠 필터
        for pattern in HARMFUL_PATTERNS:
            if pattern.search(content):
                return GuardrailResult(
                    passed=False,
                    sanitized_content="요청하신 내용에 대해 답변드리기 어렵습니다.",
                    reason="HARMFUL_CONTENT",
                )

        # 2. 출력 길이 제한 (잘라내기)
        if len(content) > self.max_output_length:
            truncated = content[: self.max_output_length] + "\n\n[출력이 잘렸습니다]"
            return GuardrailResult(
                passed=True,
                sanitized_content=truncated,
                reason="OUTPUT_TRUNCATED",
            )

        return GuardrailResult(passed=True, sanitized_content=content)
