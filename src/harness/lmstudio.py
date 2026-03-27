"""LM Studio 로컬 모델 래퍼.

LM Studio의 OpenAI 호환 API를 LangChain ChatModel로 감싸며,
thinking 모델(<think> 태그)의 출력을 자동으로 정리한다.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# <think>...</think> 패턴 (단일/다중 줄)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def strip_think_tags(text: str) -> str:
    """<think>...</think> 태그를 제거하고 앞뒤 공백을 정리한다."""
    return _THINK_PATTERN.sub("", text).strip()


def _repair_truncated_json(json_str: str) -> str:
    """잘린 JSON 문자열을 최소한으로 복구한다.

    max_tokens 제한으로 잘린 JSON에서 열린 중괄호/대괄호를 닫아준다.
    """
    # 이미 유효한 JSON이면 그대로 반환
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    # 후행 쉼표와 불완전한 키/값 제거
    repaired = json_str.rstrip()
    # 마지막 불완전한 요소 제거 (쉼표 뒤의 불완전 객체)
    repaired = re.sub(r',\s*"[^"]*"?\s*:?\s*(?:"[^"]*)?$', "", repaired)
    repaired = re.sub(r',\s*\{[^}]*$', "", repaired)
    repaired = re.sub(r',\s*$', "", repaired)

    # 열린 괄호를 역순으로 닫기
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    repaired += "]" * max(0, open_brackets)
    repaired += "}" * max(0, open_braces)

    return repaired


def create_lmstudio_llm(
    base_url: str = "http://169.254.83.107:1234/v1",
    model: str = "qwen/qwen3.5-9b",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    **kwargs: Any,
) -> ChatOpenAI:
    """LM Studio 연결용 ChatOpenAI 인스턴스를 생성한다."""
    return ChatOpenAI(
        base_url=base_url,
        api_key="lm-studio",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


class ThinkingModelWrapper:
    """<think> 태그를 자동 제거하는 ChatModel 래퍼.

    LangChain의 invoke/with_structured_output 인터페이스를 유지하면서
    thinking 모델의 출력에서 추론 과정을 제거한다.
    """

    def __init__(self, llm: BaseChatModel, max_input_chars: int = 6000):
        self._llm = llm
        self._max_input_chars = max_input_chars

    def invoke(self, input: str | list[BaseMessage], **kwargs: Any) -> AIMessage:
        """LLM을 호출하고 <think> 태그를 제거한 AIMessage를 반환한다."""
        # 입력이 너무 길면 자르기 (context 초과 방지)
        if isinstance(input, str) and len(input) > self._max_input_chars:
            input = input[:self._max_input_chars] + "\n\n[입력이 잘렸습니다. 위 내용만으로 답변하라.]"
        result = self._llm.invoke(input, **kwargs)
        cleaned = strip_think_tags(result.content)
        return AIMessage(content=cleaned)

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> Any:
        """JSON 스키마 기반 구조화 출력 체인을 반환한다.

        1) LLM 호출 → 2) <think> 제거 → 3) JSON 추출/복구 → 4) Pydantic 파싱
        """
        # 간결한 스키마 예시를 생성 (전체 JSON Schema 대신)
        example = _build_schema_example(schema)

        def _parse_structured(input_text: str | list[BaseMessage]) -> BaseModel:
            if isinstance(input_text, str):
                augmented = (
                    f"{input_text}\n\n"
                    f"반드시 아래 JSON 형식으로만 응답하라. 코드 블록이나 설명 없이 순수 JSON만 출력.\n"
                    f"형식 예시:\n{example}"
                )
            else:
                augmented = input_text

            result = self._llm.invoke(augmented)
            cleaned = strip_think_tags(result.content)

            # JSON 추출
            json_str = _extract_json(cleaned)

            # 잘린 JSON 복구
            json_str = _repair_truncated_json(json_str)

            return schema.model_validate_json(json_str)

        return RunnableLambda(_parse_structured)


def _extract_json(text: str) -> str:
    """텍스트에서 JSON 객체를 추출한다."""
    # 1) ```json ... ``` 블록
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # 2) 중괄호로 시작하는 가장 큰 JSON
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text


def _build_schema_example(schema: type[BaseModel]) -> str:
    """Pydantic 모델에서 간결한 JSON 예시를 생성한다."""
    try:
        json_schema = schema.model_json_schema()
        return json.dumps(
            _schema_to_example(json_schema, json_schema.get("$defs", {})),
            ensure_ascii=False,
            indent=2,
        )
    except Exception:
        return json.dumps(
            schema.model_json_schema(), ensure_ascii=False, indent=2
        )


def _schema_to_example(schema: dict, defs: dict) -> Any:
    """JSON Schema를 예시 값으로 변환한다."""
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        return _schema_to_example(defs[ref_name], defs)

    schema_type = schema.get("type", "string")

    if schema_type == "object":
        result = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            result[prop_name] = _schema_to_example(prop_schema, defs)
        return result
    elif schema_type == "array":
        items = schema.get("items", {"type": "string"})
        return [_schema_to_example(items, defs)]
    elif schema_type == "string":
        desc = schema.get("description", prop_name if "prop_name" in dir() else "값")
        return f"<{desc}>"
    elif schema_type == "integer":
        return 0
    elif schema_type == "boolean":
        return False
    elif schema_type == "number":
        return 0.0

    # anyOf (Optional 필드 등)
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            if option.get("type") != "null":
                return _schema_to_example(option, defs)
        return None

    return "<값>"
