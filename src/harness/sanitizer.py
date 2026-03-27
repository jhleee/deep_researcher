"""결정론적 Sanitizer.

LLM을 사용하지 않는 순수 코드 기반 검증기.
정규식과 Exact Match로 초안의 인용을 원본 데이터와 대조한다.
로컬 파일시스템의 JSON 기반 청크 DB를 사용한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


@dataclass
class SanitizeError:
    """Sanitizer 검증 오류."""

    error_type: str  # "FAKE_CITATION" | "MISSING_SOURCE" | "URL_INVALID" | "NUMBER_UNVERIFIED"
    location: str
    description: str


class LocalChunkDB:
    """로컬 JSON 파일 기반 청크 데이터베이스.

    원본 문서의 청크, URL, 수치 데이터를 저장하고 검색한다.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else None
        self._chunks: list[dict[str, Any]] = []
        self._urls: set[str] = set()
        self._numbers: set[str] = set()

        if self.db_path and self.db_path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.db_path.read_text(encoding="utf-8"))
        self._chunks = data.get("chunks", [])
        self._urls = set(data.get("urls", []))
        self._numbers = set(str(n) for n in data.get("numbers", []))

    def save(self) -> None:
        """현재 상태를 JSON 파일로 저장한다."""
        if not self.db_path:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self._chunks,
            "urls": sorted(self._urls),
            "numbers": sorted(self._numbers),
        }
        self.db_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_chunk(self, text: str, source: str = "", metadata: dict | None = None) -> None:
        """청크를 추가한다."""
        self._chunks.append({
            "text": text,
            "source": source,
            "metadata": metadata or {},
        })

    def add_url(self, url: str) -> None:
        self._urls.add(url)

    def add_number(self, number: str) -> None:
        self._numbers.add(str(number))

    def exact_match(self, citation: str) -> bool:
        """인용구가 청크 DB에 정확히 존재하는지 확인한다."""
        citation_lower = citation.strip().lower()
        return any(citation_lower in chunk["text"].lower() for chunk in self._chunks)

    def fuzzy_match(self, citation: str, threshold: float = 0.85) -> bool:
        """인용구가 청크 DB에 유사하게 존재하는지 확인한다."""
        citation_lower = citation.strip().lower()
        for chunk in self._chunks:
            chunk_text = chunk["text"].lower()
            # 긴 텍스트에서 슬라이딩 윈도우로 유사도 체크
            cite_len = len(citation_lower)
            for i in range(0, max(1, len(chunk_text) - cite_len + 1), cite_len // 2 or 1):
                window = chunk_text[i : i + cite_len + 20]
                ratio = SequenceMatcher(None, citation_lower, window).ratio()
                if ratio >= threshold:
                    return True
        return False

    def url_exists(self, url: str) -> bool:
        return url.strip() in self._urls

    def number_in_context(self, number: str) -> bool:
        return str(number).strip() in self._numbers


class DeterministicSanitizer:
    """결정론적 검증기.

    LLM 없이 정규식과 Exact Match로 초안을 검증한다.
    """

    CITATION_PATTERN = re.compile(r"\[출처:\s*(.+?)\]|\[ref:\s*(.+?)\]")
    URL_PATTERN = re.compile(r"https?://[\S]+")
    NUMBER_PATTERN = re.compile(r"(\d+\.?\d*)%|\$(\d+[.,]?\d*)")

    def __init__(self, chunk_db: LocalChunkDB):
        self.chunk_db = chunk_db

    def validate(self, draft: str) -> list[SanitizeError]:
        """초안을 검증하고 오류 목록을 반환한다."""
        errors: list[SanitizeError] = []

        # 1. 인용구 추출 및 원본 대조
        for match in self.CITATION_PATTERN.finditer(draft):
            cite = match.group(1) or match.group(2)
            if not self.chunk_db.exact_match(cite):
                if not self.chunk_db.fuzzy_match(cite, threshold=0.85):
                    errors.append(SanitizeError(
                        error_type="FAKE_CITATION",
                        location=f"pos {match.start()}",
                        description=f"인용구 '{cite[:50]}...'는 원본에 존재하지 않음.",
                    ))

        # 2. URL 유효성 검사
        for match in self.URL_PATTERN.finditer(draft):
            url = match.group()
            if not self.chunk_db.url_exists(url):
                errors.append(SanitizeError(
                    error_type="URL_INVALID",
                    location=f"pos {match.start()}",
                    description=f"URL '{url}'는 수집된 소스에 없음.",
                ))

        # 3. 수치 데이터 교차 검증
        for match in self.NUMBER_PATTERN.finditer(draft):
            num = match.group(1) or match.group(2)
            if num and not self.chunk_db.number_in_context(num):
                errors.append(SanitizeError(
                    error_type="NUMBER_UNVERIFIED",
                    location=f"pos {match.start()}",
                    description=f"수치 '{num}'의 출처를 확인할 수 없음.",
                ))

        return errors

    def validate_as_strings(self, draft: str) -> list[str]:
        """validate()의 편의 래퍼. 오류를 문자열 목록으로 반환한다."""
        return [
            f"[{e.error_type}] {e.description} (at {e.location})"
            for e in self.validate(draft)
        ]
