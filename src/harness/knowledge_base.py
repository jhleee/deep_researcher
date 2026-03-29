"""로컬 지식 베이스.

디렉토리의 파일을 읽어 청크로 분할하고, ChunkDB에 등록하며,
질문 기반 관련 청크 검색을 제공한다.

향후 웹검색·API 등 외부 소스를 추가할 수 있도록 RetrievalSource 인터페이스를 정의한다.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from harness.sanitizer import LocalChunkDB

# 지원 확장자
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log", ".html", ".xml", ".py", ".rst"}


@dataclass
class Chunk:
    """분할된 텍스트 청크."""

    text: str
    source: str  # 파일 경로
    index: int  # 청크 번호
    metadata: dict = field(default_factory=dict)


# ── 소스 인터페이스 (확장 포인트) ──


class RetrievalSource(ABC):
    """자료 소스 인터페이스.

    로컬 파일, 웹 검색, 데이터베이스 등 다양한 소스를 동일한 인터페이스로 사용한다.
    향후 구현 예시:
        - WebSearchSource: 웹 검색 결과를 청크로 변환
        - APISource: REST API 응답을 청크로 변환
        - VectorStoreSource: 벡터 DB에서 유사도 검색
    """

    @abstractmethod
    def load(self) -> list[Chunk]:
        """소스에서 청크를 로드한다."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """질문과 관련된 청크를 검색한다."""


# ── 로컬 파일 소스 ──


class LocalFileSource(RetrievalSource):
    """로컬 디렉토리의 파일을 읽어 청크로 분할하는 소스.

    Args:
        directory: 자료 디렉토리 경로.
        chunk_size: 청크당 최대 문자 수.
        chunk_overlap: 청크 간 겹치는 문자 수.
        extensions: 처리할 파일 확장자 집합. None이면 SUPPORTED_EXTENSIONS 사용.
    """

    def __init__(
        self,
        directory: str | Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        extensions: set[str] | None = None,
    ):
        self.directory = Path(directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extensions = extensions or SUPPORTED_EXTENSIONS
        self._chunks: list[Chunk] = []

    def load(self) -> list[Chunk]:
        """디렉토리의 모든 파일을 읽어 청크로 분할한다."""
        self._chunks = []
        if not self.directory.exists():
            return []

        for path in sorted(self.directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.extensions:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            if not text.strip():
                continue

            rel_path = str(path.relative_to(self.directory))
            chunks = self._split_text(text, rel_path)
            self._chunks.extend(chunks)

        return list(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """간단한 키워드+유사도 기반 검색.

        향후 임베딩 기반 검색으로 교체 가능한 자리.
        """
        if not self._chunks:
            return []

        scored: list[tuple[float, Chunk]] = []
        query_lower = query.lower()
        query_keywords = set(re.findall(r"\w{2,}", query_lower))

        for chunk in self._chunks:
            chunk_lower = chunk.text.lower()

            # 1. 키워드 매칭 점수
            chunk_words = set(re.findall(r"\w{2,}", chunk_lower))
            if query_keywords:
                keyword_score = len(query_keywords & chunk_words) / len(query_keywords)
            else:
                keyword_score = 0.0

            # 2. 부분 문자열 유사도
            similarity = SequenceMatcher(
                None,
                query_lower[:200],
                chunk_lower[:200],
            ).ratio()

            # 3. 정확한 구문 매치 보너스
            exact_bonus = 0.3 if query_lower[:30] in chunk_lower else 0.0

            score = keyword_score * 0.5 + similarity * 0.3 + exact_bonus * 0.2
            scored.append((score, chunk))

        scored.sort(key=lambda x: -x[0])
        return [chunk for _, chunk in scored[:top_k]]

    def _split_text(self, text: str, source: str) -> list[Chunk]:
        """텍스트를 고정 크기 청크로 분할한다.

        단락 경계(빈 줄)를 우선 사용하고, 그래도 크면 문장 단위로 분할한다.
        """
        # 단락 단위로 먼저 분할
        paragraphs = re.split(r"\n\s*\n", text)

        chunks: list[Chunk] = []
        current = ""
        idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 1 <= self.chunk_size:
                current = f"{current}\n{para}" if current else para
            else:
                if current:
                    chunks.append(Chunk(text=current, source=source, index=idx))
                    idx += 1
                    # 오버랩: 현재 청크 끝부분을 다음 청크 시작으로
                    if self.chunk_overlap > 0:
                        overlap_text = current[-self.chunk_overlap:]
                        current = f"{overlap_text}\n{para}"
                    else:
                        current = para
                else:
                    # 단일 단락이 chunk_size보다 큰 경우 → 강제 분할
                    for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
                        chunk_text = para[i:i + self.chunk_size]
                        chunks.append(Chunk(text=chunk_text, source=source, index=idx))
                        idx += 1
                    current = ""

        if current.strip():
            chunks.append(Chunk(text=current.strip(), source=source, index=idx))

        return chunks


# ── 통합 지식 베이스 ──


class LocalKnowledgeBase:
    """로컬 파일 기반 지식 베이스.

    1. 디렉토리에서 파일을 읽어 청크로 분할
    2. ChunkDB에 등록 (Sanitizer 검증과 연동)
    3. 질문 기반 관련 청크 검색 제공

    향후 확장:
        kb = LocalKnowledgeBase(...)
        kb.add_source(WebSearchSource(...))
        kb.add_source(VectorStoreSource(...))
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        chunk_db: LocalChunkDB | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_db = chunk_db or LocalChunkDB()
        self._sources: list[RetrievalSource] = []
        self._all_chunks: list[Chunk] = []
        self.stats: dict[str, int] = {"files": 0, "chunks": 0, "chars": 0}

        if directory:
            self.add_source(
                LocalFileSource(directory, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            )

    def add_source(self, source: RetrievalSource) -> None:
        """자료 소스를 추가한다."""
        self._sources.append(source)

    def load(self) -> dict[str, int]:
        """모든 소스에서 자료를 로드하고 ChunkDB에 등록한다.

        Returns:
            로딩 통계: {"files": N, "chunks": N, "chars": N}
        """
        self._all_chunks = []
        files_seen: set[str] = set()

        for source in self._sources:
            chunks = source.load()
            self._all_chunks.extend(chunks)
            for c in chunks:
                files_seen.add(c.source)

        # ChunkDB에 등록 (Sanitizer 검증 연동)
        for chunk in self._all_chunks:
            self.chunk_db.add_chunk(
                text=chunk.text,
                source=chunk.source,
                metadata={"index": chunk.index},
            )

        self.stats = {
            "files": len(files_seen),
            "chunks": len(self._all_chunks),
            "chars": sum(len(c.text) for c in self._all_chunks),
        }
        return dict(self.stats)

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """모든 소스에서 관련 청크를 검색한다."""
        results: list[Chunk] = []
        for source in self._sources:
            results.extend(source.search(query, top_k=top_k))

        # 중복 제거 후 상위 top_k
        seen: set[str] = set()
        unique: list[Chunk] = []
        for c in results:
            key = f"{c.source}:{c.index}"
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique[:top_k]

    def get_context_for_task(self, task_description: str, max_chars: int = 3000) -> str:
        """작업 설명에 맞는 컨텍스트 텍스트를 생성한다.

        Worker 노드 프롬프트에 직접 삽입할 수 있는 형태로 반환한다.
        """
        chunks = self.search(task_description, top_k=10)
        context_parts: list[str] = []
        total = 0

        for chunk in chunks:
            if total + len(chunk.text) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    context_parts.append(
                        f"[{chunk.source} #{chunk.index}]\n{chunk.text[:remaining]}..."
                    )
                break
            context_parts.append(f"[{chunk.source} #{chunk.index}]\n{chunk.text}")
            total += len(chunk.text)

        if not context_parts:
            return ""

        return "참고 자료:\n" + "\n---\n".join(context_parts)

    @property
    def is_loaded(self) -> bool:
        return len(self._all_chunks) > 0
