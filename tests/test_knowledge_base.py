"""LocalKnowledgeBase 유닛 테스트."""

from __future__ import annotations

import json

import pytest

from harness.knowledge_base import Chunk, LocalFileSource, LocalKnowledgeBase
from harness.sanitizer import LocalChunkDB


class TestLocalFileSource:
    """LocalFileSource 테스트."""

    def test_load_empty_directory(self, tmp_path):
        source = LocalFileSource(tmp_path)
        chunks = source.load()
        assert chunks == []

    def test_load_text_file(self, tmp_path):
        (tmp_path / "test.txt").write_text("Hello world.\n\nThis is a test.", encoding="utf-8")
        source = LocalFileSource(tmp_path)
        chunks = source.load()
        assert len(chunks) >= 1
        assert "Hello world" in chunks[0].text
        assert chunks[0].source == "test.txt"

    def test_load_skips_unsupported_extensions(self, tmp_path):
        (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")
        (tmp_path / "doc.txt").write_text("visible", encoding="utf-8")
        source = LocalFileSource(tmp_path)
        chunks = source.load()
        sources = {c.source for c in chunks}
        assert "doc.txt" in sources
        assert "data.bin" not in sources

    def test_load_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("# Nested doc", encoding="utf-8")
        source = LocalFileSource(tmp_path)
        chunks = source.load()
        assert any("sub" in c.source for c in chunks)

    def test_chunking_large_file(self, tmp_path):
        text = "A" * 500 + "\n\n" + "B" * 500 + "\n\n" + "C" * 500
        (tmp_path / "large.txt").write_text(text, encoding="utf-8")
        source = LocalFileSource(tmp_path, chunk_size=600, chunk_overlap=100)
        chunks = source.load()
        assert len(chunks) >= 2

    def test_search_keyword_matching(self, tmp_path):
        (tmp_path / "a.txt").write_text("Python is a programming language.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("The weather is sunny today.", encoding="utf-8")
        source = LocalFileSource(tmp_path)
        source.load()
        results = source.search("programming language", top_k=1)
        assert len(results) == 1
        assert "programming" in results[0].text.lower()

    def test_search_returns_empty_when_not_loaded(self, tmp_path):
        source = LocalFileSource(tmp_path)
        assert source.search("anything") == []

    def test_nonexistent_directory(self):
        source = LocalFileSource("/nonexistent/path/12345")
        chunks = source.load()
        assert chunks == []


class TestLocalKnowledgeBase:
    """LocalKnowledgeBase 통합 테스트."""

    def test_load_and_stats(self, tmp_path):
        (tmp_path / "doc1.txt").write_text("First document content.", encoding="utf-8")
        (tmp_path / "doc2.md").write_text("# Second\n\nMarkdown content.", encoding="utf-8")
        kb = LocalKnowledgeBase(directory=tmp_path)
        stats = kb.load()
        assert stats["files"] == 2
        assert stats["chunks"] >= 2
        assert stats["chars"] > 0
        assert kb.is_loaded

    def test_chunk_db_integration(self, tmp_path):
        """로딩 후 ChunkDB에 청크가 등록되는지 확인."""
        (tmp_path / "source.txt").write_text(
            "한국의 GDP는 2025년 기준 1.8조 달러이다.", encoding="utf-8"
        )
        chunk_db = LocalChunkDB()
        kb = LocalKnowledgeBase(directory=tmp_path, chunk_db=chunk_db)
        kb.load()
        assert chunk_db.fuzzy_match("한국의 GDP는 2025년 기준 1.8조 달러")

    def test_search(self, tmp_path):
        (tmp_path / "ml.txt").write_text(
            "Machine learning is a subset of artificial intelligence.",
            encoding="utf-8",
        )
        (tmp_path / "cooking.txt").write_text(
            "Boil water for 10 minutes to cook pasta.",
            encoding="utf-8",
        )
        kb = LocalKnowledgeBase(directory=tmp_path)
        kb.load()
        results = kb.search("artificial intelligence", top_k=1)
        assert len(results) == 1
        assert "machine learning" in results[0].text.lower()

    def test_get_context_for_task(self, tmp_path):
        (tmp_path / "report.txt").write_text(
            "2024년 전기차 판매량은 전년 대비 30% 증가했다.\n\n"
            "주요 제조사는 Tesla, BYD, Hyundai 순이다.",
            encoding="utf-8",
        )
        kb = LocalKnowledgeBase(directory=tmp_path)
        kb.load()
        context = kb.get_context_for_task("전기차 시장 분석")
        assert "참고 자료:" in context
        assert "report.txt" in context

    def test_get_context_empty_kb(self, tmp_path):
        kb = LocalKnowledgeBase(directory=tmp_path)
        kb.load()
        context = kb.get_context_for_task("anything")
        assert context == ""

    def test_not_loaded_returns_false(self, tmp_path):
        kb = LocalKnowledgeBase(directory=tmp_path)
        assert not kb.is_loaded

    def test_max_chars_limit(self, tmp_path):
        (tmp_path / "big.txt").write_text("X" * 5000, encoding="utf-8")
        kb = LocalKnowledgeBase(directory=tmp_path, chunk_size=2000)
        kb.load()
        context = kb.get_context_for_task("test", max_chars=500)
        assert len(context) < 700  # 참고 자료: 헤더 + 약간의 여유

    def test_add_multiple_sources(self, tmp_path):
        dir1 = tmp_path / "src1"
        dir2 = tmp_path / "src2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "a.txt").write_text("Source one content.", encoding="utf-8")
        (dir2 / "b.txt").write_text("Source two content.", encoding="utf-8")

        kb = LocalKnowledgeBase()
        kb.add_source(LocalFileSource(dir1))
        kb.add_source(LocalFileSource(dir2))
        stats = kb.load()
        assert stats["files"] == 2
        assert stats["chunks"] >= 2
