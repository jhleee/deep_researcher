"""WebSearchSource 유닛 테스트.

agent-browser가 설치되어 있지 않아도 동작하는 모킹 테스트와,
설치된 환경에서만 실행되는 통합 테스트를 포함한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from harness.knowledge_base import Chunk
from harness.web_search import (
    AgentBrowser,
    SearchResult,
    WebSearchSource,
    _clean_page_text,
    _parse_search_results_from_snapshot,
    _url_encode,
)


# ── 유틸리��� 함수 테스트 ──


class TestUrlEncode:
    def test_basic(self):
        assert _url_encode("hello world") == "hello+world"

    def test_korean(self):
        encoded = _url_encode("한국어 검색")
        assert "+" in encoded
        assert "한국어" not in encoded  # 인코딩됨


class TestCleanPageText:
    def test_removes_short_lines(self):
        raw = "Menu\nHome\nThis is a meaningful paragraph with enough text."
        cleaned = _clean_page_text(raw)
        assert "Menu" not in cleaned
        assert "meaningful paragraph" in cleaned

    def test_truncates_to_max_chars(self):
        raw = "A" * 100 + "\n" + "B sentence with enough words. " * 50
        cleaned = _clean_page_text(raw, max_chars=200)
        assert len(cleaned) <= 200

    def test_preserves_korean_endings(self):
        raw = "짧은 문장입니다\n이것도 포함되어야 합니다"
        cleaned = _clean_page_text(raw)
        assert "포함되어야" in cleaned

    def test_empty_input(self):
        assert _clean_page_text("") == ""


class TestParseSearchResults:
    def test_link_pattern(self):
        snapshot = (
            'link "Example Article Title" url="https://example.com/article"\n'
            'link "Another Result Here" url="https://other.com/page"\n'
            'link "Google Account" url="https://accounts.google.com/login"\n'
        )
        results = _parse_search_results_from_snapshot(snapshot)
        assert len(results) == 2
        assert results[0].title == "Example Article Title"
        assert results[0].url == "https://example.com/article"
        # Google 내부 링크 제외 확인
        assert all("google.com" not in r.url for r in results)

    def test_empty_snapshot(self):
        assert _parse_search_results_from_snapshot("") == []

    def test_filters_short_titles_in_link_pattern(self):
        """짧은 제목은 link 패턴에서 필터되지만 fallback에서 URL을 제목으로 사용할 수 있다."""
        snapshot = 'link "Ab" url="https://example.com/x"\n'
        results = _parse_search_results_from_snapshot(snapshot)
        # link 패�� 필터 후 fallback이 잡을 수 있음
        for r in results:
            assert len(r.title) >= 3

    def test_fallback_href_pattern(self):
        snapshot = '<a href="https://example.com/page">Interesting Article Title</a>'
        results = _parse_search_results_from_snapshot(snapshot)
        assert len(results) == 1
        assert results[0].url == "https://example.com/page"

    def test_fallback_url_in_lines(self):
        snapshot = "Some Article Title\nhttps://example.com/article\n"
        results = _parse_search_results_from_snapshot(snapshot)
        assert len(results) >= 1


# ── AgentBrowser 모킹 테스트 ──


class TestAgentBrowser:
    @patch("shutil.which", return_value=None)
    def test_raises_if_not_installed(self, mock_which):
        with pytest.raises(RuntimeError, match="agent-browser가 설치"):
            AgentBrowser()

    @patch("shutil.which", return_value="/usr/bin/agent-browser")
    @patch("subprocess.run")
    def test_run_returns_stdout(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="hello\n", stderr=""
        )
        browser = AgentBrowser()
        result = browser.run("get", "title")
        assert result == "hello"

    @patch("shutil.which", return_value="/usr/bin/agent-browser")
    @patch("subprocess.run")
    def test_run_handles_error(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error msg"
        )
        browser = AgentBrowser()
        result = browser.run("invalid")
        assert result == ""

    @patch("shutil.which", return_value="/usr/bin/agent-browser")
    @patch("subprocess.run", side_effect=TimeoutError)
    def test_run_handles_timeout(self, mock_run, mock_which):
        import subprocess as sp

        mock_run.side_effect = sp.TimeoutExpired(cmd="test", timeout=30)
        browser = AgentBrowser()
        result = browser.run("open", "https://slow.example.com")
        assert result == ""


# ── WebSearchSource 모킹 테스트 ──


class TestWebSearchSource:
    def test_is_available_returns_false_when_not_installed(self):
        with patch("shutil.which", return_value=None):
            assert WebSearchSource.is_available() is False

    def test_is_available_returns_true_when_installed(self):
        with patch("shutil.which", return_value="/usr/bin/agent-browser"):
            assert WebSearchSource.is_available() is True

    def test_load_returns_cached_chunks(self):
        source = WebSearchSource.__new__(WebSearchSource)
        source._chunks = [
            Chunk(text="test", source="url", index=0),
        ]
        assert len(source.load()) == 1

    @patch("shutil.which", return_value="/usr/bin/agent-browser")
    @patch("subprocess.run")
    def test_search_full_flow(self, mock_run, mock_which):
        """전체 검색 플로우를 모킹으로 테스트."""
        call_count = 0

        def fake_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            stdout = ""

            args = cmd[1] if len(cmd) > 1 else ""

            if "open" in cmd:
                stdout = "opened"
            elif "snapshot" in cmd:
                stdout = (
                    'link "Test Article About AI" '
                    'url="https://example.com/ai-article"\n'
                    'link "Another AI Resource" '
                    'url="https://other.com/ai"\n'
                )
            elif "goto" in cmd:
                stdout = "navigated"
            elif cmd[1:3] == ["get", "text"]:
                stdout = (
                    "This is a detailed article about artificial intelligence. "
                    "Machine learning is a subset of AI that focuses on training "
                    "algorithms to learn from data. Deep learning uses neural "
                    "networks with multiple layers to process complex patterns."
                )
            elif "close" in cmd:
                stdout = "closed"

            return MagicMock(returncode=0, stdout=stdout, stderr="")

        mock_run.side_effect = fake_run

        source = WebSearchSource(max_results=2, page_timeout=10)
        chunks = source.search("artificial intelligence", top_k=2)

        assert len(chunks) >= 1
        assert any("artificial intelligence" in c.text.lower() for c in chunks)

    @patch("shutil.which", return_value=None)
    def test_search_returns_empty_when_not_installed(self, mock_which):
        source = WebSearchSource()
        chunks = source.search("test query")
        assert chunks == []


# ── 통합 테스트 (agent-browser 필요) ──


@pytest.mark.integration
class TestWebSearchIntegration:
    """agent-browser가 설치된 ��경에서만 실행되는 통합 테스트."""

    @pytest.fixture(autouse=True)
    def check_agent_browser(self):
        if not WebSearchSource.is_available():
            pytest.skip("agent-browser not installed")

    def test_real_search(self):
        source = WebSearchSource(max_results=2, page_timeout=15)
        chunks = source.search("Python programming language", top_k=2)
        assert len(chunks) >= 1
        assert any("python" in c.text.lower() for c in chunks)
