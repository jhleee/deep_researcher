"""agent-browser 기반 웹 검색 소스.

agent-browser CLI(헤드리스 Rust 브라우저)를 subprocess로 호출하여
웹 검색과 페이지 본문 추출을 수행한다.

사전 조건:
    npm install -g agent-browser && agent-browser install

사용 예시:
    from harness.web_search import WebSearchSource
    source = WebSearchSource()
    chunks = source.search("LangGraph 아키텍처 패턴", top_k=3)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass

from harness.knowledge_base import Chunk, RetrievalSource

logger = logging.getLogger(__name__)

_SESSION_ACTIVE = False


# ── agent-browser CLI 래퍼 ──


class AgentBrowser:
    """agent-browser CLI 래퍼.

    각 명령을 subprocess로 실행하고 JSON 결과를 반환한다.
    세션(Chrome 프로세스)의 open/close 수명주기를 관리한다.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._binary = shutil.which("agent-browser")
        if not self._binary:
            raise RuntimeError(
                "agent-browser가 설치되어 있지 않습니다.\n"
                "  npm install -g agent-browser && agent-browser install"
            )

    def run(self, *args: str, timeout: int | None = None) -> str:
        """agent-browser 명령을 실행하고 stdout을 반환한다."""
        cmd = [self._binary, *args]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                logger.warning("agent-browser error: %s", stderr)
                return ""
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.warning("agent-browser timeout: %s", " ".join(args[:3]))
            return ""
        except FileNotFoundError:
            raise RuntimeError("agent-browser binary not found")

    def run_json(self, *args: str, timeout: int | None = None) -> dict | list | str:
        """--json 플래그로 실행하고 파싱된 JSON을 반환한다."""
        output = self.run(*args, "--json", timeout=timeout)
        if not output:
            return {}
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output

    def batch(self, commands: list[list[str]], timeout: int | None = None) -> list:
        """여러 명령을 batch 모드로 한 번에 실행한다."""
        input_json = json.dumps(commands)
        cmd = [self._binary, "batch", "--json"]
        try:
            result = subprocess.run(
                cmd,
                input=input_json,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout * 2,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        # fallback: 개별 실행
        return [self.run(*c) for c in commands]

    def open(self, url: str) -> str:
        return self.run("open", url)

    def goto(self, url: str) -> str:
        return self.run("goto", url)

    def snapshot(self, interactive: bool = False, compact: bool = False) -> str:
        args = ["snapshot"]
        if interactive:
            args.append("-i")
        if compact:
            args.append("-c")
        return self.run(*args)

    def click(self, ref: str) -> str:
        return self.run("click", ref)

    def fill(self, ref: str, text: str) -> str:
        return self.run("fill", ref, text)

    def press(self, key: str) -> str:
        return self.run("press", key)

    def get_text(self, selector: str = "body") -> str:
        return self.run("get", "text", selector)

    def get_title(self) -> str:
        return self.run("get", "title")

    def get_url(self) -> str:
        return self.run("get", "url")

    def wait(self, **kwargs: str) -> str:
        args = ["wait"]
        for k, v in kwargs.items():
            args.extend([f"--{k}", v])
        return self.run(*args)

    def close(self) -> str:
        return self.run("close")


# ── 검색 결과 파싱 ──


@dataclass
class SearchResult:
    """검색 결과 하나."""

    title: str
    url: str
    snippet: str = ""


def _parse_search_results_from_snapshot(snapshot: str) -> list[SearchResult]:
    """Google 검색 결과 스냅샷에서 결과를 추출한다.

    접근성 트리에서 링크 + 텍스트 패턴을 분석하여 검색 결과를 식별한다.
    """
    results: list[SearchResult] = []

    # 패턴: URL이 있는 링크 라인 추출
    # 접근성 트리에서 link "Title" url="https://..." 형태
    link_pattern = re.compile(
        r'link\s+"([^"]+)"\s+.*?url="(https?://[^"]+)"',
        re.IGNORECASE,
    )

    for match in link_pattern.finditer(snapshot):
        title = match.group(1).strip()
        url = match.group(2).strip()

        # Google 내부 링크 제외
        if "google.com" in url or "accounts.google" in url:
            continue
        if not title or len(title) < 5:
            continue

        results.append(SearchResult(title=title, url=url))

    # 대안: href 패턴
    if not results:
        href_pattern = re.compile(r'href="(https?://(?!.*google\.com)[^"]+)"[^>]*>([^<]+)')
        for match in href_pattern.finditer(snapshot):
            url = match.group(1).strip()
            title = match.group(2).strip()
            if title and len(title) >= 5:
                results.append(SearchResult(title=title, url=url))

    # 대안: 줄 단위 URL+제목 추출
    if not results:
        lines = snapshot.split("\n")
        for i, line in enumerate(lines):
            url_match = re.search(r'(https?://\S+)', line)
            if url_match:
                url = url_match.group(1).rstrip(')"\'')
                if "google.com" in url:
                    continue
                # 이전 줄을 제목으로 사용
                title = lines[i - 1].strip() if i > 0 else url
                title = re.sub(r'[@\[\]{}]', '', title).strip()
                if title and len(title) >= 3:
                    results.append(SearchResult(title=title, url=url))

    return results


def _clean_page_text(raw: str, max_chars: int = 5000) -> str:
    """페이지 본문에서 노이즈를 제거하고 핵심 텍스트만 추출한다."""
    # 네비게이션, 광고 등 짧은 줄 제거
    lines = raw.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 너무 짧은 줄 (메뉴, 버튼) 필터
        if len(line) < 15 and not line.endswith((".", "?", "!", "다", "요")):
            continue
        # 반복 공백 정리
        line = re.sub(r"\s+", " ", line)
        cleaned.append(line)

    text = "\n".join(cleaned)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


# ── WebSearchSource ──


class WebSearchSource(RetrievalSource):
    """agent-browser를 사용한 웹 검색 소스.

    Google 검색을 수행하고, 상위 결과 페이지의 본문을 추출하여
    Chunk 리스트로 반환한다.

    Args:
        max_results: 방문할 최대 검색 결과 수.
        page_timeout: 각 페이지 로딩 타임아웃 (초).
        chunk_size: 페이지 본문 청크 크기.
    """

    def __init__(
        self,
        max_results: int = 5,
        page_timeout: int = 20,
        chunk_size: int = 1500,
    ):
        self.max_results = max_results
        self.page_timeout = page_timeout
        self.chunk_size = chunk_size
        self._chunks: list[Chunk] = []
        self._browser: AgentBrowser | None = None
        self._query: str = ""

    @staticmethod
    def is_available() -> bool:
        """agent-browser가 설치되어 있는지 확인한다."""
        return shutil.which("agent-browser") is not None

    def load(self) -> list[Chunk]:
        """search()로 수집된 청크를 반환한다. 직접 호출용이 아님."""
        return list(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """웹 검색을 수행하고 결과 페이지 본문을 청크로 반환한다.

        1. Google 검색 수행
        2. 상위 N개 결과 URL 추출
        3. 각 페이지 방문하여 본문 추출
        4. 청킹 후 반환
        """
        self._query = query
        max_results = min(top_k, self.max_results)

        try:
            browser = AgentBrowser(timeout=self.page_timeout)
        except RuntimeError as e:
            logger.error("WebSearchSource: %s", e)
            return []

        chunks: list[Chunk] = []

        try:
            # 1. Google 검색
            logger.info("Searching: %s", query)
            browser.open(f"https://www.google.com/search?q={_url_encode(query)}")
            time.sleep(2)  # 검색 결과 로딩 대기

            # 2. 결과 추출
            snapshot = browser.snapshot(interactive=True)
            results = _parse_search_results_from_snapshot(snapshot)
            logger.info("Found %d search results", len(results))

            if not results:
                # 스냅샷에서 직접 텍스트 추출 시도
                text = browser.get_text()
                if text:
                    chunks.append(Chunk(
                        text=_clean_page_text(text),
                        source="google:search_results",
                        index=0,
                        metadata={"query": query, "type": "search_page"},
                    ))
                browser.close()
                self._chunks = chunks
                return chunks

            # 3. 각 결과 페이지 방문 + 본문 추출
            for i, result in enumerate(results[:max_results]):
                try:
                    logger.info("Visiting [%d/%d]: %s", i + 1, max_results, result.url)
                    browser.goto(result.url)
                    time.sleep(1)

                    page_text = browser.get_text()
                    if not page_text:
                        continue

                    cleaned = _clean_page_text(page_text)
                    if len(cleaned) < 50:
                        continue

                    # 청킹
                    page_chunks = self._split_to_chunks(
                        cleaned, source=result.url, title=result.title
                    )
                    chunks.extend(page_chunks)

                except Exception as e:
                    logger.warning("Failed to extract %s: %s", result.url, e)
                    continue

            browser.close()

        except Exception as e:
            logger.error("WebSearchSource error: %s", e)
            try:
                browser.close()
            except Exception:
                pass

        self._chunks = chunks
        return chunks

    def _split_to_chunks(
        self, text: str, source: str, title: str = ""
    ) -> list[Chunk]:
        """텍스트를 고정 크기 청크로 분할한다."""
        chunks: list[Chunk] = []
        for i in range(0, len(text), self.chunk_size):
            chunk_text = text[i:i + self.chunk_size]
            if len(chunk_text) < 30:
                continue
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                index=len(chunks),
                metadata={"title": title, "query": self._query},
            ))
        return chunks


def _url_encode(query: str) -> str:
    """검색 쿼리를 URL 인코딩한다."""
    import urllib.parse
    return urllib.parse.quote_plus(query)
