"""LangGraph Agent Harness TUI.

Textual 기반 터미널 사용자 인터페이스.
대시보드, 리서치 실행, 노드 트레이스, 체크포인트/메트릭/실패 조회, 설정 편집을 제공한다.

모든 패널은 앱 시작 시 한 번만 마운트되며, 탭 전환 시 display 토글로 전환한다.
파이프라인 실행 상태와 로그는 앱 레벨에 보관되어 탭 이동 후에도 유실되지 않는다.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
)

from harness.base import HarnessConfig, create_initial_state
from harness.checkpointer import JsonFileCheckpointer
from harness.config_store import AppConfig, SourceConfig, load_config, save_config
from harness.guardrails import InputGuardrail, OutputGuardrail
from harness.online_evaluator import OnlineEvaluator
from harness.sanitizer import LocalChunkDB
from harness.self_improvement import LocalFailureStore, SelfImprovementLoop

# ── 노드 트레이스 데이터 ──


@dataclass
class LLMCall:
    """단일 LLM 호출 기록 (노드 내부의 개별 invoke)."""

    call_id: int
    node_name: str
    prompt_preview: str
    response_preview: str
    elapsed: float
    token_hint: str = ""  # e.g. "320자 → 150자"
    role: str = ""  # 아키텍처 역할 (e.g. "비판적 검토자", "압축")


@dataclass
class NodeTrace:
    """파이프라인 노드 하나의 실행 기록."""

    node_name: str
    label: str
    desc: str
    elapsed: float
    summary: str
    input_preview: str = ""
    output_preview: str = ""
    llm_calls: list[LLMCall] = field(default_factory=list)


@dataclass
class RunSession:
    """하나의 파이프라인 실행 세션 전체 기록."""

    run_id: str
    query: str
    status: str = "running"  # running | done | error
    start_time: float = 0.0
    elapsed: float = 0.0
    step_count: int = 0
    traces: list[NodeTrace] = field(default_factory=list)
    all_llm_calls: list[LLMCall] = field(default_factory=list)
    error_msg: str = ""


# ── 트레이싱 모델 래퍼 ──


class TracingModelWrapper:
    """ThinkingModelWrapper를 감싸서 모든 LLM invoke를 서브 트레이스로 기록한다.

    파이프라인 노드 내부의 개별 LLM 호출(페르소나별 탐색, 압축, 검증 등)이
    각각 하나의 LLMCall로 기록되어 Trace 패널에서 확인할 수 있다.
    """

    def __init__(
        self,
        inner,  # ThinkingModelWrapper
        session: RunSession,
        on_call: callable | None = None,
    ):
        self._inner = inner
        self._session = session
        self._on_call = on_call  # 호출될 때마다 TUI 갱신 콜백
        self._call_counter = 0
        self._current_node: str = ""

    def set_current_node(self, node_name: str) -> None:
        self._current_node = node_name

    def invoke(self, input, **kwargs):
        """LLM 호출을 가로채서 기록한 뒤 원본 결과를 반환한다."""
        self._call_counter += 1
        call_id = self._call_counter

        # 프롬프트 미리보기
        if isinstance(input, str):
            prompt_preview = input[:400]
        elif isinstance(input, list):
            parts = []
            for m in input[-3:]:  # 최근 3개 메시지만
                c = m.content if hasattr(m, "content") else str(m)
                parts.append(c[:200])
            prompt_preview = "\n---\n".join(parts)
        else:
            prompt_preview = str(input)[:400]

        start = time.time()
        result = self._inner.invoke(input, **kwargs)
        elapsed = time.time() - start

        response_preview = result.content[:400] if hasattr(result, "content") else str(result)[:400]

        prompt_len = len(input) if isinstance(input, str) else sum(
            len(m.content) if hasattr(m, "content") else len(str(m)) for m in input
        ) if isinstance(input, list) else 0
        resp_len = len(result.content) if hasattr(result, "content") else 0

        role, role_icon = _classify_llm_role(prompt_preview)

        llm_call = LLMCall(
            call_id=call_id,
            node_name=self._current_node,
            prompt_preview=prompt_preview,
            response_preview=response_preview,
            elapsed=elapsed,
            token_hint=f"{prompt_len}자 → {resp_len}자",
            role=role,
        )

        self._session.all_llm_calls.append(llm_call)

        if self._on_call:
            self._on_call(llm_call)

        return result

    def with_structured_output(self, schema, **kwargs):
        """structured output 체인도 트레이싱한다."""
        inner_chain = self._inner.with_structured_output(schema, **kwargs)
        wrapper = self

        class TracedChain:
            def invoke(self_, input_text, **kw):
                wrapper._call_counter += 1
                call_id = wrapper._call_counter

                if isinstance(input_text, str):
                    prompt_preview = input_text[:400]
                else:
                    prompt_preview = str(input_text)[:400]

                start = time.time()
                result = inner_chain.invoke(input_text, **kw)
                elapsed = time.time() - start

                try:
                    resp_text = result.model_dump_json(indent=2)[:400]
                except Exception:
                    resp_text = str(result)[:400]

                prompt_len = len(input_text) if isinstance(input_text, str) else 0

                role, _ = _classify_llm_role(prompt_preview)
                llm_call = LLMCall(
                    call_id=call_id,
                    node_name=wrapper._current_node,
                    prompt_preview=prompt_preview,
                    response_preview=resp_text,
                    elapsed=elapsed,
                    token_hint=f"{prompt_len}자 → structured",
                    role=role,
                )
                wrapper._session.all_llm_calls.append(llm_call)
                if wrapper._on_call:
                    wrapper._on_call(llm_call)

                return result

        return TracedChain()


# ── CSS ──

APP_CSS = """
Screen {
    background: $surface;
}

#sidebar {
    width: 28;
    background: $panel;
    border-right: tall $primary;
    padding: 1 0;
}

#sidebar .nav-label {
    padding: 0 2;
    margin-bottom: 1;
    color: $text-muted;
    text-style: bold;
}

.nav-btn {
    width: 100%;
    margin: 0 1;
    min-width: 24;
}

.nav-btn.-active {
    background: $primary;
    color: $text;
    text-style: bold;
}

#main-content {
    padding: 1 2;
}

.panel {
    display: none;
}

.panel.-visible {
    display: block;
}

.section-title {
    text-style: bold;
    color: $primary;
    margin-bottom: 1;
}

.stat-card {
    height: 5;
    border: round $primary;
    padding: 0 2;
    margin: 0 1;
    min-width: 20;
}

.stat-card .stat-value {
    text-style: bold;
    color: $success;
    text-align: center;
}

.stat-card .stat-label {
    color: $text-muted;
    text-align: center;
}

.config-row {
    height: 3;
    margin-bottom: 1;
    align: left middle;
}

.config-label {
    width: 24;
    padding: 0 1;
    content-align: left middle;
}

.config-input {
    width: 40;
}

#research-input {
    height: 5;
    margin-bottom: 1;
}

#research-log {
    height: 1fr;
    border: round $primary;
    margin-top: 1;
}

DataTable {
    height: 1fr;
}

#pipeline-diagram {
    margin: 1 0;
    padding: 1 2;
    border: round $primary;
    height: auto;
    max-height: 18;
}

/* ── 세션 박스 (사이드바) ── */

#session-box {
    margin: 1 1 0 1;
    padding: 1 1;
    border: round $secondary;
    height: auto;
    max-height: 14;
    background: $surface;
}

#session-box.--running {
    border: round $warning;
}

#session-box.--done {
    border: round $success;
}

#session-box.--error {
    border: round $error;
}

/* ── 트레이스 패널 ── */

#trace-list {
    height: 1fr;
    border: round $primary;
}

#trace-detail {
    height: 1fr;
    border: round $secondary;
    margin-top: 1;
}
"""

# ── 아키텍처 단계 정의 ──

# 각 노드가 속한 아키텍처 단계 (stage)
ARCH_STAGES: dict[str, tuple[str, str, str]] = {
    # node_name → (stage_color, stage_name, stage_icon)
    "term_resolver":     ("blue",    "Term Resolution", "🔎"),
    "planner":           ("cyan",    "Plan-and-Execute", "📋"),
    "storm_worker":      ("yellow",  "STORM Map-Reduce", "🌪"),
    "synthesis":         ("green",   "STORM Map-Reduce", "🌪"),
    "cove_plan":         ("magenta", "CoVe Verification", "🔍"),
    "factored_verifier": ("magenta", "CoVe Verification", "🔍"),
    "cross_check":       ("magenta", "CoVe Verification", "🔍"),
    "sanitizer":         ("red",     "Deterministic Sanitizer", "🛡"),
    "self_correct":      ("red",     "Deterministic Sanitizer", "🛡"),
    "finalize":          ("green",   "Output", "✅"),
}

# 노드 이름 → (색상, 라벨, 설명)
NODE_LABELS: dict[str, tuple[str, str, str]] = {
    "term_resolver":     ("blue",    "Term Resolver", "용어 사전 확인"),
    "planner":           ("cyan",    "Planner", "실행 계획 생성"),
    "storm_worker":      ("yellow",  "STORM Worker", "다중 페르소나 탐색"),
    "synthesis":         ("green",   "Synthesis", "초안 합성"),
    "cove_plan":         ("magenta", "CoVe Plan", "검증 질문 생성"),
    "factored_verifier": ("yellow",  "Factored Verifier", "독립 검증"),
    "cross_check":       ("magenta", "Cross-Check", "교차 대조"),
    "sanitizer":         ("red",     "Sanitizer", "결정론적 검증 (LLM 미사용)"),
    "self_correct":      ("red",     "Self-Correct", "자기 교정"),
    "finalize":          ("green",   "Finalize", "최종 출력"),
}

# LLM 호출의 프롬프트 패턴으로 아키텍처 역할을 분류
_ROLE_PATTERNS: list[tuple[str, str, str]] = [
    # (패턴, role 이름, 아이콘)
    ("비판적 검토자",        "비판적 검토자",     "👁"),
    ("도메인 전문가",        "도메인 전문가",     "🎓"),
    ("반론 제기자",          "반론 제기자",       "⚔"),
    ("핵심 사실만",          "압축",             "📦"),
    ("200토큰 이내",         "압축",             "📦"),
    ("종합하여 보고서",      "초안 합성",         "📝"),
    ("종합하여 구조화",      "초안 합성",         "📝"),
    ("검증할 질문",          "검증 질문 생성",    "❓"),
    ("사실에 기반하여 답변",  "독립 검증",        "✓"),
    ("불일치 사항",          "교차 대조",         "⚖"),
    ("오류를 수정하여",      "자기 교정",         "🔧"),
    ("하위 작업으로 분할",    "계획 수립",        "📋"),
    ("독립적인 하위 작업",    "계획 수립",        "📋"),
    ("고유명사, 제품명, 게임명", "용어 확인",     "🔎"),
]


def _classify_llm_role(prompt: str) -> tuple[str, str]:
    """프롬프트 패턴으로 LLM 호출의 아키텍처 역할을 분류한다.

    Returns:
        (role, icon) 튜플. 매칭 없으면 ("", "").
    """
    for pattern, role, icon in _ROLE_PATTERNS:
        if pattern in prompt:
            return role, icon
    return "", ""


def _summarize_node_output(node_name: str, output: dict) -> str:
    """노드 출력에서 핵심 정보를 한 줄로 요약한다."""
    if node_name == "term_resolver":
        tc = output.get("term_context", "")
        if tc:
            return f"용어 확인 {len(tc)}자"
        return ""
    elif node_name == "planner":
        plan = output.get("execution_plan")
        if plan and hasattr(plan, "sub_tasks"):
            return f"하위 작업 {len(plan.sub_tasks)}개 생성"
        if output.get("plan"):
            return f"하위 작업 {len(output['plan'])}개 생성"
    elif node_name == "storm_worker":
        results = output.get("worker_results", [])
        if results:
            tid = results[0].get("task_id", "?")
            return f"task={tid}, {len(results[0].get('data', ''))}자 수집"
    elif node_name == "synthesis":
        draft = output.get("draft", "")
        if draft:
            return f"초안 {len(draft)}자"
    elif node_name == "cove_plan":
        qs = output.get("verification_questions", [])
        return f"검증 질문 {len(qs)}개"
    elif node_name == "factored_verifier":
        answers = output.get("verification_answers", [])
        if answers:
            return f"검증 답변 {len(answers)}개"
    elif node_name == "cross_check":
        draft = output.get("draft", "")
        if draft:
            return f"교차 검증 완료, {len(draft)}자"
    elif node_name == "sanitizer":
        errors = output.get("sanitizer_errors", [])
        count = output.get("correction_count", 0)
        if errors:
            return f"오류 {len(errors)}개 발견 (교정 {count}/3)"
        return "오류 없음 — 통과"
    elif node_name == "self_correct":
        draft = output.get("draft", "")
        if draft:
            return f"교정 완료, {len(draft)}자"
    elif node_name == "finalize":
        msgs = output.get("messages", [])
        if msgs:
            c = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
            return f"최종 출력 {len(c)}자"
    return ""


def _preview_dict(d: dict, max_len: int = 500) -> str:
    """dict를 사람이 읽을 수 있는 축약 문자열로 변환한다."""
    try:
        text = json.dumps(d, ensure_ascii=False, default=str, indent=2)
    except Exception:
        text = str(d)
    if len(text) > max_len:
        text = text[:max_len] + "\n... (truncated)"
    return text


# ══════════════════════════════════════════════════════════════
#  세션 박스 (사이드바)
# ══════════════════════════════════════════════════════════════


class SessionBox(Vertical):
    """사이드바에 항상 표시되는 활성 세션 상태 위젯."""

    def compose(self) -> ComposeResult:
        yield Static("[bold]Session[/]", id="session-title")
        yield Static("[dim]idle[/]", id="session-run-id")
        yield Static("", id="session-step")
        yield Static("", id="session-elapsed")
        yield Static("", id="session-detail")
        yield Button("View Log", id="session-go-btn", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "session-go-btn":
            app: HarnessApp = self.app  # type: ignore[assignment]
            app.action_nav("research")

    def sync_from_session(self, session: RunSession | None) -> None:
        """RunSession 상태를 위젯에 반영한다."""
        self.remove_class("--running", "--done", "--error")

        if session is None:
            self.query_one("#session-run-id", Static).update("[dim]idle[/]")
            self.query_one("#session-step", Static).update("")
            self.query_one("#session-elapsed", Static).update("")
            self.query_one("#session-detail", Static).update("")
            return

        self.query_one("#session-run-id", Static).update(f"[bold]{session.run_id}[/]")

        if session.status == "running":
            self.add_class("--running")
            if session.traces:
                last = session.traces[-1]
                self.query_one("#session-step", Static).update(
                    f"[yellow]{last.label}[/]"
                )
                self.query_one("#session-detail", Static).update(
                    f"[dim]{last.summary[:40]}[/]" if last.summary else ""
                )
            else:
                self.query_one("#session-step", Static).update(
                    "[yellow]starting...[/]"
                )
                self.query_one("#session-detail", Static).update("")
            elapsed = time.time() - session.start_time
            self.query_one("#session-elapsed", Static).update(f"[dim]{elapsed:.1f}s[/]")

        elif session.status == "done":
            self.add_class("--done")
            self.query_one("#session-step", Static).update(
                f"[green]done — {session.step_count} steps[/]"
            )
            self.query_one("#session-elapsed", Static).update(
                f"[dim]{session.elapsed:.1f}s[/]"
            )
            self.query_one("#session-detail", Static).update("")

        elif session.status == "error":
            self.add_class("--error")
            self.query_one("#session-step", Static).update("[red]error[/]")
            self.query_one("#session-elapsed", Static).update(
                f"[dim]{session.elapsed:.1f}s[/]"
            )
            self.query_one("#session-detail", Static).update(
                f"[red]{session.error_msg[:50]}[/]"
            )


# ══════════════════════════════════════════════════════════════
#  대시보드 패널
# ══════════════════════════════════════════════════════════════


class DashboardPanel(VerticalScroll):
    """프로젝트 상태 개요 대시보드."""

    def compose(self) -> ComposeResult:
        yield Static("Dashboard", classes="section-title")

        with Horizontal():
            with Vertical(classes="stat-card"):
                yield Static("--", id="stat-checkpoints", classes="stat-value")
                yield Static("Checkpoints", classes="stat-label")
            with Vertical(classes="stat-card"):
                yield Static("--", id="stat-metrics", classes="stat-value")
                yield Static("Metrics", classes="stat-label")
            with Vertical(classes="stat-card"):
                yield Static("--", id="stat-failures", classes="stat-value")
                yield Static("Failures", classes="stat-label")
            with Vertical(classes="stat-card"):
                yield Static("--", id="stat-regression", classes="stat-value")
                yield Static("Regression", classes="stat-label")

        yield Static("")
        yield Static("Pipeline Architecture", classes="section-title")
        yield Static(
            "  [bold cyan]Planner[/] → [bold yellow]STORM Workers[/] (×N) → "
            "[bold green]Synthesis[/]\n"
            "    ↓\n"
            "  [bold magenta]CoVe Plan[/] → [bold yellow]Factored Verifiers[/] (×N) → "
            "[bold magenta]Cross-Check[/]\n"
            "    ↓\n"
            "  [bold red]Sanitizer[/] → [dim](errors?)[/] → "
            "[bold red]Self-Correct[/] ↻ (max 3)\n"
            "    ↓\n"
            "  [bold green]Finalize[/] → [bold]Output[/]",
            id="pipeline-diagram",
        )
        yield Static("")
        yield Static("Modules", classes="section-title")
        yield Static(
            "  [bold]harness.base[/]           HarnessState, HarnessConfig\n"
            "  [bold]harness.guardrails[/]      InputGuardrail, OutputGuardrail\n"
            "  [bold]harness.sanitizer[/]       LocalChunkDB, DeterministicSanitizer\n"
            "  [bold]harness.checkpointer[/]    JsonFileCheckpointer\n"
            "  [bold]harness.online_evaluator[/] OnlineEvaluator\n"
            "  [bold]harness.self_improvement[/] SelfImprovementLoop\n"
            "  [bold]harness.lmstudio[/]        ThinkingModelWrapper\n"
            "  [bold]architectures[/]           ZeroHallucinationPipeline"
        )

    def refresh_stats(self) -> None:
        app: HarnessApp = self.app  # type: ignore[assignment]
        config = app.harness_config

        cp_dir = Path(config.checkpoint_dir)
        cp_count = len(list(cp_dir.glob("*.json"))) if cp_dir.exists() else 0
        self.query_one("#stat-checkpoints", Static).update(str(cp_count))

        m_dir = Path("metrics")
        m_count = len(list(m_dir.glob("*.json"))) if m_dir.exists() else 0
        self.query_one("#stat-metrics", Static).update(str(m_count))

        f_dir = Path("failures")
        f_count = len(list(f_dir.glob("*.json"))) if f_dir.exists() else 0
        self.query_one("#stat-failures", Static).update(str(f_count))

        r_path = Path("tests/datasets/regression.json")
        r_count = 0
        if r_path.exists():
            try:
                data = json.loads(r_path.read_text(encoding="utf-8"))
                r_count = len(data)
            except Exception:
                pass
        self.query_one("#stat-regression", Static).update(str(r_count))


# ══════════════════════════════════════════════════════════════
#  리서치 실행 패널
# ══════════════════════════════════════════════════════════════


class ResearchPanel(Vertical):
    """리서치 파이프라인 실행 인터페이스."""

    def compose(self) -> ComposeResult:
        yield Static("Research Pipeline", classes="section-title")
        yield Static(
            "[dim]Enter a research query to run the Zero-Hallucination Pipeline.[/]"
        )
        yield TextArea(id="research-input")

        yield Static("Sources", classes="section-title")
        with Horizontal(classes="config-row"):
            yield Switch(value=False, id="src-local-enabled")
            yield Static(" Local Files ", classes="config-label")
            yield Input(
                placeholder="디렉토리 경로 (예: ./data/sources)",
                id="src-local-dir",
                classes="config-input",
            )
        with Horizontal(classes="config-row"):
            yield Switch(value=False, id="src-web-enabled")
            yield Static(" Web Search ", classes="config-label")
            yield Static("", id="src-web-status")

        with Horizontal():
            yield Button("Run Pipeline", variant="primary", id="btn-run")
            yield Button("Clear Log", variant="default", id="btn-clear-log")
            yield Static("  ")
            yield Static("", id="src-summary")
        yield RichLog(id="research-log", highlight=True, markup=True)

    def on_mount(self) -> None:
        self._load_source_config()

    def _load_source_config(self) -> None:
        """AppConfig에서 소스 설정을 위젯에 반영한다."""
        app: HarnessApp = self.app  # type: ignore[assignment]
        src = app.app_config.sources
        self.query_one("#src-local-enabled", Switch).value = src.local_enabled
        self.query_one("#src-local-dir", Input).value = src.local_directory
        self.query_one("#src-web-enabled", Switch).value = src.web_search_enabled
        self._update_web_status()

    def _update_web_status(self) -> None:
        from harness.web_search import WebSearchSource

        if WebSearchSource.is_available():
            self.query_one("#src-web-status", Static).update(
                "[green]agent-browser installed[/]"
            )
        else:
            self.query_one("#src-web-status", Static).update(
                "[red]agent-browser not found[/]"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run":
            self._run_pipeline()
        elif event.button.id == "btn-clear-log":
            self.query_one("#research-log", RichLog).clear()

    def _log(self, msg: str) -> None:
        self.app.call_from_thread(self.query_one("#research-log", RichLog).write, msg)

    def _set_running(self, running: bool) -> None:
        btn = self.query_one("#btn-run", Button)
        btn.disabled = running
        btn.label = "Running..." if running else "Run Pipeline"

    def _sync_session(self) -> None:
        """앱의 현재 세션 상태를 사이드바에 반영한다."""
        app: HarnessApp = self.app  # type: ignore[assignment]
        self.app.call_from_thread(
            app.query_one("#session-box", SessionBox).sync_from_session,
            app.current_session,
        )

    @work(thread=True)
    def _run_pipeline(self) -> None:
        self.app.call_from_thread(self._set_running, True)
        query = self.query_one("#research-input", TextArea).text.strip()

        if not query:
            self._log("[red]Error: 쿼리를 입력하세요.[/]")
            self.app.call_from_thread(self._set_running, False)
            return

        app: HarnessApp = self.app  # type: ignore[assignment]
        config = app.harness_config
        run_id = f"run_{uuid.uuid4().hex[:8]}"

        # 세션 생성 + 디렉토리 저장소
        from harness.session_store import SessionStore

        session = RunSession(run_id=run_id, query=query, start_time=time.time())
        app.current_session = session
        store = SessionStore(run_id)
        store.save_config(app.app_config)
        self._sync_session()

        self._log(f"[cyan]━━━ Run {run_id} ━━━[/]")
        self._log(f"[dim]Query: {query[:120]}[/]")
        self._log(f"[dim]Session dir: sessions/{run_id}/[/]")

        # ── 1. Input guardrail ──
        self._log("[yellow]▸ Input Guardrail 검사 중...[/]")
        input_guard = InputGuardrail(strict=config.guardrail_strict)
        guard_result = input_guard.check(query)

        store.save_input_guardrail(
            passed=guard_result.passed,
            reason=guard_result.reason,
            sanitized=guard_result.sanitized_content,
            original=query,
        )

        if not guard_result.passed:
            self._log(f"[red]✗ Input Guardrail BLOCKED: {guard_result.reason}[/]")
            session.status = "error"
            session.error_msg = f"Guardrail: {guard_result.reason}"
            session.elapsed = time.time() - session.start_time
            store.save_session_meta(
                query, "error", session.start_time,
                session.elapsed, 0, session.error_msg,
            )
            self._sync_session()
            self.app.call_from_thread(self._set_running, False)
            return

        sanitized_query = guard_result.sanitized_content
        self._log("[green]✓ Input Guardrail 통과[/]")
        if sanitized_query != query:
            self._log("[yellow]  (PII가 마스킹되었습니다)[/]")

        # ── 2. LM Studio 연결 ──
        self._log("[yellow]▸ LM Studio 연결 중...[/]")
        try:
            from harness.lmstudio import ThinkingModelWrapper, create_lmstudio_llm

            llm = create_lmstudio_llm(
                base_url=app.app_config.lm_studio_url,
                model=app.app_config.lm_studio_model,
            )
            base_model = ThinkingModelWrapper(llm)

            def _on_llm_call(call: LLMCall) -> None:
                role_tag = f" [bold]{call.role}[/]" if call.role else ""
                self._log(
                    f"    [dim]  LLM #{call.call_id}{role_tag}[dim] "
                    f"{call.token_hint} ({call.elapsed:.1f}s)[/]"
                )
                self.app.call_from_thread(app.refresh_trace_panel)

            model = TracingModelWrapper(base_model, session, on_call=_on_llm_call)
            self._log("[green]✓ LM Studio 연결됨[/]")
        except Exception as e:
            self._log(
                f"[red]✗ LM Studio 연결 실패: {e}[/]\n"
                "[dim]LM Studio가 실행 중인지 확인하세요.[/]"
            )
            session.status = "error"
            session.error_msg = f"LM Studio: {e}"
            session.elapsed = time.time() - session.start_time
            self._sync_session()
            self.app.call_from_thread(self._set_running, False)
            return

        # ── 3. 소스 로딩 ──
        local_enabled = self.query_one("#src-local-enabled", Switch).value
        local_dir = self.query_one("#src-local-dir", Input).value.strip()
        web_enabled = self.query_one("#src-web-enabled", Switch).value
        knowledge_base = None
        sources_summary: list[str] = []

        has_any_source = (local_enabled and local_dir) or web_enabled
        if has_any_source:
            from harness.knowledge_base import LocalKnowledgeBase

            knowledge_base = LocalKnowledgeBase()

            # 로컬 파일 소스
            if local_enabled and local_dir:
                from harness.knowledge_base import LocalFileSource

                self._log(f"[yellow]▸ 로컬 소스 로딩: {local_dir}[/]")
                src_cfg = app.app_config.sources
                knowledge_base.add_source(LocalFileSource(
                    local_dir,
                    chunk_size=src_cfg.local_chunk_size,
                    chunk_overlap=src_cfg.local_chunk_overlap,
                ))

            # 웹 검색 소스
            if web_enabled:
                from harness.web_search import WebSearchSource

                if WebSearchSource.is_available():
                    src_cfg = app.app_config.sources
                    knowledge_base.add_source(WebSearchSource(
                        max_results=src_cfg.web_search_max_results,
                        page_timeout=src_cfg.web_search_page_timeout,
                    ))
                    self._log("[yellow]▸ 웹 검색 소스 활성화[/]")
                    sources_summary.append("Web")
                else:
                    self._log(
                        "[red]✗ agent-browser 미설치 — 웹 검색 비활성[/]"
                    )

            try:
                stats = knowledge_base.load()
                chunk_db = knowledge_base.chunk_db
                if stats["chunks"] > 0:
                    self._log(
                        f"[green]✓ 소스 로딩 완료 — "
                        f"파일 {stats['files']}개, "
                        f"청크 {stats['chunks']}개, "
                        f"{stats['chars']:,}자[/]"
                    )
                    sources_summary.append(f"Local({stats['files']})")
                else:
                    self._log("[dim]로컬 소스에서 로딩된 자료 없음[/]")
            except Exception as e:
                self._log(f"[red]✗ 소스 로딩 실패: {e}[/]")
                session.status = "error"
                session.error_msg = f"Source: {e}"
                session.elapsed = time.time() - session.start_time
                self._sync_session()
                self.app.call_from_thread(self._set_running, False)
                return

            store.save_knowledge_base(stats, sources_summary)
            self.app.call_from_thread(
                self.query_one("#src-summary", Static).update,
                f"[dim]Sources: {', '.join(sources_summary)}[/]"
                if sources_summary else "",
            )
        else:
            chunk_db = LocalChunkDB()
            self._log(
                "[dim]소스 미활성 — LLM 내부 지식만 사용[/]"
            )

        # ── 4. 파이프라인 빌드 ──
        self._log("[yellow]▸ 파이프라인 빌드 중...[/]")
        try:
            from architectures.zero_hallucination import build_zero_hallucination_pipeline

            graph = build_zero_hallucination_pipeline(
                planner_model=model,
                worker_model=model,
                verifier_model=model,
                synthesizer_model=model,
                chunk_db=chunk_db,
                config=config,
                knowledge_base=knowledge_base,
            )
            compiled = graph.compile()
            self._log("[green]✓ 파이프라인 빌드 완료[/]")
        except Exception as e:
            self._log(f"[red]✗ 파이프라인 빌드 실패: {e}[/]")
            session.status = "error"
            session.error_msg = f"Build: {e}"
            session.elapsed = time.time() - session.start_time
            self._sync_session()
            self.app.call_from_thread(self._set_running, False)
            return

        # ── 4. 스트리밍 실행 ──
        self._log("[yellow]▸ 파이프라인 실행 시작[/]")
        from langchain_core.messages import HumanMessage

        initial = create_initial_state()
        initial["messages"] = [HumanMessage(content=sanitized_query)]

        try:
            result: dict[str, Any] = {}
            step_count = 0
            prev_state: dict[str, Any] = dict(initial)
            llm_cursor = 0
            last_stage = ""  # 아키텍처 단계 변경 감지

            for event in compiled.stream(initial, stream_mode="updates"):
                step_count += 1
                elapsed_so_far = time.time() - session.start_time

                for node_name, node_output in event.items():
                    color, label, desc = NODE_LABELS.get(
                        node_name, ("dim", node_name, "")
                    )
                    summary = _summarize_node_output(node_name, node_output)

                    # 아키텍처 단계 헤더 출력
                    stage_info = ARCH_STAGES.get(node_name)
                    if stage_info:
                        _, stage_name, stage_icon = stage_info
                        if stage_name != last_stage:
                            last_stage = stage_name
                            self._log("")
                            self._log(
                                f"[bold {color}]"
                                f"{'━' * 3} {stage_icon} "
                                f"{stage_name} "
                                f"{'━' * 20}[/]"
                            )

                    # 이 노드 실행 중에 발생한 LLM 호출들을 수집
                    new_calls = session.all_llm_calls[llm_cursor:]
                    for c in new_calls:
                        c.node_name = node_name
                    llm_cursor = len(session.all_llm_calls)

                    # 트레이스 기록
                    trace = NodeTrace(
                        node_name=node_name,
                        label=label,
                        desc=desc,
                        elapsed=elapsed_so_far,
                        summary=summary,
                        input_preview=_preview_dict(prev_state, 800),
                        output_preview=_preview_dict(node_output, 800),
                        llm_calls=list(new_calls),
                    )
                    session.traces.append(trace)
                    session.step_count = step_count

                    # 노드 로그 출력
                    self._log(
                        f"  [{color}]▸ {label}[/] {desc} "
                        f"[dim]({elapsed_so_far:.1f}s)[/]"
                    )
                    if summary:
                        self._log(f"    [dim]{summary}[/]")

                    # LLM 호출 역할별 요약
                    if new_calls:
                        roles = {}
                        for c in new_calls:
                            r = c.role or "LLM"
                            roles[r] = roles.get(r, 0) + 1
                        role_str = ", ".join(
                            f"{r}×{n}" if n > 1 else r
                            for r, n in roles.items()
                        )
                        self._log(
                            f"    [dim]{len(new_calls)} calls: "
                            f"{role_str}[/]"
                        )

                    # 사이드바 갱신
                    self._sync_session()

                    # 트레이스 패널 실시간 갱신
                    self.app.call_from_thread(app.refresh_trace_panel)

                    # 다음 노드가 시작되기 전에 current_node 갱신
                    model.set_current_node("")

                    # 상태 누적
                    result.update(node_output)
                    prev_state = dict(result)

            session.elapsed = time.time() - session.start_time
            session.status = "done"
            self._sync_session()

            self._log(
                f"[green]✓ 파이프라인 완료 — "
                f"{step_count}단계, {session.elapsed:.1f}초[/]"
            )

            # Output guardrail
            final_text = ""
            messages = result.get("messages", [])
            if messages:
                final_content = (
                    messages[-1].content
                    if hasattr(messages[-1], "content")
                    else str(messages[-1])
                )
                out_guard = OutputGuardrail(strict=config.guardrail_strict)
                out_result = out_guard.check(final_content)

                store.save_output_guardrail(
                    out_result.passed, out_result.reason, final_content,
                )

                if out_result.passed:
                    self._log("[green]✓ Output Guardrail 통과[/]")
                    self._log("")
                    self._log("[bold cyan]━━━ Result ━━━[/]")
                    self._log(out_result.sanitized_content)
                    final_text = out_result.sanitized_content
                else:
                    self._log(
                        f"[red]✗ Output Guardrail BLOCKED: {out_result.reason}[/]"
                    )

            # Evaluate
            evaluator = OnlineEvaluator()
            scores = evaluator.evaluate_run(result, run_id)
            store.save_evaluation(scores)
            self._log("")
            self._log("[bold]Evaluation Scores:[/]")
            for k, v in scores.items():
                c = "green" if v >= 0.7 else "yellow" if v >= 0.4 else "red"
                self._log(f"  [{c}]{k}: {v:.2f}[/]")

            # 세션 디렉토리에 모든 데이터 저장
            completed_nodes = [t.node_name for t in session.traces]
            store.save_checkpoint(result)
            store.save_traces(session.traces, session.all_llm_calls)
            store.save_session_meta(
                query, "done", session.start_time,
                session.elapsed, step_count,
                last_node=completed_nodes[-1] if completed_nodes else "",
                completed_nodes=completed_nodes,
                phase="complete",
            )
            if final_text:
                store.save_result(final_text)

            self._log(f"[dim]Session saved: sessions/{run_id}/[/]")

        except Exception as e:
            import traceback as tb

            session.elapsed = time.time() - session.start_time
            session.status = "error"
            session.error_msg = str(e)
            self._sync_session()

            tb_str = tb.format_exc()
            self._log(f"[red]✗ 실행 실패 ({session.elapsed:.1f}초): {e}[/]")
            self._log(f"[dim]{tb_str}[/]")

            from harness.self_improvement import FailedRun

            fail_store = LocalFailureStore()
            fail_store.record_failure(
                FailedRun(
                    run_id=run_id,
                    input_data={"query": query},
                    output_data={},
                    error_log=[str(e)],
                )
            )

            completed_nodes = [t.node_name for t in session.traces]
            last_node = completed_nodes[-1] if completed_nodes else ""

            # 에러 발생 단계 추정
            if step_count == 0:
                phase = "pre_streaming"
            elif "messages" in result and result.get("messages"):
                phase = "post_streaming"
            else:
                phase = "streaming"

            store.save_failure(
                [str(e)],
                {"query": query},
                traceback_str=tb_str,
                phase=phase,
                last_node=last_node,
                partial_state=result if result else None,
            )
            store.save_traces(session.traces, session.all_llm_calls)
            store.save_session_meta(
                query, "error", session.start_time,
                session.elapsed, session.step_count, str(e),
                last_node=last_node,
                completed_nodes=completed_nodes,
                phase=phase,
            )

            # 부분 결과라도 checkpoint으로 저장
            if result:
                store.save_checkpoint(result)

            self._log(f"[dim]Session saved: sessions/{run_id}/[/]")
        finally:
            self.app.call_from_thread(self._set_running, False)


# ══════════════════════════════════════════════════════════════
#  노드 트레이스 패널
# ══════════════════════════════════════════════════════════════


class TracePanel(Vertical):
    """파이프라인 노드별 입출력 상세 추적 패널.

    상단 테이블: 노드 단위 트레이스 (클릭 시 하단에 상세 표시)
    하단 로그: 선택한 노드의 개별 LLM 호출(프롬프트/응답) 포함 상세
    """

    def compose(self) -> ComposeResult:
        yield Static("Node Trace", classes="section-title")
        yield Static(
            "[dim]노드를 선택하면 내부 LLM 호출(페르소나별 탐색, 압축 등)을 "
            "프롬프트/응답 단위로 확인할 수 있습니다.[/]"
        )
        with Horizontal():
            yield Button("Refresh", variant="primary", id="btn-trace-refresh")
        yield DataTable(id="trace-list")
        yield Static("Detail", classes="section-title")
        yield RichLog(id="trace-detail", highlight=True, markup=True)

    def on_mount(self) -> None:
        table = self.query_one("#trace-list", DataTable)
        table.add_columns(
            "#", "Stage", "Node", "Calls", "Roles", "Time", "Summary"
        )
        table.cursor_type = "row"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-trace-refresh":
            self.refresh_traces()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._show_detail(event.cursor_row)

    def refresh_traces(self) -> None:
        """현재 세션의 트레이스를 테이블에 반영한다."""
        app: HarnessApp = self.app  # type: ignore[assignment]
        session = app.current_session
        table = self.query_one("#trace-list", DataTable)
        table.clear()

        if session is None:
            return

        for i, t in enumerate(session.traces):
            # 아키텍처 단계
            stage_info = ARCH_STAGES.get(t.node_name)
            stage = f"{stage_info[2]} {stage_info[1]}" if stage_info else ""
            # LLM 호출 역할 요약
            roles = {}
            for c in t.llm_calls:
                r = c.role or "LLM"
                roles[r] = roles.get(r, 0) + 1
            role_str = ", ".join(
                f"{r}x{n}" if n > 1 else r
                for r, n in roles.items()
            ) if roles else "--"

            table.add_row(
                str(i + 1),
                stage[:20],
                t.label,
                str(len(t.llm_calls)),
                role_str[:30],
                f"{t.elapsed:.1f}s",
                t.summary[:40] if t.summary else "--",
            )

    def _show_detail(self, row_idx: int) -> None:
        app: HarnessApp = self.app  # type: ignore[assignment]
        session = app.current_session
        detail = self.query_one("#trace-detail", RichLog)
        detail.clear()

        if session is None or row_idx >= len(session.traces):
            detail.write("[dim]트레이스를 선택하세요.[/]")
            return

        t = session.traces[row_idx]

        # 아키텍처 단계 표시
        stage_info = ARCH_STAGES.get(t.node_name)
        if stage_info:
            sc, sn, si = stage_info
            detail.write(f"[bold {sc}]{si} {sn}[/]")

        detail.write(f"[bold cyan]{'=' * 60}[/]")
        detail.write(f"[bold cyan]{t.label} — {t.desc}[/]")
        detail.write(f"[bold cyan]{'=' * 60}[/]")
        detail.write(f"  [dim]Node:[/] {t.node_name}")
        detail.write(f"  [dim]Elapsed:[/] {t.elapsed:.1f}s")
        detail.write(f"  [dim]Summary:[/] {t.summary}")
        detail.write(f"  [dim]LLM Calls:[/] {len(t.llm_calls)}")

        # ── 개별 LLM 호출 상세 ──
        if t.llm_calls:
            detail.write("")
            for c in t.llm_calls:
                role_tag = f" [{c.role}]" if c.role else ""
                hdr = (
                    f"LLM #{c.call_id}{role_tag} "
                    f"({c.token_hint}, {c.elapsed:.1f}s)"
                )
                detail.write(f"[bold yellow]── {hdr} ──[/]")
                detail.write("")
                detail.write("[bold]Prompt:[/]")
                for line in c.prompt_preview.split("\n"):
                    detail.write(f"  [dim]{line}[/]")
                detail.write("")
                detail.write("[bold green]Response:[/]")
                for line in c.response_preview.split("\n"):
                    detail.write(f"  {line}")
                detail.write("")
        else:
            detail.write("")
            detail.write(
                "[dim]이 노드는 LLM 호출 없이 실행됨 (결정론적 노드)[/]"
            )

        # ── 노드 입출력 ──
        detail.write("")
        detail.write("[bold magenta]── Node Input (state before) ──[/]")
        detail.write(t.input_preview)
        detail.write("")
        detail.write("[bold green]── Node Output (return value) ──[/]")
        detail.write(t.output_preview)


# ══════════════════════════════════════════════════════════════
#  체크포인트 패널
# ══════════════════════════════════════════════════════════════


class CheckpointsPanel(Vertical):
    """체크포인트 조회 및 관리."""

    def compose(self) -> ComposeResult:
        yield Static("Checkpoints", classes="section-title")
        with Horizontal():
            yield Button("Refresh", variant="primary", id="btn-cp-refresh")
            yield Button("Delete Selected", variant="error", id="btn-cp-delete")
        yield DataTable(id="cp-table")

    def on_mount(self) -> None:
        table = self.query_one("#cp-table", DataTable)
        table.add_columns("Thread ID", "Size", "Keys")
        table.cursor_type = "row"
        self._refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cp-refresh":
            self._refresh()
        elif event.button.id == "btn-cp-delete":
            self._delete_selected()

    def _refresh(self) -> None:
        app: HarnessApp = self.app  # type: ignore[assignment]
        cp = JsonFileCheckpointer(app.harness_config.checkpoint_dir)
        table = self.query_one("#cp-table", DataTable)
        table.clear()

        threads = cp.list_threads()
        if not threads:
            return

        for tid in sorted(threads, reverse=True):
            state = cp.load(tid)
            if state:
                size = len(json.dumps(state, default=str))
                keys = ", ".join(sorted(state.keys())[:5])
                table.add_row(tid, f"{size:,} B", keys)

    def _delete_selected(self) -> None:
        table = self.query_one("#cp-table", DataTable)
        if table.cursor_row is not None:
            row = table.get_row_at(table.cursor_row)
            tid = str(row[0])
            app: HarnessApp = self.app  # type: ignore[assignment]
            cp = JsonFileCheckpointer(app.harness_config.checkpoint_dir)
            cp.delete(tid)
            self._refresh()


# ══════════════════════════════════════════════════════════════
#  메트릭 패널
# ══════════════════════════════════════════════════════════════


class MetricsPanel(Vertical):
    """메트릭 및 평가 결과 조회."""

    def compose(self) -> ComposeResult:
        yield Static("Metrics", classes="section-title")
        with Horizontal():
            yield Button("Refresh", variant="primary", id="btn-m-refresh")
        yield DataTable(id="metrics-table")
        yield Static("")
        yield Static("Alerts", classes="section-title")
        yield RichLog(id="alerts-log", highlight=True, markup=True)

    def on_mount(self) -> None:
        table = self.query_one("#metrics-table", DataTable)
        table.add_columns("Run ID", "Error Free", "Has Response", "Efficiency", "Timestamp")
        table.cursor_type = "row"
        self._refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-m-refresh":
            self._refresh()

    def _refresh(self) -> None:
        table = self.query_one("#metrics-table", DataTable)
        table.clear()
        alerts_log = self.query_one("#alerts-log", RichLog)
        alerts_log.clear()

        m_dir = Path("metrics")
        if not m_dir.exists():
            return

        for path in sorted(m_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                scores = data.get("scores", {})
                ts = data.get("timestamp", 0)
                ts_str = (
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "--"
                )
                table.add_row(
                    data.get("run_id", path.stem),
                    f"{scores.get('error_free', 0):.2f}",
                    f"{scores.get('has_response', 0):.2f}",
                    f"{scores.get('iteration_efficiency', 0):.2f}",
                    ts_str,
                )
            except Exception:
                continue

        alert_path = m_dir / "alerts.jsonl"
        if alert_path.exists():
            for line in alert_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                try:
                    alert = json.loads(line)
                    alerts_log.write(
                        f"[red]⚠ {alert.get('type', 'ALERT')}[/] "
                        f"avg={alert.get('avg_score', 0):.2f} "
                        f"run={alert.get('run_id', '?')}"
                    )
                except Exception:
                    continue


# ══════════════════════════════════════════════════════════════
#  실패 이력 패널
# ══════════════════════════════════════════════════════════════


class FailuresPanel(Vertical):
    """실패 이력 및 자가 개선 분석."""

    def compose(self) -> ComposeResult:
        yield Static("Failure Store", classes="section-title")
        with Horizontal():
            yield Button("Refresh", variant="primary", id="btn-f-refresh")
            yield Button("Analyze Patterns", variant="warning", id="btn-f-analyze")
        yield DataTable(id="failures-table")
        yield Static("")
        yield Static("Pattern Analysis", classes="section-title")
        yield RichLog(id="analysis-log", highlight=True, markup=True)

    def on_mount(self) -> None:
        table = self.query_one("#failures-table", DataTable)
        table.add_columns("Run ID", "Errors", "Input Preview", "Timestamp")
        table.cursor_type = "row"
        self._refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-f-refresh":
            self._refresh()
        elif event.button.id == "btn-f-analyze":
            self._analyze()

    def _refresh(self) -> None:
        table = self.query_one("#failures-table", DataTable)
        table.clear()

        f_dir = Path("failures")
        if not f_dir.exists():
            return

        for path in sorted(f_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                errors = data.get("error_log", [])
                inp = json.dumps(data.get("input", {}), ensure_ascii=False)
                ts = data.get("timestamp", 0)
                ts_str = (
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "--"
                )
                table.add_row(
                    data.get("run_id", path.stem),
                    str(len(errors)),
                    inp[:60],
                    ts_str,
                )
            except Exception:
                continue

    def _analyze(self) -> None:
        log = self.query_one("#analysis-log", RichLog)
        log.clear()

        store = LocalFailureStore()
        loop = SelfImprovementLoop(failure_store=store)

        failures = loop.collect_failures(hours=720)
        if not failures:
            log.write("[dim]분석할 실패 기록이 없습니다.[/]")
            return

        analysis = loop.analyze_failure_patterns(failures)
        log.write(f"[bold]Total failures:[/] {analysis['total_failures']}")
        log.write("")

        patterns = analysis.get("patterns", {})
        if patterns:
            log.write("[bold]Error Patterns:[/]")
            for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
                log.write(f"  [yellow]{pattern}[/]: {count}")
        else:
            log.write("[dim]패턴이 감지되지 않았습니다.[/]")

        top = analysis.get("top_pattern")
        if top:
            log.write(f"\n[bold red]Top pattern: {top}[/]")


# ══════════════════════════════════════════════════════════════
#  설정 패널
# ══════════════════════════════════════════════════════════════


class ConfigPanel(VerticalScroll):
    """AppConfig 편집 인터페이스. 설정은 .harness_config.json에 영속 저장된다."""

    def compose(self) -> ComposeResult:
        yield Static("Configuration", classes="section-title")
        yield Static(
            "[dim]Apply 시 현재 세션에 반영되고 "
            ".harness_config.json에 저장됩니다.[/]"
        )

        # ── Pipeline ──
        yield Static("")
        yield Static("Pipeline", classes="section-title")

        with Horizontal(classes="config-row"):
            yield Static("Max Iterations", classes="config-label")
            yield Input(value="10", type="integer", id="cfg-max-iter", classes="config-input")

        with Horizontal(classes="config-row"):
            yield Static("Token Budget", classes="config-label")
            yield Input(
                value="100000",
                type="integer",
                id="cfg-token-budget",
                classes="config-input",
            )

        with Horizontal(classes="config-row"):
            yield Static("Checkpoint Backend", classes="config-label")
            yield Select(
                [("memory", "memory"), ("json_file", "json_file")],
                value="memory",
                id="cfg-checkpoint-backend",
                allow_blank=False,
            )

        with Horizontal(classes="config-row"):
            yield Static("Checkpoint Dir", classes="config-label")
            yield Input(value="checkpoints", id="cfg-checkpoint-dir", classes="config-input")

        with Horizontal(classes="config-row"):
            yield Static("Guardrail Strict", classes="config-label")
            yield Switch(value=True, id="cfg-guardrail-strict")

        with Horizontal(classes="config-row"):
            yield Static("Human-in-the-Loop", classes="config-label")
            yield Switch(value=False, id="cfg-hitl")

        # ── LM Studio ──
        yield Static("")
        yield Static("LM Studio", classes="section-title")

        with Horizontal(classes="config-row"):
            yield Static("Server URL", classes="config-label")
            yield Input(value="", id="cfg-lm-url", classes="config-input")

        with Horizontal(classes="config-row"):
            yield Static("Model", classes="config-label")
            yield Input(value="", id="cfg-lm-model", classes="config-input")

        # ── Sources: Local ──
        yield Static("")
        yield Static("Source: Local Files", classes="section-title")

        with Horizontal(classes="config-row"):
            yield Static("Default Directory", classes="config-label")
            yield Input(value="", id="cfg-src-local-dir", classes="config-input")

        with Horizontal(classes="config-row"):
            yield Static("Chunk Size", classes="config-label")
            yield Input(
                value="1000", type="integer",
                id="cfg-src-chunk-size", classes="config-input",
            )

        with Horizontal(classes="config-row"):
            yield Static("Chunk Overlap", classes="config-label")
            yield Input(
                value="200", type="integer",
                id="cfg-src-chunk-overlap", classes="config-input",
            )

        # ── Sources: Web Search ──
        yield Static("")
        yield Static("Source: Web Search", classes="section-title")

        with Horizontal(classes="config-row"):
            yield Static("Max Results", classes="config-label")
            yield Input(
                value="5", type="integer",
                id="cfg-src-web-max", classes="config-input",
            )

        with Horizontal(classes="config-row"):
            yield Static("Page Timeout (s)", classes="config-label")
            yield Input(
                value="20", type="integer",
                id="cfg-src-web-timeout", classes="config-input",
            )

        yield Static("")
        with Horizontal():
            yield Button("Apply & Save", variant="primary", id="btn-cfg-apply")
            yield Button(
                "Reset to Defaults", variant="warning", id="btn-cfg-reset"
            )

    def on_mount(self) -> None:
        self._load_from_app_config()

    def _load_from_app_config(self) -> None:
        app: HarnessApp = self.app  # type: ignore[assignment]
        cfg = app.app_config

        self.query_one("#cfg-max-iter", Input).value = str(cfg.max_iterations)
        self.query_one("#cfg-token-budget", Input).value = str(cfg.max_tokens_budget)
        self.query_one("#cfg-checkpoint-backend", Select).value = cfg.checkpoint_backend
        self.query_one("#cfg-checkpoint-dir", Input).value = cfg.checkpoint_dir
        self.query_one("#cfg-guardrail-strict", Switch).value = cfg.guardrail_strict
        self.query_one("#cfg-hitl", Switch).value = cfg.human_in_the_loop

        self.query_one("#cfg-lm-url", Input).value = cfg.lm_studio_url
        self.query_one("#cfg-lm-model", Input).value = cfg.lm_studio_model

        src = cfg.sources
        self.query_one("#cfg-src-local-dir", Input).value = src.local_directory
        self.query_one("#cfg-src-chunk-size", Input).value = str(src.local_chunk_size)
        self.query_one("#cfg-src-chunk-overlap", Input).value = str(
            src.local_chunk_overlap
        )
        self.query_one("#cfg-src-web-max", Input).value = str(
            src.web_search_max_results
        )
        self.query_one("#cfg-src-web-timeout", Input).value = str(
            src.web_search_page_timeout
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cfg-apply":
            self._apply_and_save()
        elif event.button.id == "btn-cfg-reset":
            self._reset()

    def _apply_and_save(self) -> None:
        app: HarnessApp = self.app  # type: ignore[assignment]
        try:
            new_cfg = AppConfig(
                max_iterations=int(
                    self.query_one("#cfg-max-iter", Input).value or "10"
                ),
                max_tokens_budget=int(
                    self.query_one("#cfg-token-budget", Input).value or "100000"
                ),
                checkpoint_backend=str(
                    self.query_one("#cfg-checkpoint-backend", Select).value
                ),
                checkpoint_dir=self.query_one("#cfg-checkpoint-dir", Input).value
                or "checkpoints",
                guardrail_strict=self.query_one(
                    "#cfg-guardrail-strict", Switch
                ).value,
                human_in_the_loop=self.query_one("#cfg-hitl", Switch).value,
                lm_studio_url=self.query_one("#cfg-lm-url", Input).value
                or "http://169.254.83.107:1234/v1",
                lm_studio_model=self.query_one("#cfg-lm-model", Input).value
                or "qwen/qwen3.5-9b",
                sources=SourceConfig(
                    local_enabled=app.app_config.sources.local_enabled,
                    local_directory=self.query_one(
                        "#cfg-src-local-dir", Input
                    ).value,
                    local_chunk_size=int(
                        self.query_one("#cfg-src-chunk-size", Input).value
                        or "1000"
                    ),
                    local_chunk_overlap=int(
                        self.query_one("#cfg-src-chunk-overlap", Input).value
                        or "200"
                    ),
                    web_search_enabled=app.app_config.sources.web_search_enabled,
                    web_search_max_results=int(
                        self.query_one("#cfg-src-web-max", Input).value or "5"
                    ),
                    web_search_page_timeout=int(
                        self.query_one("#cfg-src-web-timeout", Input).value
                        or "20"
                    ),
                ),
            )
            app.app_config = new_cfg
            app.harness_config = new_cfg.to_harness_config()
            save_config(new_cfg)
            self.app.notify("Saved to .harness_config.json", severity="information")
        except Exception as e:
            self.app.notify(f"Error: {e}", severity="error")

    def _reset(self) -> None:
        app: HarnessApp = self.app  # type: ignore[assignment]
        app.app_config = AppConfig()
        app.harness_config = app.app_config.to_harness_config()
        save_config(app.app_config)
        self._load_from_app_config()
        self.app.notify("Reset to defaults", severity="information")


# ══════════════════════════════════════════════════════════════
#  가드레일 테스트 패널
# ══════════════════════════════════════════════════════════════


class GuardrailPanel(Vertical):
    """가드레일 인라인 테스트."""

    def compose(self) -> ComposeResult:
        yield Static("Guardrail Tester", classes="section-title")
        yield Static("[dim]Test input/output guardrails interactively.[/]")
        yield Static("")

        with TabbedContent():
            with TabPane("Input Guardrail", id="tab-input"):
                yield TextArea(id="guard-input-text")
                yield Button("Check Input", variant="primary", id="btn-guard-input")
                yield RichLog(id="guard-input-log", highlight=True, markup=True)

            with TabPane("Output Guardrail", id="tab-output"):
                yield TextArea(id="guard-output-text")
                yield Button("Check Output", variant="primary", id="btn-guard-output")
                yield RichLog(id="guard-output-log", highlight=True, markup=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-guard-input":
            self._check_input()
        elif event.button.id == "btn-guard-output":
            self._check_output()

    def _check_input(self) -> None:
        text = self.query_one("#guard-input-text", TextArea).text
        log = self.query_one("#guard-input-log", RichLog)
        log.clear()
        if not text.strip():
            log.write("[dim]텍스트를 입력하세요.[/]")
            return

        app: HarnessApp = self.app  # type: ignore[assignment]
        guard = InputGuardrail(strict=app.harness_config.guardrail_strict)
        result = guard.check(text)

        if result.passed:
            log.write("[green]✓ PASSED[/]")
            if result.sanitized_content != text:
                log.write(f"[yellow]Sanitized:[/] {result.sanitized_content[:200]}")
        else:
            log.write(f"[red]✗ BLOCKED: {result.reason}[/]")
            log.write(f"[dim]{result.rejection_message}[/]")

    def _check_output(self) -> None:
        text = self.query_one("#guard-output-text", TextArea).text
        log = self.query_one("#guard-output-log", RichLog)
        log.clear()
        if not text.strip():
            log.write("[dim]텍스트를 입력하세요.[/]")
            return

        app: HarnessApp = self.app  # type: ignore[assignment]
        guard = OutputGuardrail(strict=app.harness_config.guardrail_strict)
        result = guard.check(text)

        if result.passed:
            log.write("[green]✓ PASSED[/]")
        else:
            log.write(f"[red]✗ BLOCKED: {result.reason}[/]")
            log.write(f"[dim]{result.rejection_message}[/]")


# ══════════════════════════════════════════════════════════════
#  메인 앱
# ══════════════════════════════════════════════════════════════


PANEL_KEYS = [
    "dashboard",
    "research",
    "trace",
    "checkpoints",
    "metrics",
    "failures",
    "guardrails",
    "config",
]

PANEL_LABELS = {
    "dashboard": "Dashboard",
    "research": "Research",
    "trace": "Trace",
    "checkpoints": "Checkpoints",
    "metrics": "Metrics",
    "failures": "Failures",
    "guardrails": "Guardrails",
    "config": "Config",
}


class HarnessApp(App):
    """LangGraph Agent Harness TUI.

    모든 패널을 한 번에 마운트하고 display 토글로 전환한다.
    파이프라인 실행 세션(current_session)은 앱 레벨에 보관된다.
    """

    TITLE = "LangGraph Harness"
    SUB_TITLE = "Zero-Hallucination Research Pipeline"
    CSS = APP_CSS

    BINDINGS = [
        Binding("1", "nav('dashboard')", "Dashboard"),
        Binding("2", "nav('research')", "Research"),
        Binding("3", "nav('trace')", "Trace"),
        Binding("4", "nav('checkpoints')", "Checkpoints"),
        Binding("5", "nav('metrics')", "Metrics"),
        Binding("6", "nav('failures')", "Failures"),
        Binding("7", "nav('guardrails')", "Guardrails"),
        Binding("8", "nav('config')", "Config"),
        Binding("q", "quit", "Quit"),
    ]

    app_config: AppConfig = AppConfig()
    harness_config: HarnessConfig = HarnessConfig()
    current_session: RunSession | None = None
    _current_panel: str = "dashboard"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app_config = load_config()
        self.harness_config = self.app_config.to_harness_config()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Static("[bold]Navigation[/]", classes="nav-label")
                for key in PANEL_KEYS:
                    yield Button(
                        PANEL_LABELS[key], id=f"nav-{key}", classes="nav-btn"
                    )
                yield SessionBox(id="session-box")
            with Vertical(id="main-content"):
                yield DashboardPanel(id="panel-dashboard", classes="panel -visible")
                yield ResearchPanel(id="panel-research", classes="panel")
                yield TracePanel(id="panel-trace", classes="panel")
                yield CheckpointsPanel(id="panel-checkpoints", classes="panel")
                yield MetricsPanel(id="panel-metrics", classes="panel")
                yield FailuresPanel(id="panel-failures", classes="panel")
                yield GuardrailPanel(id="panel-guardrails", classes="panel")
                yield ConfigPanel(id="panel-config", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        self._highlight_nav("dashboard")
        self.query_one("#panel-dashboard", DashboardPanel).refresh_stats()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith("nav-"):
            panel_key = btn_id[4:]
            if panel_key in PANEL_KEYS:
                self.action_nav(panel_key)

    def action_nav(self, panel_key: str) -> None:
        if panel_key == self._current_panel:
            return

        # 이전 패널 숨기기
        old = self.query_one(f"#panel-{self._current_panel}")
        old.remove_class("-visible")

        # 새 패널 표시
        new = self.query_one(f"#panel-{panel_key}")
        new.add_class("-visible")

        self._current_panel = panel_key
        self._highlight_nav(panel_key)

        # 패널 진입 시 자동 새로고침
        if panel_key == "dashboard":
            self.query_one("#panel-dashboard", DashboardPanel).refresh_stats()
        elif panel_key == "trace":
            self.query_one("#panel-trace", TracePanel).refresh_traces()

    def _highlight_nav(self, active_key: str) -> None:
        for key in PANEL_KEYS:
            btn = self.query_one(f"#nav-{key}", Button)
            btn.remove_class("-active")
            if key == active_key:
                btn.add_class("-active")

    def refresh_trace_panel(self) -> None:
        """트레이스 패널이 보이는 상태이면 실시간 갱신한다."""
        trace_panel = self.query_one("#panel-trace", TracePanel)
        if trace_panel.has_class("-visible"):
            trace_panel.refresh_traces()


def main() -> None:
    """TUI 엔트리포인트."""
    app = HarnessApp()
    app.run()


if __name__ == "__main__":
    main()
