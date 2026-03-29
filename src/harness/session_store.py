"""세션 디렉토리 영속 저장소.

하나의 연구 세션에 관한 모든 데이터를 단일 디렉토리에 파일로 저장한다.

디렉토리 구조:
    sessions/{run_id}/
    ├── session.json           # 세션 메타데이터 (run_id, query, status, timing)
    ├── config.json            # 실행 시점의 AppConfig 스냅샷
    ├── traces.json            # 노드 트레이스 + LLM 호출 상세
    ├── checkpoint.json        # 파이프라인 최종 상태
    ├── evaluation.json        # 평가 점수
    ├── input_guardrail.json   # 입력 가드레일 결과
    ├── output_guardrail.json  # 출력 가드레일 결과
    ├── knowledge_base.json    # 지식 베이스 통계 + 소스 목록
    ├── failure.json           # 실패 기록 (실패 시에만)
    └── result.md              # 최종 연구 결과 텍스트
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


class SessionStore:
    """세션별 디렉토리에 모든 데이터를 파일로 저장한다.

    사용:
        store = SessionStore("run_abc123", base_dir="sessions")
        store.save_session_meta(run_id, query, status, ...)
        store.save_config(app_config)
        store.save_traces(traces, all_llm_calls)
        ...
    """

    def __init__(self, run_id: str, base_dir: str = "sessions"):
        self.run_id = run_id
        self.dir = Path(base_dir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, filename: str, data: Any) -> Path:
        path = self.dir / filename
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def _write_text(self, filename: str, text: str) -> Path:
        path = self.dir / filename
        path.write_text(text, encoding="utf-8")
        return path

    # ── 세션 메타 ──

    def save_session_meta(
        self,
        query: str,
        status: str,
        start_time: float,
        elapsed: float,
        step_count: int,
        error_msg: str = "",
        last_node: str = "",
        completed_nodes: list[str] | None = None,
        phase: str = "",
    ) -> None:
        """세션 메타데이터를 저장한다.

        Args:
            phase: 에러 발생 시 어느 단계였는지
                   (e.g. "input_guardrail", "lm_connect", "build",
                   "streaming", "output_guardrail", "evaluation")
        """
        self._write_json("session.json", {
            "run_id": self.run_id,
            "query": query,
            "status": status,
            "start_time": start_time,
            "start_time_iso": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(start_time)
            ),
            "elapsed": round(elapsed, 2),
            "step_count": step_count,
            "error_msg": error_msg,
            "last_node": last_node,
            "completed_nodes": completed_nodes or [],
            "phase": phase,
        })

    # ── 설정 스냅샷 ──

    def save_config(self, app_config) -> None:
        """실행 시점의 AppConfig를 저장한다."""
        self._write_json("config.json", asdict(app_config))

    # ── 트레이스 ──

    def save_traces(self, traces: list, all_llm_calls: list) -> None:
        """NodeTrace와 LLMCall 전체를 저장한다."""
        trace_data = []
        for t in traces:
            td = {
                "node_name": t.node_name,
                "label": t.label,
                "desc": t.desc,
                "elapsed": round(t.elapsed, 2),
                "summary": t.summary,
                "input_preview": t.input_preview,
                "output_preview": t.output_preview,
                "llm_calls": [
                    {
                        "call_id": c.call_id,
                        "node_name": c.node_name,
                        "role": getattr(c, "role", ""),
                        "prompt_preview": c.prompt_preview,
                        "response_preview": c.response_preview,
                        "elapsed": round(c.elapsed, 2),
                        "token_hint": c.token_hint,
                    }
                    for c in t.llm_calls
                ],
            }
            trace_data.append(td)

        all_calls_data = [
            {
                "call_id": c.call_id,
                "node_name": c.node_name,
                "role": getattr(c, "role", ""),
                "prompt_preview": c.prompt_preview,
                "response_preview": c.response_preview,
                "elapsed": round(c.elapsed, 2),
                "token_hint": c.token_hint,
            }
            for c in all_llm_calls
        ]

        self._write_json("traces.json", {
            "node_traces": trace_data,
            "all_llm_calls": all_calls_data,
            "total_nodes": len(trace_data),
            "total_llm_calls": len(all_calls_data),
        })

    # ── 체크포인트 ──

    def save_checkpoint(self, state: dict) -> None:
        """파이프라인 최종 상태를 저장한다."""
        self._write_json("checkpoint.json", _make_serializable(state))

    # ── 평가 ──

    def save_evaluation(self, scores: dict[str, float]) -> None:
        self._write_json("evaluation.json", {
            "run_id": self.run_id,
            "scores": scores,
            "timestamp": time.time(),
        })

    # ── 가드레일 ──

    def save_input_guardrail(
        self, passed: bool, reason: str, sanitized: str, original: str
    ) -> None:
        self._write_json("input_guardrail.json", {
            "passed": passed,
            "reason": reason,
            "sanitized_content": sanitized,
            "original_query": original,
            "pii_masked": sanitized != original,
        })

    def save_output_guardrail(
        self, passed: bool, reason: str, content: str
    ) -> None:
        self._write_json("output_guardrail.json", {
            "passed": passed,
            "reason": reason,
            "content_length": len(content),
        })

    # ── 지식 베이스 ──

    def save_knowledge_base(
        self, stats: dict, sources: list[str] | None = None
    ) -> None:
        self._write_json("knowledge_base.json", {
            "stats": stats,
            "sources_used": sources or [],
        })

    # ── 실패 기록 ──

    def save_failure(
        self,
        error_log: list[str],
        input_data: dict,
        traceback_str: str = "",
        phase: str = "",
        last_node: str = "",
        partial_state: dict | None = None,
    ) -> None:
        """실패 기록을 저장한다.

        Args:
            traceback_str: Python traceback 문자열.
            phase: 에러 발생 파이프라인 단계.
            last_node: 마지막으로 완료된 노드.
            partial_state: 에러 시점의 부분 상태 (있으면).
        """
        failure_data: dict[str, Any] = {
            "run_id": self.run_id,
            "input": input_data,
            "error_log": error_log,
            "phase": phase,
            "last_node": last_node,
            "timestamp": time.time(),
        }
        if traceback_str:
            failure_data["traceback"] = traceback_str
        if partial_state:
            failure_data["partial_state"] = _make_serializable(partial_state)
        self._write_json("failure.json", failure_data)

    # ── 최종 결과 ──

    def save_result(self, text: str) -> None:
        self._write_text("result.md", text)

    # ── 유틸 ──

    @staticmethod
    def list_sessions(base_dir: str = "sessions") -> list[dict]:
        """저장된 세션 목록을 반환한다."""
        base = Path(base_dir)
        if not base.exists():
            return []

        sessions = []
        for d in sorted(base.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            meta_path = d / "session.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["dir"] = str(d)
                    sessions.append(meta)
                except (json.JSONDecodeError, OSError):
                    sessions.append({"run_id": d.name, "dir": str(d)})
            else:
                sessions.append({"run_id": d.name, "dir": str(d)})
        return sessions


def _make_serializable(obj: Any) -> Any:
    """중첩 객체를 JSON 직렬화 가능한 형태로 변환한다."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "content") and hasattr(obj, "type"):
        return {"type": obj.type, "content": obj.content}
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {"__type__": type(obj).__name__, **_make_serializable(obj.__dict__)}
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
