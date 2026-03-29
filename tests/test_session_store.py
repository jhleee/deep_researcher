"""SessionStore 및 세션 디렉토리 E2E 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.config_store import AppConfig, SourceConfig
from harness.session_store import SessionStore


class TestSessionStore:
    def test_creates_directory(self, tmp_path):
        store = SessionStore("run_test1", base_dir=str(tmp_path))
        assert (tmp_path / "run_test1").is_dir()

    def test_save_session_meta(self, tmp_path):
        store = SessionStore("run_test2", base_dir=str(tmp_path))
        store.save_session_meta(
            query="테스트 쿼리",
            status="done",
            start_time=1000.0,
            elapsed=12.5,
            step_count=7,
        )
        data = json.loads(
            (tmp_path / "run_test2" / "session.json").read_text(encoding="utf-8")
        )
        assert data["run_id"] == "run_test2"
        assert data["query"] == "테스트 쿼리"
        assert data["status"] == "done"
        assert data["step_count"] == 7
        assert "start_time_iso" in data

    def test_save_config(self, tmp_path):
        store = SessionStore("run_cfg", base_dir=str(tmp_path))
        cfg = AppConfig(
            max_iterations=5,
            sources=SourceConfig(local_enabled=True, local_directory="/data"),
        )
        store.save_config(cfg)
        data = json.loads(
            (tmp_path / "run_cfg" / "config.json").read_text(encoding="utf-8")
        )
        assert data["max_iterations"] == 5
        assert data["sources"]["local_enabled"] is True

    def test_save_traces(self, tmp_path):
        from harness.tui import LLMCall, NodeTrace

        store = SessionStore("run_trace", base_dir=str(tmp_path))
        call = LLMCall(
            call_id=1,
            node_name="planner",
            prompt_preview="test prompt",
            response_preview="test response",
            elapsed=1.5,
            token_hint="100자 → 50자",
        )
        trace = NodeTrace(
            node_name="planner",
            label="1/7 Planner",
            desc="실행 계획 생성",
            elapsed=2.0,
            summary="하위 작업 3개 생성",
            input_preview='{"messages": [...]}',
            output_preview='{"plan": [...]}',
            llm_calls=[call],
        )
        store.save_traces([trace], [call])
        data = json.loads(
            (tmp_path / "run_trace" / "traces.json").read_text(encoding="utf-8")
        )
        assert data["total_nodes"] == 1
        assert data["total_llm_calls"] == 1
        assert data["node_traces"][0]["llm_calls"][0]["prompt_preview"] == "test prompt"

    def test_save_checkpoint(self, tmp_path):
        store = SessionStore("run_cp", base_dir=str(tmp_path))
        store.save_checkpoint({"messages": [], "plan": [{"id": "t1"}]})
        data = json.loads(
            (tmp_path / "run_cp" / "checkpoint.json").read_text(encoding="utf-8")
        )
        assert data["plan"][0]["id"] == "t1"

    def test_save_evaluation(self, tmp_path):
        store = SessionStore("run_eval", base_dir=str(tmp_path))
        store.save_evaluation({"error_free": 1.0, "has_response": 1.0})
        data = json.loads(
            (tmp_path / "run_eval" / "evaluation.json").read_text(encoding="utf-8")
        )
        assert data["scores"]["error_free"] == 1.0

    def test_save_guardrails(self, tmp_path):
        store = SessionStore("run_guard", base_dir=str(tmp_path))
        store.save_input_guardrail(True, "", "sanitized", "original")
        store.save_output_guardrail(True, "", "output text")
        assert (tmp_path / "run_guard" / "input_guardrail.json").exists()
        assert (tmp_path / "run_guard" / "output_guardrail.json").exists()

    def test_save_knowledge_base(self, tmp_path):
        store = SessionStore("run_kb", base_dir=str(tmp_path))
        store.save_knowledge_base(
            {"files": 3, "chunks": 15, "chars": 5000},
            ["Local(3)", "Web"],
        )
        data = json.loads(
            (tmp_path / "run_kb" / "knowledge_base.json").read_text(encoding="utf-8")
        )
        assert data["stats"]["files"] == 3
        assert "Web" in data["sources_used"]

    def test_save_failure(self, tmp_path):
        store = SessionStore("run_fail", base_dir=str(tmp_path))
        store.save_failure(["timeout error"], {"query": "test"})
        data = json.loads(
            (tmp_path / "run_fail" / "failure.json").read_text(encoding="utf-8")
        )
        assert data["error_log"][0] == "timeout error"

    def test_save_result(self, tmp_path):
        store = SessionStore("run_result", base_dir=str(tmp_path))
        store.save_result("# 연구 결과\n\n최종 보고서 내용입니다.")
        text = (tmp_path / "run_result" / "result.md").read_text(encoding="utf-8")
        assert "연구 결과" in text

    def test_list_sessions(self, tmp_path):
        SessionStore("run_a", base_dir=str(tmp_path))
        s1 = SessionStore("run_b", base_dir=str(tmp_path))
        s1.save_session_meta("q1", "done", 1000.0, 5.0, 3)
        s2 = SessionStore("run_c", base_dir=str(tmp_path))
        s2.save_session_meta("q2", "error", 2000.0, 1.0, 1, "fail")

        sessions = SessionStore.list_sessions(str(tmp_path))
        assert len(sessions) == 3
        # meta가 있는 세션에는 status 필드가 있음
        with_meta = [s for s in sessions if "status" in s]
        assert len(with_meta) == 2


class TestFullSessionDirectory:
    """전체 세션 디렉토리 구조가 올바르게 생성되는지 검증."""

    def test_complete_session_files(self, tmp_path):
        from harness.tui import LLMCall, NodeTrace

        store = SessionStore("run_full", base_dir=str(tmp_path))
        session_dir = tmp_path / "run_full"

        store.save_config(AppConfig())
        store.save_input_guardrail(True, "", "clean query", "clean query")
        store.save_knowledge_base({"files": 2, "chunks": 10, "chars": 3000})

        call = LLMCall(1, "planner", "prompt", "response", 1.0, "50자 → 30자")
        trace = NodeTrace("planner", "1/7", "계획", 1.0, "3개", llm_calls=[call])
        store.save_traces([trace], [call])

        store.save_checkpoint({"messages": [], "draft": "test draft"})
        store.save_evaluation({"error_free": 1.0, "has_response": 1.0})
        store.save_output_guardrail(True, "", "test draft")
        store.save_result("# Result\n\nFinal output here.")
        store.save_session_meta("테스트 쿼리", "done", 1000.0, 10.0, 5)

        # 모든 파일 존재 확인
        expected_files = [
            "session.json",
            "config.json",
            "input_guardrail.json",
            "knowledge_base.json",
            "traces.json",
            "checkpoint.json",
            "evaluation.json",
            "output_guardrail.json",
            "result.md",
        ]
        for filename in expected_files:
            assert (session_dir / filename).exists(), f"Missing: {filename}"

        # JSON 파싱 가능 확인
        for filename in expected_files:
            if filename.endswith(".json"):
                data = json.loads(
                    (session_dir / filename).read_text(encoding="utf-8")
                )
                assert isinstance(data, dict)
