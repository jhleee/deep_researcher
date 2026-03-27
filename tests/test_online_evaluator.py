"""online_evaluator.py 유닛 테스트."""

import json

import pytest

from harness.online_evaluator import OnlineEvaluator


class TestOnlineEvaluator:
    """OnlineEvaluator 테스트."""

    def test_evaluate_clean_run(self, tmp_dir):
        evaluator = OnlineEvaluator(metrics_dir=str(tmp_dir / "metrics"))
        state = {
            "messages": [{"role": "assistant", "content": "답변"}],
            "error_log": [],
            "iteration_count": 2,
            "metadata": {"max_iterations": 10},
        }
        scores = evaluator.evaluate_run(state, "run-001")
        assert scores["error_free"] == 1.0
        assert scores["has_response"] == 1.0
        assert scores["iteration_efficiency"] == 0.8  # 1 - 2/10

    def test_evaluate_error_run(self, tmp_dir):
        evaluator = OnlineEvaluator(metrics_dir=str(tmp_dir / "metrics"))
        state = {
            "messages": [],
            "error_log": ["SOME_ERROR"],
            "iteration_count": 10,
            "metadata": {"max_iterations": 10},
        }
        scores = evaluator.evaluate_run(state, "run-002")
        assert scores["error_free"] == 0.0
        assert scores["has_response"] == 0.0
        assert scores["iteration_efficiency"] == 0.0

    def test_metrics_saved_to_file(self, tmp_dir):
        metrics_dir = tmp_dir / "metrics"
        evaluator = OnlineEvaluator(metrics_dir=str(metrics_dir))
        state = {
            "messages": [{"content": "ok"}],
            "error_log": [],
            "iteration_count": 0,
            "metadata": {},
        }
        evaluator.evaluate_run(state, "run-save-test")
        metric_file = metrics_dir / "run-save-test.json"
        assert metric_file.exists()
        data = json.loads(metric_file.read_text())
        assert data["run_id"] == "run-save-test"
        assert "scores" in data

    def test_alert_on_low_quality(self, tmp_dir):
        metrics_dir = tmp_dir / "metrics"
        evaluator = OnlineEvaluator(
            metrics_dir=str(metrics_dir),
            alert_threshold=0.9,
            buffer_size=3,
        )
        bad_state = {
            "messages": [],
            "error_log": ["ERR"],
            "iteration_count": 10,
            "metadata": {"max_iterations": 10},
        }
        for i in range(5):
            evaluator.evaluate_run(bad_state, f"run-bad-{i}")

        alert_file = metrics_dir / "alerts.jsonl"
        assert alert_file.exists()
        alerts = alert_file.read_text().strip().split("\n")
        assert len(alerts) >= 1
        alert = json.loads(alerts[0])
        assert alert["type"] == "QUALITY_DEGRADATION"
