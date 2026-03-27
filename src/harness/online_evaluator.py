"""온라인 평가기 (로컬 파일 기반).

프로덕션 모니터링의 로컬 구현. LangSmith 대신 JSON 파일에 메트릭을 기록한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalScore:
    """개별 평가 점수."""

    key: str
    score: float
    timestamp: float = field(default_factory=time.time)


class OnlineEvaluator:
    """로컬 파일 기반 온라인 평가기.

    실행 결과를 평가하고 메트릭을 JSON 파일에 기록한다.
    """

    def __init__(
        self,
        metrics_dir: str = "metrics",
        alert_threshold: float = 0.7,
        buffer_size: int = 100,
    ):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = alert_threshold
        self.buffer_size = buffer_size
        self._buffer: list[dict[str, float]] = []

    def evaluate_run(self, state: dict[str, Any], run_id: str) -> dict[str, float]:
        """실행 결과를 평가하고 점수를 반환한다."""
        scores: dict[str, float] = {}

        # 1. 오류 발생 여부
        error_log = state.get("error_log", [])
        scores["error_free"] = 1.0 if not error_log else 0.0

        # 2. 응답 존재 여부
        messages = state.get("messages", [])
        scores["has_response"] = 1.0 if messages else 0.0

        # 3. 반복 효율성
        iteration_count = state.get("iteration_count", 0)
        max_iter = state.get("metadata", {}).get("max_iterations", 10)
        if max_iter > 0:
            scores["iteration_efficiency"] = max(0.0, 1.0 - iteration_count / max_iter)

        # 메트릭 버퍼에 추가
        self._buffer.append(scores)
        self._save_metric(run_id, scores)

        # 롤링 평균 체크
        if len(self._buffer) >= self.buffer_size:
            avg = self._compute_rolling_average()
            if avg < self.threshold:
                self._fire_alert(avg, run_id)
            self._buffer = self._buffer[-self.buffer_size // 2 :]

        return scores

    def _save_metric(self, run_id: str, scores: dict[str, float]) -> None:
        path = self.metrics_dir / f"{run_id}.json"
        path.write_text(
            json.dumps({"run_id": run_id, "scores": scores, "timestamp": time.time()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _compute_rolling_average(self) -> float:
        if not self._buffer:
            return 1.0
        all_scores = [
            sum(s.values()) / len(s) for s in self._buffer if s
        ]
        return sum(all_scores) / len(all_scores)

    def _fire_alert(self, avg_score: float, run_id: str) -> None:
        alert = {
            "type": "QUALITY_DEGRADATION",
            "avg_score": avg_score,
            "threshold": self.threshold,
            "run_id": run_id,
            "timestamp": time.time(),
        }
        alert_path = self.metrics_dir / "alerts.jsonl"
        with open(alert_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert, ensure_ascii=False) + "\n")
