"""자가 개선 루프 (로컬 파일 기반).

실패 궤적을 수집·분석하고 개선안을 생성하는 파이프라인의 로컬 구현.
LangSmith 대신 로컬 JSON 파일에서 실패 데이터를 읽는다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FailedRun:
    """실패한 실행 기록."""

    run_id: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    error_log: list[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ImprovementCandidate:
    """개선안 후보."""

    description: str
    changes: dict[str, Any]
    score: float = 0.0


class LocalFailureStore:
    """로컬 파일 기반 실패 궤적 저장소."""

    def __init__(self, store_path: str = "failures"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

    def record_failure(self, run: FailedRun) -> None:
        """실패 기록을 저장한다."""
        path = self.store_path / f"{run.run_id}.json"
        path.write_text(
            json.dumps({
                "run_id": run.run_id,
                "input": run.input_data,
                "output": run.output_data,
                "error_log": run.error_log,
                "timestamp": run.timestamp,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def collect_recent_failures(self, since_timestamp: float) -> list[FailedRun]:
        """지정 시점 이후의 실패 기록을 수집한다."""
        failures = []
        for path in self.store_path.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("timestamp", 0) >= since_timestamp:
                failures.append(FailedRun(
                    run_id=data["run_id"],
                    input_data=data.get("input", {}),
                    output_data=data.get("output", {}),
                    error_log=data.get("error_log", []),
                    timestamp=data.get("timestamp", 0),
                ))
        return failures

    def count(self) -> int:
        return len(list(self.store_path.glob("*.json")))


class LocalRegressionDataset:
    """로컬 파일 기반 회귀 테스트 데이터셋."""

    def __init__(self, dataset_path: str = "tests/datasets/regression.json"):
        self.dataset_path = Path(dataset_path)
        self._entries: list[dict[str, Any]] = []
        if self.dataset_path.exists():
            self._entries = json.loads(
                self.dataset_path.read_text(encoding="utf-8")
            )

    def add_entry(self, input_data: dict, expected_behavior: str = "should_not_fail") -> None:
        """회귀 테스트 엔트리를 추가한다."""
        self._entries.append({
            "input": input_data,
            "expected_behavior": expected_behavior,
            "added_at": time.time(),
        })
        self._save()

    def get_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def _save(self) -> None:
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class SelfImprovementLoop:
    """자가 개선 루프.

    실패 궤적을 수집 → 패턴 분석 → 개선안 생성 → 회귀 데이터셋 갱신.
    LLM 기반 분석 부분은 별도의 모델 주입으로 확장 가능하도록 설계.
    """

    def __init__(
        self,
        failure_store: LocalFailureStore | None = None,
        regression_dataset: LocalRegressionDataset | None = None,
    ):
        self.failure_store = failure_store or LocalFailureStore()
        self.regression_dataset = regression_dataset or LocalRegressionDataset()

    def collect_failures(self, hours: int = 24) -> list[FailedRun]:
        """최근 N시간의 실패 궤적을 수집한다."""
        since = time.time() - (hours * 3600)
        return self.failure_store.collect_recent_failures(since)

    def analyze_failure_patterns(self, failures: list[FailedRun]) -> dict[str, Any]:
        """실패 패턴을 분석한다 (규칙 기반 버전)."""
        patterns: dict[str, int] = {}
        for failure in failures:
            for error in failure.error_log:
                # 오류 유형별 카운트
                if "BLOCKED" in error:
                    patterns["guardrail_block"] = patterns.get("guardrail_block", 0) + 1
                elif "SANITIZER" in error:
                    patterns["sanitizer_fail"] = patterns.get("sanitizer_fail", 0) + 1
                elif "TIMEOUT" in error:
                    patterns["timeout"] = patterns.get("timeout", 0) + 1
                else:
                    patterns["unknown"] = patterns.get("unknown", 0) + 1

        return {
            "total_failures": len(failures),
            "patterns": patterns,
            "top_pattern": max(patterns, key=patterns.get) if patterns else None,
        }

    def update_regression_dataset(self, failures: list[FailedRun]) -> int:
        """실패 케이스를 회귀 데이터셋에 추가한다."""
        added = 0
        for failure in failures:
            self.regression_dataset.add_entry(
                input_data=failure.input_data,
                expected_behavior="should_not_fail",
            )
            added += 1
        return added
