"""self_improvement.py 유닛 테스트."""

import time

import pytest

from harness.self_improvement import (
    FailedRun,
    LocalFailureStore,
    LocalRegressionDataset,
    SelfImprovementLoop,
)


class TestLocalFailureStore:
    """LocalFailureStore 테스트."""

    def test_record_and_collect(self, tmp_dir):
        store = LocalFailureStore(store_path=str(tmp_dir / "failures"))
        run = FailedRun(
            run_id="fail-001",
            input_data={"query": "test"},
            output_data={"result": None},
            error_log=["BLOCKED: injection"],
            timestamp=time.time(),
        )
        store.record_failure(run)
        assert store.count() == 1

        recent = store.collect_recent_failures(since_timestamp=time.time() - 60)
        assert len(recent) == 1
        assert recent[0].run_id == "fail-001"

    def test_collect_filters_old(self, tmp_dir):
        store = LocalFailureStore(store_path=str(tmp_dir / "failures"))
        old_run = FailedRun(
            run_id="old-run",
            input_data={},
            output_data={},
            error_log=["ERR"],
            timestamp=time.time() - 100_000,
        )
        store.record_failure(old_run)

        recent = store.collect_recent_failures(since_timestamp=time.time() - 60)
        assert len(recent) == 0

    def test_empty_store(self, tmp_dir):
        store = LocalFailureStore(store_path=str(tmp_dir / "empty"))
        assert store.count() == 0
        assert store.collect_recent_failures(0) == []


class TestLocalRegressionDataset:
    """LocalRegressionDataset 테스트."""

    def test_add_and_get(self, tmp_dir):
        ds = LocalRegressionDataset(dataset_path=str(tmp_dir / "regression.json"))
        ds.add_entry({"query": "test"}, "should_not_fail")
        entries = ds.get_entries()
        assert len(entries) == 1
        assert entries[0]["input"] == {"query": "test"}
        assert entries[0]["expected_behavior"] == "should_not_fail"

    def test_persistence(self, tmp_dir):
        path = str(tmp_dir / "regression.json")
        ds1 = LocalRegressionDataset(dataset_path=path)
        ds1.add_entry({"q": "a"})
        ds1.add_entry({"q": "b"})

        ds2 = LocalRegressionDataset(dataset_path=path)
        assert len(ds2.get_entries()) == 2


class TestSelfImprovementLoop:
    """SelfImprovementLoop 테스트."""

    def test_collect_failures(self, tmp_dir):
        store = LocalFailureStore(store_path=str(tmp_dir / "failures"))
        store.record_failure(FailedRun(
            run_id="f1", input_data={}, output_data={},
            error_log=["BLOCKED"], timestamp=time.time(),
        ))
        loop = SelfImprovementLoop(failure_store=store)
        failures = loop.collect_failures(hours=1)
        assert len(failures) == 1

    def test_analyze_patterns(self, tmp_dir):
        loop = SelfImprovementLoop()
        failures = [
            FailedRun("r1", {}, {}, ["BLOCKED: injection"], time.time()),
            FailedRun("r2", {}, {}, ["BLOCKED: pii"], time.time()),
            FailedRun("r3", {}, {}, ["SANITIZER_FAIL: fake cite"], time.time()),
        ]
        analysis = loop.analyze_failure_patterns(failures)
        assert analysis["total_failures"] == 3
        assert "guardrail_block" in analysis["patterns"]
        assert "sanitizer_fail" in analysis["patterns"]
        assert analysis["patterns"]["guardrail_block"] == 2

    def test_update_regression_dataset(self, tmp_dir):
        ds = LocalRegressionDataset(dataset_path=str(tmp_dir / "reg.json"))
        loop = SelfImprovementLoop(regression_dataset=ds)
        failures = [
            FailedRun("r1", {"q": "test"}, {}, ["ERR"], time.time()),
        ]
        added = loop.update_regression_dataset(failures)
        assert added == 1
        assert len(ds.get_entries()) == 1
