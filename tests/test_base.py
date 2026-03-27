"""base.py 유닛 테스트."""

import pytest

from harness.base import HarnessConfig, HarnessState, create_initial_state


class TestHarnessConfig:
    """HarnessConfig 테스트."""

    def test_default_values(self):
        config = HarnessConfig()
        assert config.max_iterations == 10
        assert config.max_tokens_budget == 100_000
        assert config.checkpoint_backend == "memory"
        assert config.tracing_enabled is False
        assert config.human_in_the_loop is False
        assert config.guardrail_strict is True

    def test_custom_values(self):
        config = HarnessConfig(
            max_iterations=20,
            checkpoint_backend="json_file",
            guardrail_strict=False,
        )
        assert config.max_iterations == 20
        assert config.checkpoint_backend == "json_file"
        assert config.guardrail_strict is False


class TestCreateInitialState:
    """create_initial_state 테스트."""

    def test_returns_dict(self):
        state = create_initial_state()
        assert isinstance(state, dict)

    def test_has_all_keys(self):
        state = create_initial_state()
        expected_keys = {"messages", "plan", "artifacts", "metadata", "error_log", "iteration_count"}
        assert set(state.keys()) == expected_keys

    def test_empty_lists(self):
        state = create_initial_state()
        assert state["messages"] == []
        assert state["plan"] == []
        assert state["artifacts"] == []
        assert state["error_log"] == []

    def test_iteration_count_zero(self):
        state = create_initial_state()
        assert state["iteration_count"] == 0

    def test_metadata_empty_dict(self):
        state = create_initial_state()
        assert state["metadata"] == {}
