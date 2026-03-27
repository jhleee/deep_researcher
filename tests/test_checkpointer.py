"""checkpointer.py 유닛 테스트."""

import pytest

from harness.base import HarnessConfig
from harness.checkpointer import JsonFileCheckpointer, get_checkpointer

from langgraph.checkpoint.memory import MemorySaver


class TestJsonFileCheckpointer:
    """JsonFileCheckpointer 테스트."""

    def test_save_and_load(self, json_checkpointer: JsonFileCheckpointer):
        state = {"messages": ["hello"], "count": 42}
        json_checkpointer.save("thread-1", state)
        loaded = json_checkpointer.load("thread-1")
        assert loaded == state

    def test_load_nonexistent(self, json_checkpointer: JsonFileCheckpointer):
        result = json_checkpointer.load("nonexistent")
        assert result is None

    def test_list_threads(self, json_checkpointer: JsonFileCheckpointer):
        json_checkpointer.save("thread-a", {"data": 1})
        json_checkpointer.save("thread-b", {"data": 2})
        threads = json_checkpointer.list_threads()
        assert set(threads) == {"thread-a", "thread-b"}

    def test_list_threads_empty(self, json_checkpointer: JsonFileCheckpointer):
        assert json_checkpointer.list_threads() == []

    def test_delete(self, json_checkpointer: JsonFileCheckpointer):
        json_checkpointer.save("thread-del", {"data": 1})
        assert json_checkpointer.delete("thread-del") is True
        assert json_checkpointer.load("thread-del") is None

    def test_delete_nonexistent(self, json_checkpointer: JsonFileCheckpointer):
        assert json_checkpointer.delete("nope") is False

    def test_overwrite(self, json_checkpointer: JsonFileCheckpointer):
        json_checkpointer.save("thread-1", {"v": 1})
        json_checkpointer.save("thread-1", {"v": 2})
        loaded = json_checkpointer.load("thread-1")
        assert loaded == {"v": 2}

    def test_complex_state(self, json_checkpointer: JsonFileCheckpointer):
        state = {
            "messages": [{"role": "user", "content": "hello"}],
            "nested": {"a": [1, 2, 3], "b": {"c": True}},
        }
        json_checkpointer.save("complex", state)
        loaded = json_checkpointer.load("complex")
        assert loaded == state

    def test_safe_thread_id(self, json_checkpointer: JsonFileCheckpointer):
        """슬래시가 포함된 thread_id 안전 처리."""
        json_checkpointer.save("user/session/1", {"data": 1})
        loaded = json_checkpointer.load("user/session/1")
        assert loaded == {"data": 1}


class TestGetCheckpointer:
    """get_checkpointer 팩토리 테스트."""

    def test_memory_backend(self):
        config = HarnessConfig(checkpoint_backend="memory")
        cp = get_checkpointer(config)
        assert isinstance(cp, MemorySaver)

    def test_json_file_backend(self, tmp_dir):
        config = HarnessConfig(
            checkpoint_backend="json_file",
            checkpoint_dir=str(tmp_dir / "cp"),
        )
        cp = get_checkpointer(config)
        assert isinstance(cp, JsonFileCheckpointer)

    def test_unsupported_backend(self):
        config = HarnessConfig(checkpoint_backend="redis")
        with pytest.raises(ValueError, match="지원하지 않는"):
            get_checkpointer(config)
