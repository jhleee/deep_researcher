"""config_store 유닛 테스트."""

from __future__ import annotations

import json

import pytest

from harness.base import HarnessConfig
from harness.config_store import AppConfig, SourceConfig, load_config, save_config


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.max_iterations == 10
        assert cfg.sources.local_enabled is False
        assert cfg.sources.web_search_max_results == 5

    def test_to_harness_config(self):
        cfg = AppConfig(max_iterations=20, guardrail_strict=False)
        hc = cfg.to_harness_config()
        assert isinstance(hc, HarnessConfig)
        assert hc.max_iterations == 20
        assert hc.guardrail_strict is False

    def test_from_harness_config(self):
        hc = HarnessConfig(max_iterations=15, checkpoint_backend="json_file")
        cfg = AppConfig.from_harness_config(hc)
        assert cfg.max_iterations == 15
        assert cfg.checkpoint_backend == "json_file"
        assert cfg.sources.local_enabled is False  # 기본값


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "config.json"
        cfg = AppConfig(
            max_iterations=5,
            lm_studio_model="test-model",
            sources=SourceConfig(
                local_enabled=True,
                local_directory="/data/sources",
                web_search_enabled=True,
                web_search_max_results=3,
            ),
        )
        save_config(cfg, path)
        loaded = load_config(path)

        assert loaded.max_iterations == 5
        assert loaded.lm_studio_model == "test-model"
        assert loaded.sources.local_enabled is True
        assert loaded.sources.local_directory == "/data/sources"
        assert loaded.sources.web_search_enabled is True
        assert loaded.sources.web_search_max_results == 3

    def test_load_missing_file(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.json")
        assert cfg.max_iterations == 10  # 기본값

    def test_load_corrupted_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json{{{", encoding="utf-8")
        cfg = load_config(path)
        assert cfg.max_iterations == 10  # 기본값

    def test_load_ignores_unknown_keys(self, tmp_path):
        path = tmp_path / "config.json"
        data = {"max_iterations": 7, "unknown_field": "ignored", "sources": {}}
        path.write_text(json.dumps(data), encoding="utf-8")
        cfg = load_config(path)
        assert cfg.max_iterations == 7

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "new_config.json"
        assert not path.exists()
        save_config(AppConfig(), path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "max_iterations" in data
        assert "sources" in data

    def test_load_partial_sources(self, tmp_path):
        """sources에 일부 키만 있어도 로딩 가능."""
        path = tmp_path / "config.json"
        data = {
            "max_iterations": 3,
            "sources": {"local_enabled": True},
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        cfg = load_config(path)
        assert cfg.sources.local_enabled is True
        assert cfg.sources.web_search_max_results == 5  # 기본값
