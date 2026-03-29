"""TUI 설정 영속 저장소.

HarnessConfig와 소스 플러그인 설정을 JSON 파일로 저장/로드한다.
기본 저장 경로: .harness_config.json (프로젝트 루트)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from harness.base import HarnessConfig

DEFAULT_CONFIG_PATH = ".harness_config.json"


@dataclass
class SourceConfig:
    """소스 플러그인 설정."""

    # 로컬 파일 소스
    local_enabled: bool = False
    local_directory: str = ""
    local_chunk_size: int = 1000
    local_chunk_overlap: int = 200

    # 웹 검색 소스 (agent-browser)
    web_search_enabled: bool = False
    web_search_max_results: int = 5
    web_search_page_timeout: int = 20


@dataclass
class AppConfig:
    """TUI 전체 설정 (영속)."""

    # HarnessConfig 필드
    max_iterations: int = 10
    max_tokens_budget: int = 100_000
    checkpoint_backend: str = "memory"
    checkpoint_dir: str = "checkpoints"
    guardrail_strict: bool = True
    human_in_the_loop: bool = False

    # 소스 설정
    sources: SourceConfig = field(default_factory=SourceConfig)

    # LM Studio 설정
    lm_studio_url: str = "http://169.254.83.107:1234/v1"
    lm_studio_model: str = "qwen/qwen3.5-9b"

    def to_harness_config(self) -> HarnessConfig:
        """HarnessConfig 인스턴스로 변환한다."""
        return HarnessConfig(
            max_iterations=self.max_iterations,
            max_tokens_budget=self.max_tokens_budget,
            checkpoint_backend=self.checkpoint_backend,
            checkpoint_dir=self.checkpoint_dir,
            guardrail_strict=self.guardrail_strict,
            human_in_the_loop=self.human_in_the_loop,
        )

    @classmethod
    def from_harness_config(cls, config: HarnessConfig) -> AppConfig:
        """HarnessConfig에서 생성한다. 나머지는 기본값."""
        return cls(
            max_iterations=config.max_iterations,
            max_tokens_budget=config.max_tokens_budget,
            checkpoint_backend=config.checkpoint_backend,
            checkpoint_dir=config.checkpoint_dir,
            guardrail_strict=config.guardrail_strict,
            human_in_the_loop=config.human_in_the_loop,
        )


def save_config(config: AppConfig, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    """설정을 JSON 파일로 저장한다."""
    data = asdict(config)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """JSON 파일에서 설정을 로드한다. 파일이 없으면 기본값을 반환한다."""
    p = Path(path)
    if not p.exists():
        return AppConfig()

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return AppConfig()

    # sources 중첩 객체 처리
    sources_data = data.pop("sources", {})
    sources = SourceConfig(**{
        k: v for k, v in sources_data.items()
        if k in SourceConfig.__dataclass_fields__
    })

    # 알 수 없는 키 무시
    known_keys = set(AppConfig.__dataclass_fields__.keys()) - {"sources"}
    filtered = {k: v for k, v in data.items() if k in known_keys}

    return AppConfig(sources=sources, **filtered)
