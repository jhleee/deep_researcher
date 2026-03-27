"""체크포인터 팩토리.

설정에 따라 적절한 체크포인터 인스턴스를 생성한다.
로컬 개발 시에는 MemorySaver 또는 JSON 파일 기반 체크포인터를 사용한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from harness.base import HarnessConfig


class JsonFileCheckpointer:
    """JSON 파일 기반 체크포인터.

    외부 DB 없이 로컬 파일시스템에 체크포인트를 저장/복원한다.
    프로덕션이 아닌 로컬 개발·테스트 용도.
    """

    def __init__(self, directory: str = "checkpoints"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, thread_id: str) -> Path:
        safe_name = thread_id.replace("/", "_").replace("\\", "_")
        return self.directory / f"{safe_name}.json"

    def save(self, thread_id: str, state: dict[str, Any]) -> None:
        """상태를 JSON 파일로 저장한다."""
        path = self._path_for(thread_id)
        serializable = _make_serializable(state)
        path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, thread_id: str) -> dict[str, Any] | None:
        """JSON 파일에서 상태를 복원한다. 없으면 None."""
        path = self._path_for(thread_id)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return json.loads(text)

    def list_threads(self) -> list[str]:
        """저장된 모든 thread_id 목록을 반환한다."""
        return [p.stem for p in self.directory.glob("*.json")]

    def delete(self, thread_id: str) -> bool:
        """체크포인트 파일을 삭제한다."""
        path = self._path_for(thread_id)
        if path.exists():
            path.unlink()
            return True
        return False


def _make_serializable(obj: Any) -> Any:
    """중첩 객체를 JSON 직렬화 가능한 형태로 변환한다."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {"__type__": type(obj).__name__, **_make_serializable(obj.__dict__)}
    return obj


def get_checkpointer(config: HarnessConfig) -> MemorySaver | JsonFileCheckpointer:
    """설정에 따른 체크포인터 인스턴스 생성.

    로컬 개발 시 지원 백엔드:
        - "memory": MemorySaver (인메모리, 프로세스 종료 시 소멸)
        - "json_file": JsonFileCheckpointer (파일시스템 영속)

    프로덕션 백엔드 (별도 의존성 설치 필요):
        - "sqlite": AsyncSqliteSaver
        - "postgres": AsyncPostgresSaver
    """
    backend = config.checkpoint_backend

    if backend == "memory":
        return MemorySaver()

    if backend == "json_file":
        return JsonFileCheckpointer(directory=config.checkpoint_dir)

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            return AsyncSqliteSaver.from_conn_string(
                os.path.join(config.checkpoint_dir, "harness.db")
            )
        except ImportError as e:
            raise ImportError(
                "sqlite 체크포인터를 사용하려면 langgraph-checkpoint-sqlite를 설치하세요."
            ) from e

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            conn_str = os.environ.get(
                "HARNESS_POSTGRES_URL",
                "postgresql://user:pass@localhost:5432/harness_db",
            )
            return AsyncPostgresSaver.from_conn_string(conn_str)
        except ImportError as e:
            raise ImportError(
                "postgres 체크포인터를 사용하려면 langgraph-checkpoint-postgres를 설치하세요."
            ) from e

    raise ValueError(f"지원하지 않는 체크포인터 백엔드: {backend}")
