from __future__ import annotations

from pathlib import Path
from typing import Iterable

from evocast.domain.knowledge_paths import package_root


def configs_dir() -> Path:
    return package_root() / "configs"


def candidate_config_paths(config_root: Path, value: str | Path) -> list[Path]:
    raw = Path(value)
    if raw.is_absolute():
        return [raw]
    text = raw.as_posix()
    return [config_root / text]


def resolve_config_path(
    value: str | Path,
    *,
    config_root: Path | None = None,
    fallback_paths: Iterable[Path] = (),
) -> Path:
    root = config_root or configs_dir()
    for path in [*candidate_config_paths(root, value), *fallback_paths]:
        if path.exists():
            return path
    candidates = candidate_config_paths(root, value)
    return candidates[0] if candidates else root / str(value)
