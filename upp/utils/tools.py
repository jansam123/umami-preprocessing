from __future__ import annotations

from pathlib import Path


def path_append(path: Path | str, suffix: str):
    path = Path(path)
    return path.parent / f"{path.stem}_{suffix}{path.suffix}"
