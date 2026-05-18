from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_read_csv(path: Path) -> pd.DataFrame:
    # Keep it forgiving for student projects: try utf-8 then latin-1
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def list_output_files(outputs_dir: Path) -> list[str]:
    if not outputs_dir.exists():
        return []
    return sorted([p.name for p in outputs_dir.iterdir() if p.is_file()])


def resolve_output_file(outputs_dir: Path, file_name: str) -> Optional[Path]:
    p = (outputs_dir / file_name).resolve()
    try:
        p.relative_to(outputs_dir.resolve())
    except Exception:
        return None
    if not p.exists() or not p.is_file():
        return None
    return p

