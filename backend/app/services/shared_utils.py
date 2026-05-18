from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOKS_SCRIPTS = PROJECT_ROOT / "notebooks" / "scripts"
GLOBAL_KPI_PATH = PROJECT_ROOT / "data" / "vanilla_kpi_dataset.json"
GLOBAL_OUTPUT_DIR = PROJECT_ROOT / "storage" / "outputs" / "business_momentum_single"


def load_module(module_name: str, file_path: Path) -> ModuleType:
    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_uploaded_dataset(uploaded_file_path: str) -> pd.DataFrame:
    p = Path(uploaded_file_path)
    if p.is_absolute() and p.exists():
        path = p
    else:
        candidates = [PROJECT_ROOT / uploaded_file_path, PROJECT_ROOT / "data" / uploaded_file_path]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Uploaded file not found: {uploaded_file_path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source_name}: {', '.join(missing)}")


def safe_identifier(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_").lower()
    return safe or "unknown"


def safe_output_directory(subpath: str) -> Path:
    out = (GLOBAL_OUTPUT_DIR / subpath).resolve()
    root = GLOBAL_OUTPUT_DIR.resolve()
    if root not in out.parents and out != root:
        raise ValueError("Unsafe output directory path.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_path_string(path: str | Path) -> str:
    return str(Path(path).resolve())

