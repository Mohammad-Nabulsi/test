from __future__ import annotations

import io
import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import settings
from app.services.hashtag_stage import HASHTAG_STAGE_SPEC, run_hashtag_stage
from app.services.kpi_stage import KPI_SPEC, engineer_kpis_from_notebook_logic
from app.services.preprocess_agnostic import PREPROCESS_SPEC, dataframe_to_records, preprocess_dataset
from app.services.topic_insights_stage import TOPIC_STAGE_SPEC, run_topic_insights_stage
from app.utils.file_utils import ensure_dir, write_csv, write_json


_KNOWN_DATASET_CACHE_KEY = "marketing_vanilla_kpi_dataset"
_CANONICAL_CACHE_UPLOAD_NAME = "vanilla.json"


def _storage(dataset_id: str) -> dict[str, Path]:
    base = settings.storage_path()
    return {
        "base": base,
        "raw": ensure_dir(base / "raw" / dataset_id),
        "cleaned": ensure_dir(base / "cleaned" / dataset_id),
        "outputs": ensure_dir(base / "outputs" / dataset_id),
        "reports": ensure_dir(base / "reports" / dataset_id),
    }


def _cache_dir() -> Path:
    return ensure_dir(settings.storage_path() / "cache" / _KNOWN_DATASET_CACHE_KEY)


def _is_canonical_dataset_upload_name(filename: str) -> bool:
    return Path(filename).name.strip().lower() == _CANONICAL_CACHE_UPLOAD_NAME


def _cache_manifest_path() -> Path:
    return _cache_dir() / "cache_manifest.json"


def _copy_relative_path(src_root: Path, dst_root: Path, rel_path: str) -> None:
    src = src_root / rel_path
    dst = dst_root / rel_path
    if not src.exists():
        return
    ensure_dir(dst.parent)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _persist_cached_stage_outputs(dataset_id: str, hashtag_info: dict[str, Any], topic_info: dict[str, Any]) -> None:
    storage = _storage(dataset_id)
    outputs = storage["outputs"]
    cache = _cache_dir()

    hashtag_files = list(hashtag_info.get("output_files", []))
    topic_files = list(topic_info.get("output_files", []))
    rel_files = sorted(set(hashtag_files + topic_files + ["hashtag_stage_response.json", "topic_stage_response.json"]))
    for rel in rel_files:
        _copy_relative_path(outputs, cache, rel)

    write_json(
        _cache_manifest_path(),
        {
            "canonical_upload_name": _CANONICAL_CACHE_UPLOAD_NAME,
            "hashtag_output_files": hashtag_files,
            "topic_output_files": topic_files,
            "saved_from_dataset_id": dataset_id,
        },
    )


def _load_cached_stage_outputs(dataset_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    manifest_path = _cache_manifest_path()
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("canonical_upload_name") != _CANONICAL_CACHE_UPLOAD_NAME:
        return None

    storage = _storage(dataset_id)
    outputs = storage["outputs"]
    cache = _cache_dir()

    hashtag_files = list(manifest.get("hashtag_output_files", []))
    topic_files = list(manifest.get("topic_output_files", []))
    rel_files = sorted(set(hashtag_files + topic_files + ["hashtag_stage_response.json", "topic_stage_response.json"]))

    for rel in rel_files:
        if not (cache / rel).exists():
            return None

    for rel in rel_files:
        _copy_relative_path(cache, outputs, rel)

    hashtag_response = json.loads((outputs / "hashtag_stage_response.json").read_text(encoding="utf-8"))
    topic_response = json.loads((outputs / "topic_stage_response.json").read_text(encoding="utf-8"))
    return (
        {"dataset_id": dataset_id, "hashtag_stage": hashtag_response, "output_files": hashtag_files},
        {"dataset_id": dataset_id, "topic_stage": topic_response, "output_files": topic_files},
    )


def _read_uploaded_bytes(filename: str, content: bytes) -> tuple[pd.DataFrame, str]:
    lower = filename.lower()
    if lower.endswith(".csv"):
        decoded = content.decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(decoded)), "csv"
    if lower.endswith(".json"):
        decoded = content.decode("utf-8", errors="replace")
        data = json.loads(decoded)
        if isinstance(data, list):
            return pd.DataFrame(data), "json"
        if isinstance(data, dict):
            if "records" in data and isinstance(data["records"], list):
                return pd.DataFrame(data["records"]), "json"
            if "data" in data and isinstance(data["data"], list):
                return pd.DataFrame(data["data"]), "json"
            return pd.DataFrame([data]), "json"
        raise ValueError("JSON must be an object or array of objects.")
    raise ValueError("Unsupported file type. Only .csv and .json are accepted.")


def _write_frames_as_csv(output_dir: Path, frames: dict[str, pd.DataFrame]) -> list[str]:
    written: list[str] = []
    for name, frame in frames.items():
        file_name = f"{name}.csv"
        write_csv(output_dir / file_name, frame)
        written.append(file_name)
    return written


def _frame_to_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return json.loads(out.to_json(orient="records", force_ascii=False))


def _raw_metadata_path(dataset_id: str) -> Path:
    return _storage(dataset_id)["raw"] / "raw_meta.json"


def _save_raw(df_type: str, filename: str, content: bytes, dataset_id: str) -> Path:
    storage = _storage(dataset_id)
    raw_path = storage["raw"] / f"raw.{df_type}"
    raw_path.write_bytes(content)
    write_json(
        _raw_metadata_path(dataset_id),
        {
            "dataset_id": dataset_id,
            "source_filename": filename,
            "file_type": df_type,
            "raw_path": str(raw_path),
        },
    )
    return raw_path


def _load_raw_dataframe(dataset_id: str) -> tuple[pd.DataFrame, str]:
    storage = _storage(dataset_id)
    candidates = list(storage["raw"].glob("raw.*"))
    if not candidates:
        raise FileNotFoundError(f"No raw upload found for dataset_id={dataset_id}")
    raw_path = candidates[0]
    file_type = raw_path.suffix.lower().lstrip(".")
    return _read_uploaded_bytes(raw_path.name, raw_path.read_bytes())[0], file_type


def _load_cleaned_dataframe(dataset_id: str) -> pd.DataFrame:
    storage = _storage(dataset_id)
    cleaned_path = storage["cleaned"] / "cleaned_dataset.json"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset artifact not found: {cleaned_path}")
    data = json.loads(cleaned_path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    return pd.DataFrame(records)


def _load_kpi_dataframe(dataset_id: str) -> pd.DataFrame:
    storage = _storage(dataset_id)
    kpi_path = storage["outputs"] / "kpi_dataset.json"
    if not kpi_path.exists():
        raise FileNotFoundError(f"KPI dataset artifact not found: {kpi_path}")
    data = json.loads(kpi_path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    return pd.DataFrame(records)


def _save_cleaned_artifacts(dataset_id: str, cleaned_df: pd.DataFrame, preprocess_report: dict[str, Any]) -> None:
    storage = _storage(dataset_id)
    write_csv(storage["cleaned"] / "cleaned_dataset.csv", cleaned_df)
    write_json(storage["cleaned"] / "cleaned_dataset.json", {"records": dataframe_to_records(cleaned_df)})
    write_json(storage["reports"] / "preprocess_report.json", preprocess_report)


def _save_kpi_artifacts(dataset_id: str, kpi_df: pd.DataFrame, kpi_meta: dict[str, Any]) -> None:
    storage = _storage(dataset_id)
    write_csv(storage["outputs"] / "kpi_dataset.csv", kpi_df)
    write_json(storage["outputs"] / "kpi_dataset.json", {"records": _frame_to_json(kpi_df)})
    write_json(storage["reports"] / "kpi_report.json", kpi_meta)


def _save_hashtag_artifacts(dataset_id: str, hashtag_response: dict[str, Any], hashtag_frames: dict[str, pd.DataFrame]) -> list[str]:
    storage = _storage(dataset_id)
    files = _write_frames_as_csv(storage["outputs"], hashtag_frames)
    write_json(storage["outputs"] / "hashtag_stage_response.json", hashtag_response)
    return files


def _save_topic_artifacts(dataset_id: str, topic_response: dict[str, Any], topic_frames: dict[str, pd.DataFrame]) -> list[str]:
    storage = _storage(dataset_id)
    files = _write_frames_as_csv(storage["outputs"], topic_frames)
    write_json(storage["outputs"] / "topic_stage_response.json", topic_response)
    visualization_files = topic_response.get("visualizations", {}).get("files", {})
    for path_value in visualization_files.values():
        try:
            path = Path(str(path_value)).resolve()
            files.append(str(path.relative_to(storage["outputs"].resolve())))
        except Exception:
            files.append(Path(str(path_value)).name)
    return files


def get_stage_specs() -> dict[str, Any]:
    return {
        "preprocess": {
            "input_columns": PREPROCESS_SPEC.required_inputs + PREPROCESS_SPEC.optional_inputs,
            "expected_outputs": PREPROCESS_SPEC.outputs,
        },
        "kpi_engineering": {
            "input_columns": KPI_SPEC.required_inputs + KPI_SPEC.optional_inputs,
            "expected_outputs": KPI_SPEC.outputs,
        },
        "hashtag_association_recommendations": {
            "input_columns": HASHTAG_STAGE_SPEC.required_inputs + HASHTAG_STAGE_SPEC.optional_inputs,
            "expected_outputs": HASHTAG_STAGE_SPEC.outputs,
        },
        "business_topic_insights_backend": {
            "input_columns": TOPIC_STAGE_SPEC.required_inputs + TOPIC_STAGE_SPEC.optional_inputs,
            "expected_outputs": TOPIC_STAGE_SPEC.outputs,
            "source_script": "notebooks/business_topic_insights_backend.py",
        },
    }


def upload_raw_only(filename: str, content: bytes) -> dict[str, Any]:
    dataset_id = str(uuid.uuid4())
    raw_df, file_type = _read_uploaded_bytes(filename, content)
    _save_raw(file_type, filename, content, dataset_id)
    write_json(_storage(dataset_id)["reports"] / "stage_specs.json", get_stage_specs())
    return {
        "dataset_id": dataset_id,
        "file_type": file_type,
        "rows_received": int(len(raw_df)),
        "next_stage": f"/api/datasets/{dataset_id}/stages/preprocess",
        "preview_columns": list(raw_df.columns),
    }


def run_preprocess_stage_for_dataset(dataset_id: str) -> dict[str, Any]:
    raw_df, file_type = _load_raw_dataframe(dataset_id)
    cleaned_df, preprocess_report = preprocess_dataset(raw_df)
    _save_cleaned_artifacts(dataset_id, cleaned_df, preprocess_report)
    return {
        "dataset_id": dataset_id,
        "file_type": file_type,
        "rows_received": int(len(raw_df)),
        "rows_cleaned": int(len(cleaned_df)),
        "preprocess_report": preprocess_report,
        "cleaned_json": dataframe_to_records(cleaned_df),
        "next_stage": f"/api/datasets/{dataset_id}/stages/kpi",
    }


def run_preprocess_stage_from_upload(filename: str, content: bytes) -> dict[str, Any]:
    upload_info = upload_raw_only(filename, content)
    return run_preprocess_stage_for_dataset(upload_info["dataset_id"])


def run_kpi_stage_for_dataset(dataset_id: str) -> dict[str, Any]:
    cleaned_df = _load_cleaned_dataframe(dataset_id)
    if cleaned_df.empty:
        raise ValueError("Cleaned dataset is empty; preprocess stage did not produce rows.")
    kpi_df, kpi_meta = engineer_kpis_from_notebook_logic(cleaned_df)
    _save_kpi_artifacts(dataset_id, kpi_df, kpi_meta)
    return {
        "dataset_id": dataset_id,
        "rows_kpi": int(len(kpi_df)),
        "kpi_report": kpi_meta,
        "kpi_json": _frame_to_json(kpi_df),
        "next_stage_hashtag": f"/api/datasets/{dataset_id}/stages/hashtag",
        "next_stage_topic": f"/api/datasets/{dataset_id}/stages/business-topic-insights",
    }


def run_kpi_stage_from_upload(filename: str, content: bytes) -> dict[str, Any]:
    dataset_id = str(uuid.uuid4())
    cleaned_df, file_type = _read_uploaded_bytes(filename, content)
    _save_cleaned_artifacts(
        dataset_id=dataset_id,
        cleaned_df=cleaned_df,
        preprocess_report={
            "note": "cleaned dataset was uploaded directly to KPI stage",
            "file_type": file_type,
            "rows_uploaded": int(len(cleaned_df)),
        },
    )
    write_json(_storage(dataset_id)["reports"] / "stage_specs.json", get_stage_specs())
    return run_kpi_stage_for_dataset(dataset_id)


def run_hashtag_stage_for_dataset(dataset_id: str) -> dict[str, Any]:
    kpi_df = _load_kpi_dataframe(dataset_id)
    if kpi_df.empty:
        raise ValueError("KPI dataset is empty; KPI stage did not produce rows.")
    hashtag_response, hashtag_frames = run_hashtag_stage(kpi_df)
    files = _save_hashtag_artifacts(dataset_id, hashtag_response, hashtag_frames)
    return {
        "dataset_id": dataset_id,
        "hashtag_stage": hashtag_response,
        "output_files": files,
    }


def run_hashtag_stage_from_upload(filename: str, content: bytes) -> dict[str, Any]:
    dataset_id = str(uuid.uuid4())
    kpi_df, file_type = _read_uploaded_bytes(filename, content)
    _save_kpi_artifacts(
        dataset_id=dataset_id,
        kpi_df=kpi_df,
        kpi_meta={
            "note": "kpi dataset was uploaded directly to hashtag stage",
            "file_type": file_type,
            "rows_uploaded": int(len(kpi_df)),
        },
    )
    write_json(_storage(dataset_id)["reports"] / "stage_specs.json", get_stage_specs())
    return run_hashtag_stage_for_dataset(dataset_id)


def run_topic_stage_for_dataset(dataset_id: str) -> dict[str, Any]:
    kpi_df = _load_kpi_dataframe(dataset_id)
    if kpi_df.empty:
        raise ValueError("KPI dataset is empty; KPI stage did not produce rows.")
    topic_response, topic_frames = run_topic_insights_stage(
        kpi_df,
        output_dir=str(_storage(dataset_id)["outputs"] / "business_topic_outputs"),
        include_posts=False,
    )
    files = _save_topic_artifacts(dataset_id, topic_response, topic_frames)
    return {
        "dataset_id": dataset_id,
        "topic_stage": topic_response,
        "output_files": files,
    }


def run_topic_stage_from_upload(filename: str, content: bytes) -> dict[str, Any]:
    dataset_id = str(uuid.uuid4())
    kpi_df, file_type = _read_uploaded_bytes(filename, content)
    _save_kpi_artifacts(
        dataset_id=dataset_id,
        kpi_df=kpi_df,
        kpi_meta={
            "note": "kpi dataset was uploaded directly to topic stage",
            "file_type": file_type,
            "rows_uploaded": int(len(kpi_df)),
        },
    )
    write_json(_storage(dataset_id)["reports"] / "stage_specs.json", get_stage_specs())

    if _is_canonical_dataset_upload_name(filename):
        cached = _load_cached_stage_outputs(dataset_id)
        if cached is not None:
            _, topic_info = cached
            topic_info["used_cached_stage_outputs"] = True
            return topic_info

    topic_info = run_topic_stage_for_dataset(dataset_id)
    if _is_canonical_dataset_upload_name(filename):
        _persist_cached_stage_outputs(
            dataset_id,
            {"dataset_id": dataset_id, "hashtag_stage": {}, "output_files": []},
            topic_info,
        )
    topic_info["used_cached_stage_outputs"] = False
    return topic_info


def run_staged_pipeline(filename: str, content: bytes) -> dict[str, Any]:
    upload_info = upload_raw_only(filename, content)
    dataset_id = upload_info["dataset_id"]
    preprocess_info = run_preprocess_stage_for_dataset(dataset_id)
    kpi_info = run_kpi_stage_for_dataset(dataset_id)

    used_cached_stage_outputs = False
    cache_payload = None
    if _is_canonical_dataset_upload_name(filename):
        cache_payload = _load_cached_stage_outputs(dataset_id)

    if cache_payload is not None:
        hashtag_info, topic_info = cache_payload
        used_cached_stage_outputs = True
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            hashtag_future = executor.submit(run_hashtag_stage_for_dataset, dataset_id)
            topic_future = executor.submit(run_topic_stage_for_dataset, dataset_id)
            hashtag_info = hashtag_future.result()
            topic_info = topic_future.result()
        if _is_canonical_dataset_upload_name(filename):
            _persist_cached_stage_outputs(dataset_id, hashtag_info, topic_info)

    storage = _storage(dataset_id)

    stage_specs = get_stage_specs()
    return {
        "dataset_id": dataset_id,
        "file_type": upload_info["file_type"],
        "rows_received": upload_info["rows_received"],
        "rows_cleaned": preprocess_info["rows_cleaned"],
        "rows_kpi": kpi_info["rows_kpi"],
        "cleaned_json": preprocess_info["cleaned_json"],
        "kpi_json": kpi_info["kpi_json"],
        "hashtag_stage": hashtag_info["hashtag_stage"],
        "topic_stage": topic_info["topic_stage"],
        "used_cached_stage_outputs": used_cached_stage_outputs,
        "storage_paths": {
            "raw_dir": str(storage["raw"]),
            "cleaned_dir": str(storage["cleaned"]),
            "outputs_dir": str(storage["outputs"]),
            "reports_dir": str(storage["reports"]),
            "hashtag_output_files": hashtag_info["output_files"],
            "topic_output_files": topic_info["output_files"],
        },
        "stage_specs": stage_specs,
    }


def run_preprocess_kpi_pipeline(filename: str, content: bytes) -> dict[str, Any]:
    upload_info = upload_raw_only(filename, content)
    dataset_id = upload_info["dataset_id"]
    preprocess_info = run_preprocess_stage_for_dataset(dataset_id)
    kpi_info = run_kpi_stage_for_dataset(dataset_id)
    storage = _storage(dataset_id)
    return {
        "dataset_id": dataset_id,
        "file_type": upload_info["file_type"],
        "rows_received": upload_info["rows_received"],
        "rows_cleaned": preprocess_info["rows_cleaned"],
        "rows_kpi": kpi_info["rows_kpi"],
        "cleaned_json": preprocess_info["cleaned_json"],
        "kpi_json": kpi_info["kpi_json"],
        "storage_paths": {
            "raw_dir": str(storage["raw"]),
            "cleaned_dir": str(storage["cleaned"]),
            "outputs_dir": str(storage["outputs"]),
            "reports_dir": str(storage["reports"]),
        },
        "stage_specs": get_stage_specs(),
    }


def load_stage_json(dataset_id: str, name: str) -> dict[str, Any]:
    storage = _storage(dataset_id)
    path_map = {
        "cleaned": storage["cleaned"] / "cleaned_dataset.json",
        "kpi": storage["outputs"] / "kpi_dataset.json",
        "hashtag": storage["outputs"] / "hashtag_stage_response.json",
        "topic": storage["outputs"] / "topic_stage_response.json",
        "preprocess_report": storage["reports"] / "preprocess_report.json",
        "kpi_report": storage["reports"] / "kpi_report.json",
        "stage_specs": storage["reports"] / "stage_specs.json",
    }
    if name not in path_map:
        raise KeyError(f"Unknown stage artifact: {name}")
    target = path_map[name]
    if not target.exists():
        raise FileNotFoundError(f"Artifact not found: {target}")
    return json.loads(target.read_text(encoding="utf-8"))
