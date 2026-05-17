#!/usr/bin/env python3
"""
business_topic_insights_backend.py

Backend-ready script for turning a social-media posts JSON file into:
- adaptive topic/content clusters
- OpenAI-generated business labels
- KPI benchmarks per theme
- actionable recommendation cards
- JSON + CSV outputs for dashboards/APIs

Input:
    A JSON file containing either:
    1) a list of post records:
       [{"caption_text": "...", "likes_count": 10, ...}, ...]
    2) a dict with one of these keys:
       {"posts": [...]}, {"data": [...]}, {"items": [...]}, {"records": [...]}

Minimum required column:
    - caption_text OR caption OR text OR post_text
      You can override this using --text-col.

Useful optional columns:
    - engagement_rate
    - view_rate
    - likes_count
    - comments_count
    - views_count
    - shares_count
    - saves_count
    - hashtags_count
    - hashtags
    - post_type
    - language
    - CTA_present
    - promo_post
    - discount_percent
    - mentions_location
    - religious_theme
    - patriotic_theme
    - arabic_dialect_style

Install:
    pip install -U pandas numpy scikit-learn sentence-transformers bertopic umap-learn hdbscan openai python-dotenv

Run:
    export OPENAI_API_KEY="your_key_here"
    python business_topic_insights_backend.py --input uploaded_posts.json --output-dir outputs

Import in backend:
    from business_topic_insights_backend import analyze_business_json

    result = analyze_business_json(
        input_path="uploaded_posts.json",
        output_dir="outputs",
        text_col=None,                 # auto-detect
        require_openai=False,          # True = raise if OpenAI is unavailable
        include_posts=False,           # True = include all post-level rows in returned JSON
    )

Return shape:
    {
      "summary": {...},
      "topic_labels": [...],
      "topic_kpis": [...],
      "topic_recommendations": [...],
      "insight_cards": [...],
      "posts_with_topics_preview": [...],
      "files": {...},
      "fit_attempts": [...]
    }
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    SentenceTransformer = None
    SENTENCE_TRANSFORMER_IMPORT_ERROR = exc
else:
    SENTENCE_TRANSFORMER_IMPORT_ERROR = None

try:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
except Exception as exc:  # pragma: no cover
    BERTopic = None
    HDBSCAN = None
    UMAP = None
    BERTOPIC_IMPORT_ERROR = exc
else:
    BERTOPIC_IMPORT_ERROR = None


TEXT_CANDIDATES = ["caption_text", "caption", "text", "post_text", "content", "body"]

DEFAULT_OUTPUT_FILES = {
    "result_json": "business_topic_insights_result.json",
    "posts_with_topics_csv": "posts_with_business_topics.csv",
    "topic_labels_csv": "topic_business_labels.csv",
    "topic_kpis_csv": "topic_kpi_benchmarks.csv",
    "topic_recommendations_csv": "topic_recommendations.csv",
    "insight_cards_csv": "topic_business_insights.csv",
}

DEFAULT_VISUALIZATION_FILES = {
    "intertopic": "business_topic_intertopic_map.html",
    "barchart": "business_topic_barchart.html",
    "heatmap": "business_topic_heatmap.html",
}

_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_DIR: Optional[str] = None
_EMBEDDING_DEVICE: Optional[str] = None


@dataclass
class BusinessTopicConfig:
    """
    Configuration for adaptive business-topic insight generation.

    Most backend calls only need input_path and output_dir. The remaining values
    are safe defaults for Arabic/English SME social-media datasets.
    """

    # Input / output
    input_path: str
    output_dir: str = "business_topic_outputs"
    text_col: Optional[str] = None
    hashtags_col: str = "hashtags"
    include_posts: bool = False

    # Embeddings / clustering
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    random_state: int = 42
    min_docs_for_bertopic: int = 12
    max_fit_attempts: int = 8
    reduce_too_many_topics: bool = True

    # Vectorizer / terms
    min_df: Optional[int] = None
    token_pattern: str = r"(?u)\b[^\W\d_][^\W_]+\b"
    top_n_words: int = 12

    # OpenAI
    openai_model: str = "gpt-4.1-mini"
    openai_fallback_models: Tuple[str, ...] = ("gpt-4o-mini",)
    openai_max_retries: int = 3
    openai_timeout_seconds: float = 60.0
    require_openai: bool = False

    # Reporting
    preview_posts_limit: int = 100
    verbose: bool = True


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def load_env_files() -> None:
    """
    Load .env files without requiring python-dotenv.
    """
    if load_dotenv is not None:
        load_dotenv()
        load_dotenv("../.env")

    for env_path in [Path(".env"), Path("../.env")]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def make_openai_client(require_openai: bool = False):
    """
    Create an OpenAI client from OPENAI_API_KEY.

    If require_openai=True, raise an error when the client is unavailable.
    Otherwise return None and the pipeline will use deterministic fallbacks.
    """
    load_env_files()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if OpenAI is None:
        msg = "openai package is not installed. Run: pip install -U openai"
        if require_openai:
            raise RuntimeError(msg)
        return None

    if not api_key:
        msg = "OPENAI_API_KEY is not set."
        if require_openai:
            raise RuntimeError(msg)
        return None

    return OpenAI(api_key=api_key)


def robust_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from a model response.

    Handles normal JSON and occasional markdown-wrapped JSON.
    """
    if not text:
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Last-resort extraction of the largest JSON-looking object.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None

    return None


def openai_json_call(
    client,
    system_prompt: str,
    user_payload: Dict[str, Any],
    *,
    model: str,
    fallback_models: Sequence[str] = (),
    max_retries: int = 3,
    timeout_seconds: float = 60.0,
) -> Optional[Dict[str, Any]]:
    """
    Call OpenAI and force a JSON-object response.

    Retries across model names and transient failures. Returns None if all
    attempts fail, so the rest of the backend can still return useful output.
    """
    if client is None:
        return None

    models = [model] + [m for m in fallback_models if m != model]
    user_content = json.dumps(user_payload, ensure_ascii=False)

    for model_name in models:
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.2,
                    timeout=timeout_seconds,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                parsed = robust_json_loads(content)
                if parsed is not None:
                    return parsed
            except Exception:
                if attempt == max_retries:
                    break
                # Jitter helps when several backend requests hit the API at once.
                time.sleep(min(8.0, 0.8 * (2 ** (attempt - 1)) + random.random()))

    return None


def normalize_arabic_text(text: str) -> str:
    """
    Normalize common Arabic spelling variants and remove diacritics/tatweel.
    """
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ـ", "", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    return text


def parse_hashtag_tokens(value: Any) -> List[str]:
    """
    Parse hashtags whether the source value is:
    - a real list
    - a stringified list
    - comma-separated text
    - pipe-separated text
    - raw #hashtag text
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, list):
        raw = value
    else:
        s = str(value).strip()
        if not s:
            return []

        try:
            parsed = ast.literal_eval(s)
            raw = parsed if isinstance(parsed, list) else re.findall(r"(?<!\w)#([^\s#]+)", s)
        except Exception:
            if "|" in s:
                raw = s.split("|")
            elif "," in s:
                raw = s.split(",")
            else:
                raw = re.findall(r"(?<!\w)#([^\s#]+)", s)
                if not raw and s:
                    raw = [s]

    out: List[str] = []
    for item in raw:
        t = str(item).strip().lower().lstrip("#")
        t = normalize_arabic_text(t)
        t = re.sub(r"[^\w\u0600-\u06FF]+", "", t)
        if t:
            out.append(t)

    return list(dict.fromkeys(out))


def clean_caption(text: Any, hashtags_value: Any = None) -> str:
    """
    Clean social captions for mixed Arabic/English topic modeling.

    Important: explicit hashtag words are removed from the caption so topics are
    based more on caption language than repeated hashtag spam. Hashtags can still
    be used separately as features if present.
    """
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"\b\d{5,}\b", " ", text)

    text = normalize_arabic_text(text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

    for term in parse_hashtag_tokens(hashtags_value):
        text = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", " ", text)
        for part in term.split("_"):
            if len(part) >= 3:
                text = re.sub(rf"(?<!\w){re.escape(part)}(?!\w)", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def load_posts_dataframe(input_path: str) -> pd.DataFrame:
    """
    Load JSON or CSV into a DataFrame.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".json", ".jsonl"]:
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)

        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return pd.DataFrame(raw)

        if isinstance(raw, dict):
            for key in ["posts", "data", "items", "records", "rows"]:
                if isinstance(raw.get(key), list):
                    return pd.DataFrame(raw[key])

            # If it is a dict of columns, pandas can often load it directly.
            try:
                df = pd.DataFrame(raw)
                if not df.empty:
                    return df
            except Exception:
                pass

        # Fallback to pandas JSON inference.
        return pd.read_json(path)

    raise ValueError(f"Unsupported input type: {suffix}. Use .json, .jsonl, or .csv")


def detect_text_column(df: pd.DataFrame, explicit_text_col: Optional[str] = None) -> str:
    """
    Detect caption/text column if not explicitly provided.
    """
    if explicit_text_col:
        if explicit_text_col not in df.columns:
            raise KeyError(f"text_col='{explicit_text_col}' not found. Available columns: {list(df.columns)}")
        return explicit_text_col

    for col in TEXT_CANDIDATES:
        if col in df.columns:
            return col

    fallback = [c for c in df.columns if "caption" in c.lower() or "text" in c.lower()]
    if fallback:
        return fallback[0]

    raise KeyError(
        "Could not detect a text column. Expected one of "
        f"{TEXT_CANDIDATES}, or pass --text-col explicitly."
    )


def prepare_texts(df: pd.DataFrame, text_col: str, hashtags_col: str = "hashtags") -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean the input DataFrame and return modeling texts.
    """
    work = df.copy()
    if hashtags_col in work.columns:
        work["_clean_text"] = [
            clean_caption(t, h) for t, h in zip(work[text_col].fillna(""), work[hashtags_col])
        ]
    else:
        work["_clean_text"] = work[text_col].fillna("").apply(lambda x: clean_caption(x, None))

    work["_clean_text"] = work["_clean_text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Keep rows with meaningful text.
    work = work[work["_clean_text"].str.len() >= 3].copy()
    texts = work["_clean_text"].tolist()

    if not texts:
        raise ValueError("No usable captions after cleaning. Check the text column and input JSON.")

    return work.reset_index(drop=True), texts


def build_stop_words(extra_noise_words: Optional[Iterable[str]] = None) -> List[str]:
    """
    Arabic + English + social-commerce noise words.
    """
    arabic_stop_words = [
        "في", "من", "على", "الى", "إلى", "عن", "مع", "هذا", "هذه", "ذلك",
        "هو", "هي", "هم", "ما", "لا", "نعم", "او", "أو", "كل", "تم",
        "الله", "ان", "إن", "أن", "كان", "كانت", "بعد", "قبل", "الي",
        "لما", "اذا", "بس", "فيه", "فيها", "عند", "عنا", "لكل", "هاي",
    ]

    default_noise_words = [
        "2024", "2025", "2026",
        "whatsapp", "واتساب", "للطلب", "تواصل", "اتصال", "رقم",
        "available", "new", "today",
    ]

    words = set(arabic_stop_words).union(set(ENGLISH_STOP_WORDS)).union(default_noise_words)
    if extra_noise_words:
        words = words.union(set(extra_noise_words))
    return sorted(words)


def adaptive_min_df(n_docs: int, configured_min_df: Optional[int]) -> int:
    """
    Avoid impossible CountVectorizer settings on small uploads.
    """
    if configured_min_df is not None:
        return max(1, int(configured_min_df))
    if n_docs < 50:
        return 1
    return 2


def build_vectorizer(n_docs: int, config: BusinessTopicConfig, ngram_range: Tuple[int, int] = (1, 2)) -> CountVectorizer:
    """
    Create a vectorizer robust to small datasets.
    """
    return CountVectorizer(
        stop_words=build_stop_words(),
        ngram_range=ngram_range,
        min_df=adaptive_min_df(n_docs, config.min_df),
        token_pattern=config.token_pattern,
        max_features=5000,
    )


def get_topic_target_range(n_docs: int) -> Tuple[int, int]:
    """
    Decide a realistic target number of non-outlier clusters by dataset size.

    This prevents the two common production problems:
    - tiny uploads producing fake over-specific clusters
    - large uploads being collapsed into too few broad clusters
    """
    n_docs = max(int(n_docs), 1)

    if n_docs < 12:
        return 1, min(3, n_docs)
    if n_docs < 30:
        return 2, 4
    if n_docs < 75:
        return 3, 6
    if n_docs < 150:
        return 4, 9
    if n_docs < 300:
        return 5, 12
    if n_docs < 700:
        return 7, 18
    if n_docs < 1500:
        return 10, 25
    return 12, 35


def get_min_cluster_candidates(n_docs: int) -> List[int]:
    """
    Adaptive min_cluster_size candidates for BERTopic/HDBSCAN.

    Larger values produce fewer, safer clusters.
    Smaller values recover more granular clusters if the first attempts are too broad.
    """
    n_docs = max(int(n_docs), 1)

    if n_docs < 20:
        raw = [max(2, n_docs // 4), max(2, n_docs // 5), 2]
    else:
        raw = [
            max(2, int(n_docs * 0.16)),
            max(2, int(n_docs * 0.12)),
            max(2, int(n_docs * 0.09)),
            max(2, int(n_docs * 0.07)),
            max(2, int(n_docs * 0.05)),
            max(2, int(n_docs * 0.035)),
        ]

    out: List[int] = []
    for x in raw:
        x = min(max(2, int(x)), max(2, n_docs - 1))
        if x not in out:
            out.append(x)
    return out


def get_umap_neighbors_candidates(n_docs: int) -> List[int]:
    """
    Adaptive UMAP neighbors. Smaller uploads need small n_neighbors.
    """
    upper = max(2, min(n_docs - 1, 40))
    raw = [
        min(upper, max(2, int(math.sqrt(max(n_docs, 2))))),
        min(upper, 10),
        min(upper, 15),
        min(upper, 25),
    ]
    out: List[int] = []
    for x in raw:
        if x >= 2 and x not in out:
            out.append(x)
    return out


def load_embedding_model(config: BusinessTopicConfig):
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_DIR, _EMBEDDING_DEVICE

    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not available. "
            "Run: pip install -U sentence-transformers. "
            f"Original error: {SENTENCE_TRANSFORMER_IMPORT_ERROR}"
        )

    project_root = Path(__file__).resolve().parents[1]
    backend_root = project_root / "backend"
    model_dir = Path(
        os.getenv(
            "BERT_EMBEDDING_MODEL_DIR",
            str(backend_root / ".local_model" / "paraphrase-multilingual-MiniLM-L12-v2"),
        )
    ).resolve()
    cache_dir = Path(
        os.getenv(
            "BERT_EMBEDDING_CACHE_DIR",
            str(backend_root / ".local_model" / "_cache"),
        )
    ).resolve()
    local_only = os.getenv("BERT_LOCAL_ONLY", "0").strip().lower() in {"1", "true", "yes"}

    requested_device = os.getenv("BERT_EMBEDDING_DEVICE", "").strip().lower()
    if not requested_device:
        requested_device = "auto"
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if requested_device == "auto":
        device = "cuda" if cuda_available else "cpu"
    elif requested_device == "cuda" and not cuda_available:
        log("BERT_EMBEDDING_DEVICE=cuda requested but CUDA is unavailable; falling back to cpu.", True)
        device = "cpu"
    elif requested_device in {"cpu", "cuda"}:
        device = requested_device
    else:
        device = "cpu"

    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_dir_str = str(model_dir)

    if (
        _EMBEDDING_MODEL is not None
        and _EMBEDDING_MODEL_DIR == model_dir_str
        and _EMBEDDING_DEVICE == device
    ):
        return _EMBEDDING_MODEL

    model_files_exist = (model_dir / "config_sentence_transformers.json").exists() or (model_dir / "modules.json").exists()

    if model_files_exist:
        _EMBEDDING_MODEL = SentenceTransformer(str(model_dir), local_files_only=True, device=device)
        _EMBEDDING_MODEL_DIR = model_dir_str
        _EMBEDDING_DEVICE = device
        return _EMBEDDING_MODEL

    if local_only:
        raise RuntimeError(
            f"Local embedding model not found at {model_dir}. "
            "Set BERT_LOCAL_ONLY=0 for first download, then rerun."
        )

    # First run: download once from model hub, persist to local dir.
    downloaded = SentenceTransformer(config.embedding_model_name, cache_folder=str(cache_dir), device=device)
    downloaded.save(str(model_dir))

    # Always serve from local path after first download.
    _EMBEDDING_MODEL = SentenceTransformer(str(model_dir), local_files_only=True, device=device)
    _EMBEDDING_MODEL_DIR = model_dir_str
    _EMBEDDING_DEVICE = device
    return _EMBEDDING_MODEL


def count_non_outlier_topics_from_list(topics: Sequence[int]) -> int:
    return len({int(t) for t in topics if int(t) != -1})


def outlier_ratio(topics: Sequence[int]) -> float:
    if not topics:
        return 0.0
    return float(np.mean([int(t) == -1 for t in topics]))


def build_bertopic_model(
    *,
    n_docs: int,
    config: BusinessTopicConfig,
    embedding_model,
    min_cluster_size: int,
    n_neighbors: int,
):
    """
    Construct BERTopic with adaptive UMAP/HDBSCAN parameters.
    """
    if BERTopic is None or UMAP is None or HDBSCAN is None:
        raise RuntimeError(
            "BERTopic dependencies are unavailable. "
            "Run: pip install -U bertopic umap-learn hdbscan. "
            f"Original error: {BERTOPIC_IMPORT_ERROR}"
        )

    n_components = max(2, min(5, n_docs - 2)) if n_docs > 3 else 2
    min_samples = max(1, min(5, min_cluster_size // 3))

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=config.random_state,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    return BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=build_vectorizer(n_docs, config, ngram_range=(1, 2)),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="multilingual",
        min_topic_size=min_cluster_size,
        calculate_probabilities=True,
        verbose=False,
        top_n_words=config.top_n_words,
    )


def fit_bertopic_adaptive(
    texts: List[str],
    config: BusinessTopicConfig,
    embedding_model,
) -> Tuple[Any, List[int], Any, List[Dict[str, Any]]]:
    """
    Fit BERTopic with several adaptive attempts and select the best one.

    Selection prefers:
    1) topic count inside realistic target range
    2) low outlier ratio
    3) topic count close to target middle
    """
    n_docs = len(texts)
    min_topics, max_topics = get_topic_target_range(n_docs)
    target_mid = (min_topics + max_topics) / 2.0

    attempts: List[Dict[str, Any]] = []
    fitted: List[Tuple[float, Any, List[int], Any, Dict[str, Any]]] = []

    cluster_candidates = get_min_cluster_candidates(n_docs)
    neighbor_candidates = get_umap_neighbors_candidates(n_docs)

    log(f"Target non-outlier topics: {min_topics}-{max_topics}", config.verbose)
    log(f"min_cluster_size candidates: {cluster_candidates}", config.verbose)
    log(f"UMAP n_neighbors candidates: {neighbor_candidates}", config.verbose)

    tried = 0
    for min_cluster_size in cluster_candidates:
        for n_neighbors in neighbor_candidates:
            if tried >= config.max_fit_attempts:
                break
            tried += 1

            attempt_meta = {
                "method": "bertopic",
                "attempt": tried,
                "min_cluster_size": min_cluster_size,
                "n_neighbors": n_neighbors,
                "status": "started",
            }

            try:
                model = build_bertopic_model(
                    n_docs=n_docs,
                    config=config,
                    embedding_model=embedding_model,
                    min_cluster_size=min_cluster_size,
                    n_neighbors=n_neighbors,
                )
                topics_arr, probs = model.fit_transform(texts)
                topics = [int(t) for t in topics_arr]
                k = count_non_outlier_topics_from_list(topics)
                noise = outlier_ratio(topics)

                attempt_meta.update(
                    {
                        "status": "success",
                        "non_outlier_topics": k,
                        "outlier_ratio": round(noise, 4),
                    }
                )

                # Lower score is better.
                outside_penalty = 0 if min_topics <= k <= max_topics else 1
                distance_penalty = abs(k - target_mid) / max(target_mid, 1)
                score = outside_penalty * 10 + distance_penalty + noise * 1.5

                # Penalize degenerate one-cluster results when more data exists.
                if n_docs >= 30 and k < 2:
                    score += 5

                fitted.append((score, model, topics, probs, attempt_meta))
                attempts.append(attempt_meta)
                log(f"Attempt {tried}: topics={k}, outliers={noise:.2%}, score={score:.3f}", config.verbose)

                if min_topics <= k <= max_topics and noise <= 0.45:
                    return model, topics, probs, attempts

            except Exception as exc:
                attempt_meta.update(
                    {
                        "status": "failed",
                        "error": str(exc)[:500],
                    }
                )
                attempts.append(attempt_meta)
                log(f"Attempt {tried} failed: {exc}", config.verbose)

        if tried >= config.max_fit_attempts:
            break

    if not fitted:
        raise RuntimeError("All BERTopic attempts failed.")

    fitted.sort(key=lambda x: x[0])
    _, best_model, best_topics, best_probs, best_meta = fitted[0]

    best_k = count_non_outlier_topics_from_list(best_topics)
    if config.reduce_too_many_topics and best_k > max_topics:
        try:
            log(f"Reducing topics from {best_k} to about {max_topics}.", config.verbose)
            best_model.reduce_topics(texts, nr_topics=max_topics)
            try:
                new_topics, new_probs = best_model.transform(texts)
                best_topics = [int(t) for t in new_topics]
                best_probs = new_probs
            except Exception:
                best_topics = [int(t) for t in getattr(best_model, "topics_", best_topics)]
            attempts.append(
                {
                    "method": "bertopic_reduce_topics",
                    "status": "success",
                    "from_topics": best_k,
                    "to_topics": count_non_outlier_topics_from_list(best_topics),
                    "nr_topics": max_topics,
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "method": "bertopic_reduce_topics",
                    "status": "failed",
                    "error": str(exc)[:500],
                }
            )

    return best_model, best_topics, best_probs, attempts


def fallback_kmeans_topics(
    texts: List[str],
    config: BusinessTopicConfig,
    embedding_model,
) -> Tuple[None, List[int], Optional[np.ndarray], List[Dict[str, Any]]]:
    """
    Last-resort fallback when BERTopic is unavailable or fails.

    It still returns meaningful clusters using sentence-transformer embeddings +
    KMeans. Topic IDs are 0..k-1 and probabilities are approximated from distances.
    """
    n_docs = len(texts)
    min_topics, max_topics = get_topic_target_range(n_docs)
    k = int(round((min_topics + max_topics) / 2))
    k = max(1, min(k, n_docs))

    if n_docs < 3:
        return None, [0] * n_docs, None, [
            {"method": "kmeans_fallback", "status": "success", "k": 1, "reason": "tiny_dataset"}
        ]

    embeddings = embedding_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    kmeans = KMeans(n_clusters=k, random_state=config.random_state, n_init="auto")
    labels = kmeans.fit_predict(embeddings).astype(int)

    probs = None
    try:
        distances = kmeans.transform(embeddings)
        nearest = distances.min(axis=1)
        max_d = max(float(nearest.max()), 1e-9)
        approx_conf = 1 - (nearest / max_d)
        probs = approx_conf
    except Exception:
        pass

    sil = None
    try:
        if len(set(labels)) > 1 and len(set(labels)) < n_docs:
            sil = float(silhouette_score(embeddings, labels))
    except Exception:
        sil = None

    return None, labels.tolist(), probs, [
        {
            "method": "kmeans_fallback",
            "status": "success",
            "k": k,
            "silhouette_score": sil,
            "reason": "BERTopic unavailable or failed",
        }
    ]


def fit_topics_adaptive(
    texts: List[str],
    config: BusinessTopicConfig,
    embedding_model,
) -> Tuple[Any, List[int], Any, str, List[Dict[str, Any]]]:
    """
    Fit topic model with BERTopic first; fallback to KMeans.
    """
    n_docs = len(texts)

    if n_docs < config.min_docs_for_bertopic:
        model, topics, probs, attempts = fallback_kmeans_topics(texts, config, embedding_model)
        return model, topics, probs, "kmeans_fallback_tiny_dataset", attempts

    try:
        model, topics, probs, attempts = fit_bertopic_adaptive(texts, config, embedding_model)
        return model, topics, probs, "bertopic", attempts
    except Exception as exc:
        attempts = [
            {
                "method": "bertopic",
                "status": "failed_all",
                "error": str(exc)[:1000],
                "traceback": traceback.format_exc(limit=2),
            }
        ]
        log("BERTopic failed; using KMeans fallback.", config.verbose)
        model, topics, probs, fb_attempts = fallback_kmeans_topics(texts, config, embedding_model)
        return model, topics, probs, "kmeans_fallback_after_bertopic_failure", attempts + fb_attempts


def safe_probability_for_row(probs: Any, row_idx: int, topic_id: int) -> Optional[float]:
    """
    Extract a confidence-like value from BERTopic/KMeans probabilities.

    BERTopic probability arrays are not always indexed exactly by topic_id after
    reduction, so max probability is safer for dashboard confidence.
    """
    if probs is None:
        return None

    try:
        if isinstance(probs, (list, tuple)):
            arr = np.array(probs, dtype=object)
        else:
            arr = probs

        row = arr[row_idx]
        if np.isscalar(row):
            val = float(row)
        else:
            row_arr = np.array(row, dtype=float)
            if row_arr.size == 0:
                return None
            val = float(np.nanmax(row_arr))

        if math.isnan(val) or math.isinf(val):
            return None
        return max(0.0, min(1.0, val))
    except Exception:
        return None


def attach_topics_to_posts(df: pd.DataFrame, topics: List[int], probs: Any) -> pd.DataFrame:
    out = df.copy()
    out["topic_id"] = [int(t) for t in topics]
    out["topic_probability"] = [
        safe_probability_for_row(probs, i, int(t)) for i, t in enumerate(topics)
    ]
    return out


def extract_terms_from_assignments(
    texts: List[str],
    topics: List[int],
    config: BusinessTopicConfig,
    top_n: int = 12,
) -> Dict[int, List[str]]:
    """
    Extract top terms by assigned topic using c-TF-IDF-like scoring.
    Works for both BERTopic and KMeans fallback.
    """
    topic_texts = pd.DataFrame({"text": texts, "topic_id": topics})
    topic_texts = topic_texts[topic_texts["topic_id"] != -1].copy()

    if topic_texts.empty:
        return {}

    grouped = (
        topic_texts.groupby("topic_id")["text"]
        .apply(lambda s: " ".join(s.astype(str)))
        .reset_index()
        .sort_values("topic_id")
    )

    try:
        vectorizer = build_vectorizer(len(texts), config, ngram_range=(1, 2))
        X = vectorizer.fit_transform(grouped["text"])
        terms = np.array(vectorizer.get_feature_names_out())
    except Exception:
        return {int(t): [] for t in grouped["topic_id"].tolist()}

    tf = X.toarray().astype(float)
    topic_lengths = tf.sum(axis=1, keepdims=True)
    topic_lengths[topic_lengths == 0] = 1.0
    tf_norm = tf / topic_lengths

    df_term = (tf > 0).sum(axis=0)
    idf = np.log((1 + tf.shape[0]) / (1 + df_term)) + 1
    scores = tf_norm * idf

    out: Dict[int, List[str]] = {}
    for row_idx, topic_id in enumerate(grouped["topic_id"].tolist()):
        top_indices = np.argsort(scores[row_idx])[::-1][:top_n]
        out[int(topic_id)] = [
            str(terms[j])
            for j in top_indices
            if scores[row_idx, j] > 0 and str(terms[j]).strip()
        ]
    return out


def build_topic_summary(
    df_with_topics: pd.DataFrame,
    texts: List[str],
    topics: List[int],
    model: Any,
    config: BusinessTopicConfig,
    text_col: str,
) -> pd.DataFrame:
    """
    Build a topic summary with top terms and representative posts.
    """
    terms_by_assignment = extract_terms_from_assignments(texts, topics, config, top_n=config.top_n_words)

    rows: List[Dict[str, Any]] = []
    topic_counts = pd.Series(topics).value_counts().sort_index()

    for topic_id, count in topic_counts.items():
        topic_id = int(topic_id)
        if topic_id == -1:
            continue

        words: List[str] = []
        if model is not None:
            try:
                raw_terms = model.get_topic(topic_id) or []
                words = [str(w) for w, _ in raw_terms[: config.top_n_words] if str(w).strip()]
            except Exception:
                words = []

        if not words:
            words = terms_by_assignment.get(topic_id, [])

        sample_posts = (
            df_with_topics.loc[df_with_topics["topic_id"] == topic_id, text_col]
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )

        rows.append(
            {
                "topic_id": topic_id,
                "count": int(count),
                "top_words": ", ".join(words[: config.top_n_words]),
                "representative_docs_sample": " || ".join(sample_posts),
            }
        )

    outlier_count = int(topic_counts.get(-1, 0))
    if outlier_count:
        rows.append(
            {
                "topic_id": -1,
                "count": outlier_count,
                "top_words": "",
                "representative_docs_sample": "",
            }
        )

    return pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)


def fallback_business_label(row: pd.Series) -> Dict[str, str]:
    words = [w.strip() for w in str(row.get("top_words", "")).split(",") if w.strip()]
    topic_id = int(row.get("topic_id", -999))

    if topic_id == -1:
        return {
            "business_label": "Outlier / Mixed Content",
            "short_description": "Mixed or noisy posts that do not map cleanly to one strategy theme.",
            "content_strategy_meaning": "Review manually and avoid using this as a main content pillar.",
        }

    if words:
        label = " ".join(w.title() for w in words[:2])
    else:
        label = f"Theme Topic {topic_id}"

    return {
        "business_label": label[:70],
        "short_description": "Keyword-driven content theme detected from captions.",
        "content_strategy_meaning": "Test this theme with clearer creative, hook, and CTA; compare against KPI leaders.",
    }


def generate_topic_labels(
    topic_summary: pd.DataFrame,
    client,
    config: BusinessTopicConfig,
) -> pd.DataFrame:
    """
    Generate dashboard-friendly business labels with OpenAI, then fill gaps
    using deterministic fallback labels.
    """
    if topic_summary.empty:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "count",
                "top_words",
                "representative_docs_sample",
                "business_label",
                "short_description",
                "content_strategy_meaning",
            ]
        )

    base_records = topic_summary.to_dict(orient="records")
    non_outlier_records = [r for r in base_records if int(r.get("topic_id", -1)) != -1]

    labels_by_topic: Dict[int, Dict[str, str]] = {}

    if client is not None and non_outlier_records:
        system_prompt = (
            "You are a senior social-media marketing strategist. "
            "Convert raw topic words and sample captions into concise business-friendly content theme labels. "
            "Return JSON only with this shape: "
            "{'labels':[{'topic_id':0,'business_label':'','short_description':'','content_strategy_meaning':''}]}. "
            "Rules: business_label must be 2-5 words, dashboard-friendly, not just a keyword list. "
            "Consider Arabic and English context."
        )

        payload = {
            "topics": non_outlier_records,
            "label_rules": {
                "business_label": "2-5 words",
                "short_description": "one sentence",
                "content_strategy_meaning": "one actionable sentence",
            },
        }

        parsed = openai_json_call(
            client,
            system_prompt,
            payload,
            model=config.openai_model,
            fallback_models=config.openai_fallback_models,
            max_retries=config.openai_max_retries,
            timeout_seconds=config.openai_timeout_seconds,
        )

        if parsed and isinstance(parsed.get("labels"), list):
            for item in parsed["labels"]:
                try:
                    topic_id = int(item.get("topic_id"))
                    labels_by_topic[topic_id] = {
                        "business_label": str(item.get("business_label", "")).strip(),
                        "short_description": str(item.get("short_description", "")).strip(),
                        "content_strategy_meaning": str(item.get("content_strategy_meaning", "")).strip(),
                    }
                except Exception:
                    continue

    rows: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}

    for _, row in topic_summary.iterrows():
        topic_id = int(row["topic_id"])
        generated = labels_by_topic.get(topic_id) or fallback_business_label(row)

        # Guard against empty/invalid OpenAI output.
        if not generated.get("business_label"):
            generated = fallback_business_label(row)

        label = str(generated["business_label"]).strip()
        key = label.lower()
        duplicate_count = seen.get(key, 0)
        seen[key] = duplicate_count + 1
        if duplicate_count:
            label = f"{label} (Topic {topic_id})"

        rows.append(
            {
                **row.to_dict(),
                "business_label": label,
                "short_description": generated.get("short_description", ""),
                "content_strategy_meaning": generated.get("content_strategy_meaning", ""),
            }
        )

    return pd.DataFrame(rows)


def truthy_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip().isin(["1", "true", "yes", "y", "t"])


def numeric_if_present(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def get_metric_columns(df: pd.DataFrame) -> List[str]:
    """
    Dynamically detect useful performance/count/rate columns.
    """
    preferred = [
        "engagement_rate",
        "view_rate",
        "view_engagement_rate",
        "comments_to_likes_ratio",
        "likes_count",
        "comments_count",
        "views_count",
        "shares_count",
        "saves_count",
        "reach_count",
        "impressions_count",
        "hashtags_count",
        "caption_length",
        "emoji_count",
        "discount_percent",
    ]
    found = [c for c in preferred if c in df.columns]

    keywords = [
        "like", "comment", "view", "engagement", "rate", "share", "save",
        "reach", "impression", "click", "hashtag", "caption", "emoji",
    ]
    for col in df.columns:
        if col in found:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and any(k in col.lower() for k in keywords):
            found.append(col)

    return found


def aggregate_topic_kpis(df_with_topics: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Aggregate all available KPIs per business label.
    """
    work = df_with_topics.copy()

    metric_cols = get_metric_columns(work)
    for col in metric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    rows: List[Dict[str, Any]] = []
    group_cols = ["business_label"]

    for label, grp in work.groupby(group_cols[0], dropna=False):
        row: Dict[str, Any] = {
            "business_label": str(label),
            "post_count": int(len(grp)),
            "dominant_topic_id": int(grp["topic_id"].mode(dropna=True).iloc[0]) if grp["topic_id"].notna().any() else None,
            "topic_id_list": ", ".join(sorted({str(int(x)) for x in grp["topic_id"].dropna().tolist()})),
        }

        if "topic_probability" in grp.columns:
            row["avg_topic_probability"] = float(pd.to_numeric(grp["topic_probability"], errors="coerce").mean())

        for col in metric_cols:
            row[f"avg_{col}"] = float(grp[col].mean()) if grp[col].notna().any() else None
            row[f"median_{col}"] = float(grp[col].median()) if grp[col].notna().any() else None

        if "post_type" in grp.columns:
            post_type = grp["post_type"].astype(str).str.lower()
            row["percentage_reels"] = float((post_type == "reel").mean() * 100)
            row["percentage_images"] = float((post_type == "image").mean() * 100)
            row["percentage_carousels"] = float((post_type == "carousel").mean() * 100)
            row["dominant_post_type"] = post_type.mode().iloc[0] if not post_type.mode().empty else None

        for bool_col in [
            "CTA_present",
            "promo_post",
            "mentions_location",
            "religious_theme",
            "patriotic_theme",
            "arabic_dialect_style",
        ]:
            if bool_col in grp.columns:
                row[f"percentage_{bool_col}"] = float(truthy_series(grp[bool_col]).mean() * 100)

        sample_col = "caption_text" if "caption_text" in grp.columns else None
        if sample_col:
            row["representative_posts"] = " || ".join(grp[sample_col].dropna().astype(str).head(2).tolist())

        rows.append(row)

    kpi_df = pd.DataFrame(rows)

    ranking_candidates = [
        "avg_engagement_rate",
        "median_engagement_rate",
        "avg_view_rate",
        "median_view_rate",
        "avg_views_count",
        "median_views_count",
        "avg_likes_count",
        "avg_comments_count",
        "post_count",
    ]
    ranking_metric = next((c for c in ranking_candidates if c in kpi_df.columns), "post_count")

    kpi_df = kpi_df.sort_values(ranking_metric, ascending=False).reset_index(drop=True)
    return kpi_df, ranking_metric


def add_topic_scores(kpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add performance/support/opportunity scores for ranking recommendations.
    """
    if kpi_df.empty:
        return kpi_df

    out = kpi_df.copy()
    metric_candidates = [
        "avg_engagement_rate",
        "median_engagement_rate",
        "avg_view_rate",
        "median_view_rate",
        "avg_views_count",
        "avg_likes_count",
        "avg_comments_count",
    ]
    metric_cols = [c for c in metric_candidates if c in out.columns and pd.to_numeric(out[c], errors="coerce").notna().any()]

    if metric_cols:
        clean = out[metric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        if len(out) > 1:
            scaled = MinMaxScaler().fit_transform(clean)
            out["performance_score"] = scaled.mean(axis=1)
        else:
            out["performance_score"] = 1.0
        out["scored_metrics"] = ", ".join(metric_cols)
    else:
        out["performance_score"] = np.nan
        out["scored_metrics"] = ""

    max_posts = max(float(out["post_count"].max()), 1.0)
    out["support_score"] = out["post_count"] / max_posts

    if out["performance_score"].notna().any():
        out["opportunity_score"] = 0.7 * out["performance_score"].fillna(0) + 0.3 * out["support_score"]
    else:
        out["opportunity_score"] = out["support_score"]

    return out.sort_values("opportunity_score", ascending=False).reset_index(drop=True)


def generate_rule_based_recommendations(scored_kpis: pd.DataFrame) -> pd.DataFrame:
    """
    Theme-level recommendations that adapt to sample size and performance.
    """
    if scored_kpis.empty:
        return pd.DataFrame(columns=["business_label", "action", "reason", "suggested_next_step", "confidence"])

    df = scored_kpis.copy()
    perf_q75 = df["performance_score"].quantile(0.75) if df["performance_score"].notna().any() else np.nan
    perf_q25 = df["performance_score"].quantile(0.25) if df["performance_score"].notna().any() else np.nan
    support_median = df["post_count"].median()

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        posts = int(row["post_count"])
        perf = row.get("performance_score", np.nan)
        label = str(row["business_label"])

        if posts < 3:
            action = "Monitor"
            confidence = "Low"
            reason = "Very few posts, so the theme is not reliable enough for a strong decision."
            next_step = "Collect more examples before scaling or avoiding this theme."
        elif pd.notna(perf) and posts >= support_median and perf >= perf_q75:
            action = "Scale"
            confidence = "High" if posts >= max(5, support_median) else "Medium"
            reason = "Strong performance with enough repeated examples."
            next_step = "Create more posts using this theme, then A/B test hook, creative, and CTA variants."
        elif pd.notna(perf) and posts < support_median and perf >= perf_q75:
            action = "Test More"
            confidence = "Medium"
            reason = "Good performance but low support, so it may be promising but not proven."
            next_step = "Run 3-5 additional posts before treating it as a main content pillar."
        elif pd.notna(perf) and posts >= support_median and perf <= perf_q25:
            action = "Rework / Avoid Scaling"
            confidence = "Medium"
            reason = "Repeated theme but weaker performance than other themes."
            next_step = "Change the hook, visual format, offer framing, or CTA before increasing volume."
        else:
            action = "Maintain"
            confidence = "Medium"
            reason = "Average theme compared with the rest of the uploaded data."
            next_step = "Keep it in the mix but prioritize stronger opportunity-score themes."

        rows.append(
            {
                "business_label": label,
                "post_count": posts,
                "performance_score": None if pd.isna(perf) else float(perf),
                "opportunity_score": None if pd.isna(row.get("opportunity_score", np.nan)) else float(row["opportunity_score"]),
                "action": action,
                "reason": reason,
                "suggested_next_step": next_step,
                "confidence": confidence,
            }
        )

    return pd.DataFrame(rows)


def generate_rule_based_insights(scored_kpis: pd.DataFrame, ranking_metric: str) -> pd.DataFrame:
    """
    Dashboard insight cards.
    """
    cards: List[Dict[str, Any]] = []

    if scored_kpis.empty:
        return pd.DataFrame(columns=["insight_title", "finding", "evidence", "recommendation", "priority", "metric_used"])

    df = scored_kpis.copy()
    metric = ranking_metric if ranking_metric in df.columns else "post_count"
    top = df.sort_values(metric, ascending=False).iloc[0]
    bottom = df.sort_values(metric, ascending=True).iloc[0]

    cards.append(
        {
            "insight_title": "Top Performing Theme",
            "finding": f"{top['business_label']} is the strongest theme by {metric}.",
            "evidence": f"{metric}={top[metric]:.4f}" if pd.notna(top[metric]) and isinstance(top[metric], (int, float, np.number)) else f"{metric}={top[metric]}",
            "recommendation": "Scale this theme with consistent creative direction and test stronger CTA variants.",
            "priority": "High",
            "metric_used": metric,
        }
    )

    if len(df) > 1:
        cards.append(
            {
                "insight_title": "Weakest Theme",
                "finding": f"{bottom['business_label']} is the weakest theme by {metric}.",
                "evidence": f"{metric}={bottom[metric]:.4f}" if pd.notna(bottom[metric]) and isinstance(bottom[metric], (int, float, np.number)) else f"{metric}={bottom[metric]}",
                "recommendation": "Do not scale this theme until the hook, visual, format, or offer framing is improved.",
                "priority": "High",
                "metric_used": metric,
            }
        )

    try:
        if len(df) > 1 and float(bottom[metric]) != 0:
            ratio = float(top[metric]) / float(bottom[metric])
            if np.isfinite(ratio):
                cards.append(
                    {
                        "insight_title": "Theme Performance Gap",
                        "finding": f"The top theme outperforms the weakest theme by {ratio:.2f}x.",
                        "evidence": f"{top['business_label']} vs {bottom['business_label']}",
                        "recommendation": "Shift content volume toward proven themes while redesigning weaker themes.",
                        "priority": "Medium",
                        "metric_used": metric,
                    }
                )
    except Exception:
        pass

    if "percentage_CTA_present" in df.columns:
        cta_top = df.sort_values("percentage_CTA_present", ascending=False).iloc[0]
        cards.append(
            {
                "insight_title": "CTA Usage Pattern",
                "finding": f"{cta_top['business_label']} has the highest CTA share.",
                "evidence": f"CTA share={cta_top['percentage_CTA_present']:.1f}%",
                "recommendation": "Compare CTA-heavy posts against softer community captions to avoid over-selling.",
                "priority": "Medium",
                "metric_used": "percentage_CTA_present",
            }
        )

    if "percentage_reels" in df.columns and metric in df.columns:
        reel_sorted = df.sort_values("percentage_reels", ascending=False).iloc[0]
        cards.append(
            {
                "insight_title": "Format Signal",
                "finding": f"{reel_sorted['business_label']} uses reels the most among detected themes.",
                "evidence": f"Reel share={reel_sorted['percentage_reels']:.1f}%",
                "recommendation": "Check whether reel-heavy themes also rank high in performance before increasing reel production.",
                "priority": "Medium",
                "metric_used": "percentage_reels",
            }
        )

    return pd.DataFrame(cards)


def rewrite_insights_with_openai(
    cards_df: pd.DataFrame,
    topic_recommendations_df: pd.DataFrame,
    client,
    config: BusinessTopicConfig,
) -> pd.DataFrame:
    """
    Optional OpenAI rewrite to make cards more dashboard-ready.
    """
    if client is None or cards_df.empty:
        return cards_df

    system_prompt = (
        "You are a marketing analytics assistant. Rewrite insight cards into concise dashboard copy. "
        "Keep evidence factual; do not invent metrics. Return JSON only with key 'cards'. "
        "Each card must include: insight_title, finding, evidence, recommendation, priority, metric_used."
    )

    payload = {
        "insight_cards": cards_df.to_dict(orient="records"),
        "theme_recommendations": topic_recommendations_df.head(10).to_dict(orient="records"),
    }

    parsed = openai_json_call(
        client,
        system_prompt,
        payload,
        model=config.openai_model,
        fallback_models=config.openai_fallback_models,
        max_retries=config.openai_max_retries,
        timeout_seconds=config.openai_timeout_seconds,
    )

    if not parsed or not isinstance(parsed.get("cards"), list):
        return cards_df

    rewritten = pd.DataFrame(parsed["cards"])
    needed = ["insight_title", "finding", "evidence", "recommendation", "priority", "metric_used"]
    for col in needed:
        if col not in rewritten.columns:
            rewritten[col] = cards_df[col] if col in cards_df.columns else ""
    return rewritten[needed]


def add_business_labels_to_posts(df_with_topics: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    label_map = labels_df.set_index("topic_id")["business_label"].to_dict()
    desc_map = labels_df.set_index("topic_id")["short_description"].to_dict()

    out = df_with_topics.copy()
    out["business_label"] = out["topic_id"].map(label_map).fillna("Unknown Theme")
    out["topic_description"] = out["topic_id"].map(desc_map).fillna("")
    return out


def clean_json_value(value: Any) -> Any:
    """
    Make numpy/pandas values JSON-safe.
    """
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False:
        return None
    return value


def records_json_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    records = df.replace([np.inf, -np.inf], np.nan).to_dict(orient="records")
    return [{k: clean_json_value(v) for k, v in row.items()} for row in records]


def write_interactive_visualizations(topic_model: Any, output_dir: str) -> Dict[str, Any]:
    """
    Save BERTopic's interactive Plotly visualizations as standalone HTML files.

    KMeans fallback runs do not expose BERTopic visualization methods, so this
    returns a clear unavailable status instead of failing the full insight run.
    """
    if topic_model is None:
        return {
            "available": False,
            "reason": "BERTopic model was not available; the run used the KMeans fallback.",
            "files": {},
            "errors": [],
        }

    visualizers = {
        "intertopic": "visualize_topics",
        "barchart": "visualize_barchart",
        "heatmap": "visualize_heatmap",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files: Dict[str, str] = {}
    errors: List[Dict[str, str]] = []
    for name, method_name in visualizers.items():
        method = getattr(topic_model, method_name, None)
        if method is None:
            errors.append({"view": name, "error": f"{method_name} is not available on this model."})
            continue
        try:
            figure = method()
            path = out / DEFAULT_VISUALIZATION_FILES[name]
            figure.write_html(str(path), include_plotlyjs=True, full_html=True)
            files[name] = str(path)
        except Exception as exc:
            errors.append({"view": name, "error": str(exc)[:500]})

    return {
        "available": bool(files),
        "reason": None if files else "No BERTopic visualization could be generated for this run.",
        "files": files,
        "errors": errors,
    }


def write_outputs(
    *,
    output_dir: str,
    result: Dict[str, Any],
    posts_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    kpis_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    insights_df: pd.DataFrame,
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = {name: str(out / filename) for name, filename in DEFAULT_OUTPUT_FILES.items()}

    posts_df.to_csv(files["posts_with_topics_csv"], index=False)
    labels_df.to_csv(files["topic_labels_csv"], index=False)
    kpis_df.to_csv(files["topic_kpis_csv"], index=False)
    recommendations_df.to_csv(files["topic_recommendations_csv"], index=False)
    insights_df.to_csv(files["insight_cards_csv"], index=False)

    # Put files map into the JSON result before writing.
    result["files"] = files
    Path(files["result_json"]).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return files


def analyze_business_json(
    input_path: str,
    output_dir: str = "business_topic_outputs",
    text_col: Optional[str] = None,
    hashtags_col: str = "hashtags",
    openai_model: str = "gpt-4.1-mini",
    require_openai: bool = False,
    include_posts: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Main backend function.

    Give it a JSON/CSV path and it returns a dashboard-ready dict while also
    saving JSON/CSV outputs to output_dir.
    """
    config = BusinessTopicConfig(
        input_path=input_path,
        output_dir=output_dir,
        text_col=text_col,
        hashtags_col=hashtags_col,
        openai_model=openai_model,
        require_openai=require_openai,
        include_posts=include_posts,
        verbose=verbose,
    )

    random.seed(config.random_state)
    np.random.seed(config.random_state)

    client = make_openai_client(require_openai=config.require_openai)

    raw_df = load_posts_dataframe(config.input_path)
    detected_text_col = detect_text_column(raw_df, config.text_col)

    log(f"Loaded rows: {len(raw_df)}", config.verbose)
    log(f"Detected text column: {detected_text_col}", config.verbose)

    work_df, texts = prepare_texts(raw_df, detected_text_col, hashtags_col=config.hashtags_col)
    n_docs = len(texts)

    if n_docs == 0:
        raise ValueError("No valid posts to analyze after cleaning.")

    embedding_model = load_embedding_model(config)
    topic_model, topics, probs, fit_method, fit_attempts = fit_topics_adaptive(texts, config, embedding_model)

    posts_with_topics = attach_topics_to_posts(work_df, topics, probs)

    topic_summary = build_topic_summary(
        df_with_topics=posts_with_topics,
        texts=texts,
        topics=topics,
        model=topic_model,
        config=config,
        text_col=detected_text_col,
    )

    topic_labels = generate_topic_labels(topic_summary, client, config)
    posts_with_topics = add_business_labels_to_posts(posts_with_topics, topic_labels)

    topic_kpis, ranking_metric = aggregate_topic_kpis(posts_with_topics)
    scored_kpis = add_topic_scores(topic_kpis)

    topic_recommendations = generate_rule_based_recommendations(scored_kpis)
    insight_cards = generate_rule_based_insights(scored_kpis, ranking_metric)
    insight_cards = rewrite_insights_with_openai(insight_cards, topic_recommendations, client, config)
    visualizations = write_interactive_visualizations(topic_model, output_dir)

    min_topics, max_topics = get_topic_target_range(n_docs)
    actual_topics = count_non_outlier_topics_from_list(topics)
    outliers = int(sum(1 for t in topics if int(t) == -1))

    best_theme = None
    weakest_theme = None
    if not scored_kpis.empty:
        best_theme = str(scored_kpis.sort_values(ranking_metric if ranking_metric in scored_kpis.columns else "opportunity_score", ascending=False).iloc[0]["business_label"])
        if len(scored_kpis) > 1:
            weakest_theme = str(scored_kpis.sort_values(ranking_metric if ranking_metric in scored_kpis.columns else "opportunity_score", ascending=True).iloc[0]["business_label"])

    result: Dict[str, Any] = {
        "summary": {
            "input_path": str(input_path),
            "rows_loaded": int(len(raw_df)),
            "valid_posts_analyzed": int(n_docs),
            "text_col": detected_text_col,
            "fit_method": fit_method,
            "target_topic_range": {"min": min_topics, "max": max_topics},
            "actual_non_outlier_topics": int(actual_topics),
            "outlier_posts": outliers,
            "outlier_ratio": float(outliers / max(n_docs, 1)),
            "ranking_metric": ranking_metric,
            "best_theme": best_theme,
            "weakest_theme": weakest_theme,
            "openai_used": client is not None,
            "openai_model_requested": config.openai_model,
        },
        "topic_labels": records_json_safe(topic_labels),
        "topic_kpis": records_json_safe(scored_kpis),
        "topic_recommendations": records_json_safe(topic_recommendations),
        "insight_cards": records_json_safe(insight_cards),
        "posts_with_topics_preview": records_json_safe(posts_with_topics.head(config.preview_posts_limit)),
        "visualizations": visualizations,
        "fit_attempts": fit_attempts,
    }

    if include_posts:
        result["posts_with_topics"] = records_json_safe(posts_with_topics)

    files = write_outputs(
        output_dir=output_dir,
        result=result,
        posts_df=posts_with_topics,
        labels_df=topic_labels,
        kpis_df=scored_kpis,
        recommendations_df=topic_recommendations,
        insights_df=insight_cards,
    )
    result["files"] = files

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adaptive business topic insights from uploaded social-media JSON.")
    parser.add_argument("--input", required=True, help="Path to uploaded .json/.jsonl/.csv file.")
    parser.add_argument("--output-dir", default="business_topic_outputs", help="Directory to save JSON/CSV outputs.")
    parser.add_argument("--text-col", default=None, help="Optional text/caption column name. Auto-detected if omitted.")
    parser.add_argument("--hashtags-col", default="hashtags", help="Optional hashtags column name.")
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), help="OpenAI model for labels/insights.")
    parser.add_argument("--require-openai", action="store_true", help="Fail if OpenAI client/API key is unavailable.")
    parser.add_argument("--include-posts", action="store_true", help="Include all post-level rows in the returned JSON file.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_business_json(
        input_path=args.input,
        output_dir=args.output_dir,
        text_col=args.text_col,
        hashtags_col=args.hashtags_col,
        openai_model=args.openai_model,
        require_openai=args.require_openai,
        include_posts=args.include_posts,
        verbose=not args.quiet,
    )

    print(json.dumps({
        "summary": result["summary"],
        "files": result["files"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
