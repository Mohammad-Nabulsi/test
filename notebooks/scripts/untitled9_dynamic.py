from __future__ import annotations

import argparse
import ast
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

warnings.filterwarnings("ignore")


@dataclass
class TopicPipelineConfig:
    data_path: str
    text_col: str = "caption_text"
    hashtags_col: str = "hashtags"

    local_model_dir: str = "./.local_models/sentence_transformers/all-MiniLM-L6-v2"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    ngram_sizes: tuple[int, ...] = (1, 2, 3)
    base_ngram_size: int = 1
    min_df: int = 2
    token_pattern: str = r"(?u)\b[^\W\d_][^\W_]+\b"
    top_n_words: int = 20

    n_neighbors: int = 10
    n_components: int = 5
    min_dist: float = 0.0
    umap_metric: str = "cosine"
    min_cluster_size: int = 8
    min_samples: int = 2
    hdbscan_metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    min_topic_size: int = 8
    calculate_probabilities: bool = True
    random_state: int = 42
    language: str = "multilingual"
    verbose: bool = True

    min_topic_posts_for_strong_claim: int = 8
    sample_docs_per_topic: int = 2
    recommendation_top_k: int = 3


def load_dataset(config: TopicPipelineConfig) -> pd.DataFrame:
    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Use JSON/CSV/TXT.")


def normalize_arabic_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ـ", "", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    return text


def _parse_hashtag_tokens(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        raw = val
    else:
        s = str(val).strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            raw = parsed if isinstance(parsed, list) else re.findall(r"(?<!\w)#([^\s#]+)", s, flags=re.UNICODE)
        except Exception:
            if "|" in s:
                raw = s.split("|")
            elif "," in s:
                raw = s.split(",")
            else:
                raw = re.findall(r"(?<!\w)#([^\s#]+)", s, flags=re.UNICODE)
    out = []
    for x in raw:
        t = str(x).strip().lower().lstrip("#")
        t = normalize_arabic_text(t)
        t = re.sub(r"[^\w\u0600-\u06FF]+", "", t)
        if t:
            out.append(t)
    return list(dict.fromkeys(out))


def clean_caption(text, hashtags_value=None):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"\b\d{5,}\b", " ", text)
    text = normalize_arabic_text(text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    tag_terms = _parse_hashtag_tokens(hashtags_value)
    for term in tag_terms:
        text = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", " ", text)
        for part in term.split("_"):
            if len(part) >= 3:
                text = re.sub(rf"(?<!\w){re.escape(part)}(?!\w)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_texts(df: pd.DataFrame, config: TopicPipelineConfig) -> tuple[pd.DataFrame, list[str]]:
    if config.text_col not in df.columns:
        raise KeyError(f"Missing text column '{config.text_col}'. Available: {list(df.columns)}")
    df = df.copy()
    raw_text = df[config.text_col].fillna("")
    if config.hashtags_col in df.columns:
        df["clean_caption"] = [clean_caption(t, h) for t, h in zip(raw_text, df[config.hashtags_col])]
    else:
        df["clean_caption"] = raw_text.apply(lambda x: clean_caption(x, None))
    texts = df["clean_caption"].fillna("").astype(str).tolist()
    return df, texts


def build_stop_words(extra_noise_words=None):
    arabic_stop_words = [
        "في", "من", "على", "الى", "إلى", "عن", "مع", "هذا", "هذه", "ذلك",
        "هو", "هي", "هم", "ما", "لا", "نعم", "او", "أو", "كل", "تم",
        "الله", "ان", "إن", "أن", "كان", "كانت", "بعد", "قبل",
    ]
    default_noise_words = ["2024", "2025", "2026", "gaza"]
    if extra_noise_words is None:
        extra_noise_words = default_noise_words
    return list(set(arabic_stop_words).union(ENGLISH_STOP_WORDS).union(extra_noise_words))


def build_vectorizer(config: TopicPipelineConfig, ngram_size: int = 1, extra_noise_words=None) -> CountVectorizer:
    return CountVectorizer(
        stop_words=build_stop_words(extra_noise_words=extra_noise_words),
        ngram_range=(ngram_size, ngram_size),
        min_df=config.min_df,
        token_pattern=config.token_pattern,
    )


def load_embedding_model(config: TopicPipelineConfig) -> SentenceTransformer:
    local_dir = Path(config.local_model_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if local_dir.exists() and any(local_dir.iterdir()):
        model = SentenceTransformer(str(local_dir), local_files_only=True)
    else:
        model = SentenceTransformer(config.embedding_model_name)
        model.save(str(local_dir))
    return model


def build_topic_model(config: TopicPipelineConfig, embedding_model, ngram_size: int) -> BERTopic:
    vectorizer_model = build_vectorizer(config, ngram_size=ngram_size)
    umap_model = UMAP(
        n_neighbors=config.n_neighbors,
        n_components=config.n_components,
        min_dist=config.min_dist,
        metric=config.umap_metric,
        random_state=config.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric=config.hdbscan_metric,
        cluster_selection_method=config.cluster_selection_method,
        prediction_data=True,
    )
    return BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language=config.language,
        min_topic_size=config.min_topic_size,
        calculate_probabilities=config.calculate_probabilities,
        verbose=config.verbose,
        top_n_words=config.top_n_words,
    )


def fit_topic_model(texts: list[str], config: TopicPipelineConfig, embedding_model, ngram_size: int):
    topic_model = build_topic_model(config, embedding_model, ngram_size=ngram_size)
    topics, probs = topic_model.fit_transform(texts)
    topic_info = topic_model.get_topic_info()
    return topic_model, topics, probs, topic_info


def extract_exact_ngram_terms_by_topic(
    texts: list[str], topics: list[int], config: TopicPipelineConfig, ngram_sizes=(1, 2, 3), top_n: int = 15
) -> pd.DataFrame:
    df_topic_texts = pd.DataFrame({"text": texts, "topic_id": topics})
    df_topic_texts = df_topic_texts[df_topic_texts["topic_id"] != -1].copy()
    grouped = (
        df_topic_texts.groupby("topic_id")["text"].apply(lambda s: " ".join(s.astype(str))).reset_index().sort_values("topic_id")
    )
    rows = []
    for n in ngram_sizes:
        vectorizer = build_vectorizer(config, ngram_size=n)
        X = vectorizer.fit_transform(grouped["text"])
        terms = np.array(vectorizer.get_feature_names_out())
        tf = X.toarray().astype(float)
        topic_lengths = tf.sum(axis=1, keepdims=True)
        topic_lengths[topic_lengths == 0] = 1
        tf_norm = tf / topic_lengths
        df_term = (tf > 0).sum(axis=0)
        idf = np.log((1 + tf.shape[0]) / (1 + df_term)) + 1
        scores = tf_norm * idf
        for row_idx, topic_id in enumerate(grouped["topic_id"].tolist()):
            if scores.shape[1] == 0:
                continue
            top_indices = np.argsort(scores[row_idx])[::-1][:top_n]
            for rank, term_idx in enumerate(top_indices, start=1):
                if scores[row_idx, term_idx] <= 0:
                    continue
                rows.append(
                    {
                        "topic_id": topic_id,
                        "ngram_size": n,
                        "rank": rank,
                        "term": terms[term_idx],
                        "score": float(scores[row_idx, term_idx]),
                    }
                )
    return pd.DataFrame(rows)


def pivot_top_ngram_terms(ngram_terms: pd.DataFrame, top_n_per_ngram: int = 8) -> pd.DataFrame:
    if ngram_terms.empty:
        return pd.DataFrame()
    filtered = ngram_terms[ngram_terms["rank"] <= top_n_per_ngram].copy()
    return (
        filtered.sort_values(["topic_id", "ngram_size", "rank"])
        .groupby(["topic_id", "ngram_size"])["term"]
        .apply(lambda s: ", ".join(s.astype(str)))
        .unstack("ngram_size")
        .reset_index()
        .rename(columns={1: "top_1grams", 2: "top_2grams", 3: "top_3grams"})
    )


def attach_topics(df: pd.DataFrame, topics: list[int], topic_model: BERTopic) -> pd.DataFrame:
    out = df.copy()
    out["topic_id"] = topics
    topic_info = topic_model.get_topic_info()
    return out.merge(topic_info[["Topic", "Name"]], left_on="topic_id", right_on="Topic", how="left").drop(columns=["Topic"])


def detect_numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    candidate_keywords = [
        "like", "comment", "view", "engagement", "rate", "share", "save", "reach", "impression", "click", "caption_length",
        "hashtag", "emoji",
    ]
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and any(k in col.lower() for k in candidate_keywords):
            numeric_cols.append(col)
    return numeric_cols


def summarize_topic_performance(df_with_topics: pd.DataFrame) -> pd.DataFrame:
    df_topics = df_with_topics[df_with_topics["topic_id"] != -1].copy()
    if df_topics.empty:
        return pd.DataFrame()
    numeric_cols = detect_numeric_metric_columns(df_topics)
    agg_spec = {"posts": ("topic_id", "size")}
    for col in numeric_cols:
        agg_spec[f"median_{col}"] = (col, "median")
        agg_spec[f"mean_{col}"] = (col, "mean")
    return df_topics.groupby("topic_id").agg(**agg_spec).reset_index().sort_values("posts", ascending=False)


def choose_primary_performance_metrics(topic_summary: pd.DataFrame) -> list[str]:
    preferred_order = [
        "median_engagement_rate", "mean_engagement_rate", "median_view_rate", "mean_view_rate", "median_likes_count",
        "median_comments_count", "median_views_count", "mean_likes_count", "mean_comments_count", "mean_views_count",
    ]
    selected = [c for c in preferred_order if c in topic_summary.columns]
    if selected:
        return selected[:4]
    return [c for c in topic_summary.columns if c != "posts" and (c.startswith("median_") or c.startswith("mean_"))][:4]


def add_dynamic_topic_scores(topic_summary: pd.DataFrame) -> pd.DataFrame:
    if topic_summary.empty:
        return topic_summary
    out = topic_summary.copy()
    metric_cols = choose_primary_performance_metrics(out)
    if not metric_cols:
        out["performance_score"] = np.nan
        out["support_score"] = out["posts"] / max(out["posts"].max(), 1)
        out["opportunity_score"] = out["support_score"]
        return out
    clean_metrics = out[metric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = MinMaxScaler().fit_transform(clean_metrics)
    out["performance_score"] = scaled.mean(axis=1)
    out["support_score"] = out["posts"] / max(out["posts"].max(), 1)
    out["opportunity_score"] = 0.7 * out["performance_score"] + 0.3 * out["support_score"]
    out["scored_metrics"] = ", ".join(metric_cols)
    return out.sort_values("opportunity_score", ascending=False)


def make_dynamic_recommendations(scored_summary: pd.DataFrame, ngram_pivot: pd.DataFrame, config: TopicPipelineConfig) -> pd.DataFrame:
    if scored_summary.empty:
        return pd.DataFrame(columns=["topic_id", "action", "reason", "suggested_next_step"])
    df = scored_summary.copy()
    if not ngram_pivot.empty:
        df = df.merge(ngram_pivot, on="topic_id", how="left")
    perf_q75 = df["performance_score"].quantile(0.75)
    perf_q25 = df["performance_score"].quantile(0.25)
    support_median = df["posts"].median()
    recs = []
    for _, row in df.iterrows():
        posts = row["posts"]
        performance = row.get("performance_score", np.nan)
        terms = row.get("top_2grams", None) or row.get("top_1grams", None) or ""
        terms = str(terms)[:180]
        if posts >= support_median and performance >= perf_q75:
            action, reason = "Scale", "This topic has both solid support and above-average performance."
            next_step = f"Create more content around repeated themes: {terms}"
        elif posts < support_median and performance >= perf_q75:
            action, reason = "Test more", "This topic performs well but has fewer posts."
            next_step = f"Run 3-5 new posts using related phrases: {terms}"
        elif posts >= support_median and performance <= perf_q25:
            action, reason = "Rework", "This topic appears often but underperforms."
            next_step = "Change hook/creative/CTA/posting time before scaling."
        elif posts < config.min_topic_posts_for_strong_claim:
            action, reason = "Monitor", "Topic is small; avoid strong conclusions."
            next_step = "Collect more examples before making major decisions."
        else:
            action, reason = "Maintain", "Topic is around the middle of performance."
            next_step = "Keep in mix and prioritize higher opportunity topics."
        recs.append(
            {
                "topic_id": row["topic_id"],
                "posts": posts,
                "performance_score": performance,
                "opportunity_score": row.get("opportunity_score", np.nan),
                "action": action,
                "reason": reason,
                "suggested_next_step": next_step,
            }
        )
    return pd.DataFrame(recs).sort_values("opportunity_score", ascending=False)


def export_outputs(
    output_dir: Path,
    df_with_topics: pd.DataFrame,
    topic_summary: pd.DataFrame,
    ngram_terms: pd.DataFrame,
    recommendations: pd.DataFrame,
    topic_model: BERTopic,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    df_with_topics.to_csv(output_dir / "dataset_with_topics.csv", index=False)
    topic_summary.to_csv(output_dir / "topic_summary.csv", index=False)
    ngram_terms.to_csv(output_dir / "topic_terms_exact_1_2_3grams.csv", index=False)
    recommendations.to_csv(output_dir / "dynamic_recommendations.csv", index=False)

    info = topic_model.get_topic_info()
    non_outliers = info[info["Topic"] != -1]["Topic"].tolist()
    if len(non_outliers) >= 3:
        topic_model.visualize_topics(topics=non_outliers).write_html(str(output_dir / "intertopic_map.html"))
    topic_model.visualize_barchart(
        topics=non_outliers if non_outliers else None, top_n_topics=max(1, min(10, len(non_outliers) or 1))
    ).write_html(str(output_dir / "topic_barchart.html"))
    if len(non_outliers) >= 2:
        topic_model.visualize_hierarchy().write_html(str(output_dir / "topic_hierarchy.html"))
        topic_model.visualize_heatmap().write_html(str(output_dir / "topic_heatmap.html"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run Untitled9 dynamic topic pipeline in one command.")
    parser.add_argument("--dataset-path", required=True, help="Path to input dataset (.json/.csv/.txt).")
    parser.add_argument("--text-col", default="caption_text", help="Text column name.")
    parser.add_argument("--hashtags-col", default="hashtags", help="Hashtags column name.")
    parser.add_argument("--output-dir", default="notebooks/topic_outputs", help="Directory for outputs.")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--min-topic-size", type=int, default=8)
    parser.add_argument("--top-n-words", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Disable BERTopic verbose logs.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TopicPipelineConfig(
        data_path=args.dataset_path,
        text_col=args.text_col,
        hashtags_col=args.hashtags_col,
        min_df=args.min_df,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        random_state=args.random_state,
        verbose=not args.quiet,
    )

    df = load_dataset(config)
    df, texts = prepare_texts(df, config)
    print(f"Loaded rows={len(df)} from {Path(config.data_path).resolve()}")

    embedding_model = load_embedding_model(config)
    base_topic_model, base_topics, _, base_info = fit_topic_model(
        texts=texts, config=config, embedding_model=embedding_model, ngram_size=config.base_ngram_size
    )
    print(f"Fitted base model with {len(base_info[base_info['Topic'] != -1])} non-outlier topics.")

    exact_ngram_terms = extract_exact_ngram_terms_by_topic(
        texts=texts, topics=base_topics, config=config, ngram_sizes=config.ngram_sizes, top_n=15
    )
    topic_ngram_labels = pivot_top_ngram_terms(exact_ngram_terms, top_n_per_ngram=8)
    df_with_topics = attach_topics(df, base_topics, base_topic_model)
    scored_topic_summary = add_dynamic_topic_scores(summarize_topic_performance(df_with_topics))
    scored_topic_summary_with_terms = scored_topic_summary.merge(topic_ngram_labels, on="topic_id", how="left")
    recommendations = make_dynamic_recommendations(scored_topic_summary, topic_ngram_labels, config)

    export_outputs(
        output_dir=Path(args.output_dir),
        df_with_topics=df_with_topics,
        topic_summary=scored_topic_summary_with_terms,
        ngram_terms=exact_ngram_terms,
        recommendations=recommendations,
        topic_model=base_topic_model,
    )
    print(f"Saved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

