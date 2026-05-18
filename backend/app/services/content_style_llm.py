from __future__ import annotations

import json
from typing import Any, Dict

from app.config import settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


def generate_content_style_recommendation(
    business_name: str,
    low_cluster: Dict[str, Any],
    high_cluster: Dict[str, Any],
) -> str:
    """
    Generate a recommendation using OpenAI based on low/high cluster behavior.
    Raises RuntimeError when OpenAI setup is unavailable.
    """
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")

    client = OpenAI(api_key=settings.openai_api_key)

    payload = {
        "business_name": business_name,
        "instruction": (
            "Follow high-cluster behaviors and avoid low-cluster behaviors. "
            "Use the behavior averages and performance averages."
        ),
        "low_cluster": low_cluster,
        "high_cluster": high_cluster,
    }

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": settings.openai_recommendation_system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Create a recommendation for this business using the provided cluster analysis data.\n"
                    f"{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=0.2,
    )

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty recommendation.")
    return text


def generate_similar_business_recommendations_llm(
    business_name: str,
    target: Dict[str, Any],
    peers: list[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> list[Dict[str, Any]]:
    """
    Generate business recommendations using OpenAI LLM.
    Returns a list of recommendation dicts with fields: recommendation, reason,
    your_value, successful_peer_value, evidence_metric, priority_score, priority,
    comparison_businesses.
    """
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")

    client = OpenAI(api_key=settings.openai_api_key)

    payload = {
        "business_name": business_name,
        "target": target,
        "peers": peers,
        "metadata": metadata,
    }

    prompt = (
        "Generate a JSON array of practical, actionable recommendations for the business. "
        "Use the target profile and list of nearest successful peers. "
        "Return only valid JSON (an array) with objects containing: "
        "\"recommendation\", \"reason\", \"your_value\", \"successful_peer_value\", "
        "\"evidence_metric\", \"priority_score\" (0-1), \"priority\" (High/Medium/Low), "
        "\"comparison_businesses\".\n\n"
        "Focus on metrics where peers outperform the target by >15%. "
        "Each recommendation should be concrete and actionable.\n\n"
        f"Data:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": settings.openai_recommendation_system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned no recommendations.")

    try:
        recs = json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"OpenAI returned invalid JSON: {exc}\n{text}")

    if not isinstance(recs, list):
        raise RuntimeError("OpenAI must return a JSON array of recommendations.")

    return recs


def generate_content_style_ai_insights(
    business_name: str,
    low_cluster: Dict[str, Any],
    high_cluster: Dict[str, Any],
    all_clusters: list[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Generate AI-powered meaning and recommendation for low/high clusters.
    Returns {'low_meaning': ..., 'high_meaning': ..., 'recommendation': ...}.
    """
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")

    client = OpenAI(api_key=settings.openai_api_key)

    prompt = (
        "You are analyzing social media content clusters for a Palestinian SME business.\n\n"
        f"Business: {business_name}\n\n"
        "=== LOW-PERFORMING CLUSTER (the worst) ===\n"
        f"{json.dumps(low_cluster, ensure_ascii=False, default=str)}\n\n"
        "=== HIGH-PERFORMING CLUSTER (the best) ===\n"
        f"{json.dumps(high_cluster, ensure_ascii=False, default=str)}\n\n"
        "Return ONLY a valid JSON object with these three fields, no markdown, no extra text:\n"
        "{\n"
        '  "low_cluster_meaning": "<one-sentence label describing the low cluster>",\n'
        '  "high_cluster_meaning": "<one-sentence label describing the high cluster>",\n'
        '  "recommendation": "<tell the business to DO MORE of what the '
        'high-performing cluster does and AVOID what the low-performing cluster does. '
        'Reference metrics, post types, CTA usage, promo ratio, dialect, sector patterns. '
        'Keep under 300 words.>"\n'
        "}"
    )

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    text = (response.choices[0].message.content or "").strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"OpenAI returned invalid JSON: {exc}\n{text}")

    return {
        "low_meaning": result.get("low_cluster_meaning", f"Cluster {low_cluster.get('cluster', '?')}"),
        "high_meaning": result.get("high_cluster_meaning", f"Cluster {high_cluster.get('cluster', '?')}"),
        "recommendation": result.get("recommendation", "No recommendation generated."),
    }

