from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def _time_bucket(hour: float) -> str:
    try:
        h = int(hour)
    except Exception:
        return "unknown"
    if 5 <= h <= 11:
        return "morning"
    if 12 <= h <= 16:
        return "afternoon"
    if 17 <= h <= 21:
        return "evening"
    return "night"


def build_transactions(kpis_df: pd.DataFrame) -> pd.DataFrame:
    df = kpis_df.copy()
    items = []
    for _, r in df.iterrows():
        row_items = [
            f"sector={str(r.get('sector','unknown'))}",
            f"post_type={str(r.get('post_type','unknown'))}",
            f"language={str(r.get('language','unknown'))}",
            f"CTA={'yes' if bool(r.get('CTA_present', False)) else 'no'}",
            f"promo={'yes' if bool(r.get('promo_post', False)) else 'no'}",
            f"religious_theme={'yes' if bool(r.get('religious_theme', False)) else 'no'}",
            f"patriotic_theme={'yes' if bool(r.get('patriotic_theme', False)) else 'no'}",
            f"dialect={'yes' if bool(r.get('arabic_dialect_style', False)) else 'no'}",
            f"time_bucket={_time_bucket(r.get('posting_hour', np.nan))}",
        ]
        if bool(r.get("high_engagement", False)):
            row_items.append("result=high_engagement")
        if bool(r.get("viral_post", False)):
            row_items.append("result=viral_post")
        if bool(r.get("high_comment_post", False)):
            row_items.append("result=high_comment_post")
        items.append({"transaction_id": int(len(items)), "items": "|".join(sorted(set(row_items)))})
    return pd.DataFrame(items)


def mine_association_rules(transactions_df: pd.DataFrame, min_support: float = 0.03) -> Tuple[pd.DataFrame, Dict]:
    if transactions_df.empty:
        return pd.DataFrame(), {"ok": False, "message": "No transactions."}

    # One-hot encode items
    all_items = sorted(
        set(
            item
            for s in transactions_df["items"].astype(str)
            for item in s.split("|")
            if item.strip()
        )
    )
    if len(all_items) < 5:
        return pd.DataFrame(), {"ok": False, "message": "Too few unique items for rules."}

    matrix = []
    for s in transactions_df["items"].astype(str):
        set_items = set(s.split("|"))
        matrix.append([1 if it in set_items else 0 for it in all_items])
    ohe = pd.DataFrame(matrix, columns=all_items).astype(bool)

    try:
        freq = apriori(ohe, min_support=min_support, use_colnames=True)
        if freq.empty:
            return pd.DataFrame(), {"ok": False, "message": "No frequent itemsets at the current support threshold."}
        rules = association_rules(freq, metric="confidence", min_threshold=0.35)
        if rules.empty:
            return pd.DataFrame(), {"ok": False, "message": "No rules found at the current confidence threshold."}
        rules = rules.sort_values(["lift", "confidence"], ascending=False).copy()

        # Flatten frozensets
        rules["antecedents"] = rules["antecedents"].apply(lambda x: "|".join(sorted(list(x))))
        rules["consequents"] = rules["consequents"].apply(lambda x: "|".join(sorted(list(x))))
        # Add business value score heuristic: reward result labels and lift
        rules["has_result"] = rules["consequents"].str.contains("result=")
        rules["business_value_score"] = (
            10 * rules["confidence"].astype(float)
            + 6 * rules["lift"].astype(float)
            + 2 * rules["support"].astype(float)
            + 4 * rules["has_result"].astype(int)
        )
        keep_cols = [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
            "business_value_score",
        ]
        out = rules[keep_cols].reset_index(drop=True)
        return out, {"ok": True, "items": len(all_items), "rules": int(len(out))}
    except Exception as e:
        return pd.DataFrame(), {"ok": False, "message": f"Rule mining failed: {e}"}


def business_value_rules(rules_df: pd.DataFrame) -> pd.DataFrame:
    if rules_df.empty:
        return rules_df
    out = rules_df.copy()
    # Simple post-processing: prefer actionable antecedents (CTA/promo/dialect)
    actionable = out["antecedents"].str.contains("CTA=|promo=|dialect=|post_type=|language=")
    out["business_value_score"] = out["business_value_score"] + actionable.astype(int) * 1.5
    return out.sort_values("business_value_score", ascending=False).reset_index(drop=True)

