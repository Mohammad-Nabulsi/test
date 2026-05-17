"""
association_rules_positive_module.py

This module implements Positive Association Rule Mining
for social media marketing analytics.

The system discovers patterns that are strongly associated
with HIGH engagement posts using the Apriori algorithm.

Main pipeline:
- Compute weighted engagement
- Normalize engagement rate
- Create high engagement target
- Perform feature engineering
- Generate transactions
- Extract positive association rules
"""

import pandas as pd
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def compute_engagement(df):
    """
    Computes weighted engagement and normalized engagement rate.

    Formula:
    engagement = likes + 2*comments + 0.1*views

    engagement_rate = engagement / followers_count
    """

    df["engagement"] = (
            df["likes_count"]
            + (df["comments_count"] * 2)
            + (df["views_count"] * 0.1)
    )

    df["followers_count_safe"] = (
        df["followers_count"].replace(0, np.nan)
    )

    df["engagement_rate"] = (
            df["engagement"] / df["followers_count_safe"]
    )

    df["engagement_rate"] = (
        df["engagement_rate"].fillna(0)
    )

    df = df.drop(columns=["followers_count_safe"])

    return df


def create_positive_target(df):
    """
    Creates binary target variable for high engagement posts.

    High engagement is defined as the top 25%
    of engagement rate values.
    """

    threshold = df["engagement_rate"].quantile(0.75)

    df["high_engagement"] = (
            df["engagement_rate"] >= threshold
    )

    return df


def feature_engineering(df):
    """
    Converts numerical features into categorical groups
    suitable for association rule mining.
    """

    df["time_group"] = df["posting_hour"].apply(
        lambda x:
        "morning" if x < 12
        else ("afternoon" if x < 18 else "evening")
    )

    df["caption_group"] = pd.cut(
        df["caption_length"],
        bins=[0, 50, 100, 1000],
        labels=["short", "medium", "long"]
    )

    df["hashtags_group"] = pd.cut(
        df["hashtags_count"],
        bins=[0, 5, 15, 100],
        labels=["low", "medium", "high"]
    )

    df["discount_group"] = df["discount_percent"].apply(
        lambda x:
        "none" if x == 0
        else ("low" if x <= 25 else "high")
    )

    return df


def clean_and_select_features(df):
    """
    Removes unnecessary and redundant columns.
    """

    drop_cols = [

        # Text columns
        "business_name",
        "caption_text",
        "post_date",

        # Raw numerical columns
        "posting_hour",
        "caption_length",
        "hashtags_count",
        "discount_percent",

        # Internal calculated columns
        "engagement",
        "engagement_rate"
    ]

    df = df.drop(
        columns=drop_cols,
        errors="ignore"
    )

    return df


def prepare_transactions(df):
    """
    Converts dataset rows into transaction format.

    Each row becomes a list of:
    feature=value items.
    """

    transactions = []

    for _, row in df.iterrows():

        items = []

        for col in df.columns:

            val = row[col]

            if pd.notna(val):
                items.append(f"{col}={val}")

        transactions.append(items)

    return transactions


def run_positive_apriori(
        transactions,
        min_support=0.05,
        min_confidence=0.6
):
    """
    Runs Apriori Association Rule Mining
    to discover positive engagement patterns.
    """

    te = TransactionEncoder()

    te_array = te.fit(transactions).transform(transactions)

    df_encoded = pd.DataFrame(
        te_array,
        columns=te.columns_
    )

    frequent_itemsets = apriori(
        df_encoded,
        min_support=min_support,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    rules = rules[
        rules["consequents"].apply(
            lambda x: x == frozenset({"high_engagement=True"})
        )
    ]

    rules = rules.sort_values(
        by=["confidence", "lift"],
        ascending=False
    )

    return rules


def display_positive_rules(rules, top_n=10):
    """
    Displays positive association rules
    in readable format.
    """

    print("\n" + "=" * 100)
    print("TOP POSITIVE ASSOCIATION RULES")
    print("=" * 100)

    for counter, (_, row) in enumerate(
            rules.head(top_n).iterrows(),
            start=1
    ):

        antecedents = list(row["antecedents"])
        consequents = list(row["consequents"])

        left_side = "  +  ".join(antecedents)
        right_side = "  +  ".join(consequents)

        print(f"\nRule {counter}")
        print("-" * 100)

        print(f"\n{left_side}")

        print("\n------------->\n")

        print(f"{right_side}")

        print("\nMETRICS:")

        print(f"Support     : {row['support']:.3f}")
        print(f"Confidence  : {row['confidence']:.3f}")
        print(f"Lift        : {row['lift']:.3f}")

        print("=" * 100)


def generate_positive_association_rules(df):
    """
    Executes the complete positive association
    rule mining pipeline.
    """

    df = compute_engagement(df)

    df = create_positive_target(df)

    df = feature_engineering(df)

    df = clean_and_select_features(df)

    transactions = prepare_transactions(df)

    rules = run_positive_apriori(transactions)

    return rules