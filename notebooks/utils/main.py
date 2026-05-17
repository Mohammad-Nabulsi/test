import pandas as pd
from pathlib import Path

from recommendation_system import (
    generate_recommendations,
    display_recommendations
)
from similar_business_recommender import (
    generate_similar_business_recommendations,
    display_similar_business_report
)


def load_or_generate_rules(base_dir, df):
    """
    Loads saved association rules, or regenerates them if pickle loading fails.
    """

    positive_rules_path = base_dir / "data" / "positive_rules.pkl"
    negative_rules_path = base_dir / "data" / "negative_rules.pkl"

    try:
        positive_rules = pd.read_pickle(positive_rules_path)
        negative_rules = pd.read_pickle(negative_rules_path)

        print("\nAssociation rules loaded successfully.")
        print(f"Positive rules: {len(positive_rules)}")
        print(f"Negative rules: {len(negative_rules)}")

        return positive_rules, negative_rules

    except Exception as error:
        print("\nSaved rule files could not be loaded.")
        print(f"Reason: {error}")
        print("Trying to regenerate rules from the sample dataset...")

    try:
        from association_rules_positive_module import (
            generate_positive_association_rules
        )
        from association_rules_negative_module import (
            generate_negative_association_rules
        )

        positive_rules = generate_positive_association_rules(df.copy())
        negative_rules = generate_negative_association_rules(df.copy())

        print("Rules regenerated successfully.")
        print(f"Positive rules: {len(positive_rules)}")
        print(f"Negative rules: {len(negative_rules)}")

        return positive_rules, negative_rules

    except Exception as error:
        print("\nCould not regenerate association rules in this environment.")
        print(f"Reason: {error}")
        print("Continuing with the Similar Business Recommender.")

        return pd.DataFrame(), pd.DataFrame()


def choose_business_for_comparison(df):
    """
    Picks a business with useful peer comparisons.
    """

    if "engagement_rate_followers" not in df.columns:
        return str(df["business_name"].dropna().iloc[0])

    sector_counts = df.groupby("sector")["business_name"].nunique()
    sectors_with_peers = sector_counts[sector_counts > 1].index

    comparison_df = df[df["sector"].isin(sectors_with_peers)].copy()

    if comparison_df.empty:
        comparison_df = df.copy()

    business_scores = (
        comparison_df
        .groupby(["business_name", "sector"], dropna=False)
        .agg(
            avg_engagement_rate=(
                "engagement_rate_followers",
                "mean"
            ),
            posts=("business_name", "size")
        )
        .reset_index()
        .sort_values(
            by=["avg_engagement_rate", "posts"],
            ascending=[True, False]
        )
    )

    return str(business_scores.iloc[0]["business_name"])


def main():
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "sample_synthetic_posts.csv"
    df = pd.read_csv(data_path)

    print("\n" + "=" * 80)
    print("MARKETING RECOMMENDATION MAIN RUN")
    print("=" * 80)
    print(f"Dataset: {data_path}")
    print(f"Rows: {len(df)}")
    print(f"Businesses: {df['business_name'].nunique()}")
    print(f"Sectors: {df['sector'].nunique()}")

    positive_rules, negative_rules = load_or_generate_rules(
        base_dir,
        df
    )

    user_post = {
        "language": "Arabic",
        "post_type": "image",
        "posting_hour": 10,
        "hashtags_count": 3,
        "caption_length": 70,
        "discount_percent": 0,
        "arabic_dialect_style": True,
        "CTA_present": True,
        "day_of_week": "Saturday"
    }

    print("\n" + "=" * 80)
    print("POST ENGAGEMENT RECOMMENDER")
    print("=" * 80)

    if positive_rules.empty and negative_rules.empty:
        print("No association-rule recommendations available.")
    else:
        recommendations = generate_recommendations(
            user_post=user_post,
            positive_rules=positive_rules,
            negative_rules=negative_rules
        )

        display_recommendations(
            recommendations,
            top_n=10
        )

    comparison_business = choose_business_for_comparison(df)

    similar_business_recommendations, metadata = (
        generate_similar_business_recommendations(
            posts_df=df,
            business_name=comparison_business,
            top_n=5
        )
    )

    display_similar_business_report(
        recommendations_df=similar_business_recommendations,
        metadata=metadata
    )


if __name__ == "__main__":
    main()
