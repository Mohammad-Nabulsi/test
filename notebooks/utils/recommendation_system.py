"""
engagement_recommendation_system.py

Recommendation system for improving social media engagement
using positive and negative association rules.
"""

import pandas as pd


# Positive recommendation mapping
positive_mapping = {

    "post_type=reel":
        "Use reel videos instead of static images.",

    "time_group=evening":
        "Post during evening hours.",

    "time_group=afternoon":
        "Post during afternoon hours.",

    "hashtags_group=medium":
        "Use a moderate number of hashtags.",

    "hashtags_group=high":
        "Increase hashtag usage.",

    "CTA_present=True":
        "Add a call-to-action in the caption.",

    "CTA_present=False":
        "Avoid excessive call-to-actions.",

    "emoji_count=2":
        "Use a small number of emojis.",

    "mentions_location=True":
        "Mention locations in your posts.",

    "mentions_location=False":
        "Avoid unnecessary location mentions.",

    "promo_post=True":
        "Use promotional-style content.",

    "promo_post=False":
        "Avoid overly promotional content.",

    "caption_group=medium":
        "Use medium-length captions.",

    "caption_group=short":
        "Use shorter captions.",

    "day_of_week=Saturday":
        "Post on Saturdays for better engagement.",

    "arabic_dialect_style=True":
        "Use conversational Arabic dialect.",

    "religious_theme=False":
        "Avoid excessive religious themes.",

    "patriotic_theme=False":
        "Avoid excessive patriotic themes."
}


# Negative recommendation mapping
negative_mapping = {

    "time_group=morning":
        "posting in the morning",

    "hashtags_group=low":
        "using very few hashtags",

    "CTA_present=True":
        "adding too many call-to-actions",

    "mentions_location=False":
        "not mentioning locations",

    "promo_post=False":
        "avoiding promotional content",

    "religious_theme=False":
        "avoiding religious themes",

    "patriotic_theme=False":
        "avoiding patriotic themes",

    "discount_group=none":
        "not offering discounts",

    "post_type=image":
        "using static image posts",

    "language=Arabic":
        "using Arabic language"
}


def prepare_user_post(user_post):
    """
    Applies feature engineering
    on user input.
    """

    # Time grouping
    if "posting_hour" in user_post:

        hour = user_post["posting_hour"]

        if hour < 12:
            user_post["time_group"] = "morning"

        elif hour < 18:
            user_post["time_group"] = "afternoon"

        else:
            user_post["time_group"] = "evening"

    # Caption grouping
    if "caption_length" in user_post:

        length = user_post["caption_length"]

        if length <= 50:
            user_post["caption_group"] = "short"

        elif length <= 100:
            user_post["caption_group"] = "medium"

        else:
            user_post["caption_group"] = "long"

    # Hashtag grouping
    if "hashtags_count" in user_post:

        hashtags = user_post["hashtags_count"]

        if hashtags <= 5:
            user_post["hashtags_group"] = "low"

        elif hashtags <= 15:
            user_post["hashtags_group"] = "medium"

        else:
            user_post["hashtags_group"] = "high"

    # Discount grouping
    if "discount_percent" in user_post:

        discount = user_post["discount_percent"]

        if discount == 0:
            user_post["discount_group"] = "none"

        elif discount <= 25:
            user_post["discount_group"] = "low"

        else:
            user_post["discount_group"] = "high"

    return user_post


def calculate_recommendation_score(
        confidence,
        lift,
        support
):
    """
    Calculates recommendation score.
    """

    score = (
            (confidence * 100)
            + (lift * 10)
            + (support * 100)
    )

    return round(score, 1)


def generate_recommendations(
        user_post,
        positive_rules,
        negative_rules
):
    """
    Generates engagement recommendations.
    """

    # User preprocessing
    user_post = prepare_user_post(
        user_post
    )

    # Remove raw features
    for key in (
        "posting_hour",
        "caption_length",
        "hashtags_count",
        "discount_percent"
    ):
        user_post.pop(key, None)

    # Default values
    default_user_post = {

        "post_type": "image",

        "time_group": "morning",

        "hashtags_group": "low",

        "caption_group": "short",

        "discount_group": "none",

        "promo_post": False,

        "CTA_present": False,

        "mentions_location": False,

        "arabic_dialect_style": False,

        "religious_theme": False,

        "patriotic_theme": False
    }

    # Fill missing features
    for key, value in default_user_post.items():

        if key not in user_post:

            user_post[key] = value

    # Transaction format
    user_items = {
        f"{key}={value}"
        for key, value in user_post.items()
    }

    recommendations = []

    # Best recommendation per feature
    best_feature_recommendations = {}

    # Positive recommendations
    for _, rule in positive_rules.iterrows():

        antecedents = set(
            rule["antecedents"]
        )

        overlap = antecedents.intersection(
            user_items
        )

        missing_items = antecedents - user_items

        if (
                len(overlap) > 0
                and
                len(missing_items) > 0
        ):

            for item in missing_items:

                feature_name = item.split("=")[0]

                recommendation_text = (
                    positive_mapping.get(
                        item,
                        f"Consider using {item}"
                    )
                )

                score = (
                    calculate_recommendation_score(
                        confidence=rule[
                            "confidence"
                        ],

                        lift=rule["lift"],

                        support=rule[
                            "support"
                        ]
                    )
                )

                # Keep only highest score
                if (
                        feature_name
                        in
                        best_feature_recommendations
                ):

                    old_score = (
                        best_feature_recommendations[
                            feature_name
                        ]["score"]
                    )

                    if score <= old_score:
                        continue

                best_feature_recommendations[
                    feature_name
                ] = {

                    "type": "positive",

                    "recommendation":
                        recommendation_text,

                    "score": score
                }

    # Convert positive recommendations
    for recommendation_data in (
            best_feature_recommendations.values()
    ):

        recommendations.append(
            recommendation_data
        )

    # Negative recommendations
    added_negative_patterns = set()

    for _, rule in negative_rules.iterrows():

        antecedents = set(
            rule["antecedents"]
        )

        if antecedents.issubset(
                user_items
        ):

            formatted_items = []

            skip_pattern = False

            for item in antecedents:

                feature_name = item.split("=")[0]

                if (
                        feature_name
                        in
                        best_feature_recommendations
                ):
                    skip_pattern = True
                    break

                formatted_items.append(
                    negative_mapping.get(
                        item,
                        item
                    )
                )

            if skip_pattern:
                continue

            pattern_text = (
                ", ".join(formatted_items[:4])
            )

            recommendation_text = (
                f"Avoid combining: "
                f"{pattern_text}"
            )

            if (
                    recommendation_text
                    in
                    added_negative_patterns
            ):
                continue

            added_negative_patterns.add(
                recommendation_text
            )

            score = (
                calculate_recommendation_score(
                    confidence=rule[
                        "confidence"
                    ],

                    lift=rule["lift"],

                    support=rule[
                        "support"
                    ]
                )
            )

            recommendations.append({

                "type": "negative",

                "recommendation":
                    recommendation_text,

                "score": score
            })

    recommendations_df = pd.DataFrame(
        recommendations
    )

    # Sort recommendations
    if not recommendations_df.empty:

        recommendations_df = (
            recommendations_df
            .sort_values(
                by="score",
                ascending=False
            )
        )

    return recommendations_df


def display_recommendations(
        recommendations_df,
        top_n=10
):
    """
    Displays recommendations.
    """

    if recommendations_df.empty:

        print(
            "\nNo recommendations found."
        )

        return

    for counter, (_, row) in enumerate(
            recommendations_df
            .head(top_n)
            .iterrows(),

            start=1
    ):

        print(
            f"\n{counter}. "
            f"{row['recommendation']}"
        )

        print(
            f"Recommendation Score: "
            f"{row['score']}"
        )

        print("-" * 80)
