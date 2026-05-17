import pandas as pd
from pathlib import Path

from association_rules_positive_module import (
    generate_positive_association_rules
)

from association_rules_negative_module import (
    generate_negative_association_rules
)


def main():

    BASE_DIR = Path(__file__).resolve().parents[2]

    data_path = (
        BASE_DIR
        / "data"
        / "sample_synthetic_posts.csv"
    )

    df = pd.read_csv(data_path)

    print("Generating positive rules...")

    positive_rules = (
        generate_positive_association_rules(df)
    )
    print(
    f"Positive rules count: "
    f"{len(positive_rules)}"
)

    print("Generating negative rules...")

    negative_rules = (
        generate_negative_association_rules(df)
    )
    print(
    f"Negative rules count: "
    f"{len(negative_rules)}"
)

    # Save rules
    positive_rules.to_pickle(
        BASE_DIR
        / "data"
        / "positive_rules.pkl"
    )

    negative_rules.to_pickle(
        BASE_DIR
        / "data"
        / "negative_rules.pkl"
    )

    print("\nRules saved successfully.")


if __name__ == "__main__":
    main()