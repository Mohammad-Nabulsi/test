# 00 KPI Engineering
# Create all required KPIs, bins, quality checks, summaries, and exports.


import warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "utils" / "utils.py").exists():
    if (PROJECT_ROOT / "notebooks" / "utils" / "utils.py").exists():
        PROJECT_ROOT = PROJECT_ROOT / "notebooks"
    elif (PROJECT_ROOT.parent / "utils" / "utils.py").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import ensure_project_dirs, load_raw_dataset, clean_dataset, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR
from utils.features import engineer_kpis, build_post_feature_sets, aggregate_business_features
from utils.evaluation import regression_metrics, rank_models
from utils.visualization import set_plot_style, save_figure
from pathlib import Path

set_plot_style()
ensure_project_dirs()

PROJECT_ROOT = Path(__file__).resolve()

while PROJECT_ROOT.name != "marketing":
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DATA_PATH = PROJECT_ROOT / "jsons" / "all_final_appended.json"

if not RAW_DATA_PATH.exists():
    RAW_DATA_PATH = PROJECT_ROOT / "synthetic_generator" / "synthetic_social_media_posts.csv"

KPI_PATH = PROJECT_ROOT / "data" / "processed" / "kpi_dataset.csv"

# Load and Clean Data

def load_raw_dataset(path):
    data_path = Path(path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path)

    if data_path.suffix.lower() == ".json":
        return pd.read_json(data_path)

    raise ValueError(f"Unsupported file type: {data_path.suffix}")


raw_df = load_raw_dataset(RAW_DATA_PATH)
clean_df = clean_dataset(raw_df)

print("Raw:", raw_df.shape, "Clean:", clean_df.shape)

display(clean_df.head())


# Engineer KPIs and Validate Division Safety

df = engineer_kpis(clean_df)
df["post_date"] = pd.to_datetime(
    df["post_date"],
    errors="coerce"
)
rate_cols = ["engagement_rate", "like_rate", "comment_rate", "view_rate", "view_engagement_rate"]
invalid = {c: int(np.isinf(df[c]).sum() + df[c].isna().sum()) for c in rate_cols}
print("Invalid rate values:", invalid)
print("followers_count==0:", int((df["followers_count"] == 0).sum()))
print("views_count==0:", int((df["views_count"] == 0).sum()))
# Weekly KPI
df["week"] = (
    df["post_date"]
    .dt.to_period("W")
    .astype(str)
)
display(df.head())


# Summary Tables and Exports

if "week" not in df.columns:
    df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
    df["week"] = df["post_date"].dt.to_period("W").astype(str)
summaries = {}
for key, grp in {
    "sector": ["sector"],
    "business": ["business_name", "sector"],
    "post_type": ["post_type"],
    "week": ["week"],

}.items():
    summaries[key] = df.groupby(grp, as_index=False).agg(
        posts_count=("business_name", "size"),
        engagement_mean=("engagement", "mean"),
        engagement_rate_mean=("engagement_rate", "mean"),
        view_rate_mean=("view_rate", "mean"),
        comment_rate_mean=("comment_rate", "mean"),
    ).sort_values("engagement_rate_mean", ascending=False)
    display(summaries[key].head(10))

df.to_csv(PROCESSED_DIR / "kpi_dataset.csv", index=False)
for k, v in summaries.items():
    v.to_csv(REPORTS_DIR / f"kpi_summary_{k}.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
plot_df = summaries["sector"].head(15)
sns.barplot(data=plot_df, x="engagement_rate_mean", y="sector", ax=ax, palette="viridis")
ax.set_title("Sector Engagement Rate Benchmark")
save_figure(fig, FIGURES_DIR, "kpi_sector_engagement_rate.png")
plt.show()

top_sector = summaries["sector"].iloc[0]
print(f"Insight: Top sector is {top_sector['sector']} with engagement rate {top_sector['engagement_rate_mean']:.4f}.")

