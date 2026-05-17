# 04 Business Momentum Weekly Trends
# Classify business momentum with rolling windows and threshold experiments.

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

# Add project root to Python path
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

# Load KPI Dataset and Weekly Aggregation


if KPI_PATH.exists():
    df = pd.read_csv(KPI_PATH, parse_dates=["post_date"])
else:
    df = engineer_kpis(clean_dataset(load_raw_dataset(RAW_DATA_PATH)))
df.head()


# Overall Weekly Trends

weekly_trends = df.groupby("week", as_index=False).agg(
    total_engagement=("engagement", "sum"),
    avg_engagement_rate=("engagement_rate", "mean"),
    total_likes=("likes_count", "sum"),
    total_comments=("comments_count", "sum"),
    total_views=("views_count", "sum"),
    total_posts=("business_name", "size"),
).sort_values("week")

weekly_trends["engagement_growth"] = (
    weekly_trends["total_engagement"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

weekly_trends["trend_class"] = np.where(
    weekly_trends["engagement_growth"] > 0.10,
    "improving",
    np.where(
        weekly_trends["engagement_growth"] < -0.10,
        "declining",
        "stable"
    )
)

weekly_trends.head()


# Business Weekly Aggregation

business_weekly = df.groupby(["business_name", "sector", "week"], as_index=False).agg(
    engagement_rate=("engagement_rate", "mean"),
    engagement=("engagement", "mean"),
    posts_count=("business_name", "size"),
).sort_values(["business_name", "week"])

business_weekly.head()


# Rolling Window and Threshold Experiments

windows = [2,3,4]
thresholds = [0.05,0.10,0.15]
rows, store = [], {}
# Hyperparameter Experimentation:
# Test different rolling windows and growth thresholds to choose a stable,
# interpretable setup for business momentum classification.
# Values are selected by comparing stability, inconsistency, and clarity.
for w in windows:
    for t in thresholds:
        tmp = business_weekly.copy().copy()
        tmp["rolling_engagement"] = tmp.groupby("business_name")["engagement_rate"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        tmp["growth"] = tmp.groupby("business_name")["rolling_engagement"].pct_change().replace([np.inf,-np.inf], np.nan).fillna(0)
        tmp["trend_class"] = np.where(tmp["growth"] > t, "improving", np.where(tmp["growth"] < -t, "declining", "stable"))
        changes = tmp.groupby("business_name")["trend_class"].nunique().rename("n_states")
        tmp = tmp.merge(changes, on="business_name", how="left")
        tmp["final_class"] = np.where(tmp["n_states"] >= 3, "inconsistent", tmp["trend_class"])
        final = tmp.groupby("business_name", as_index=False).tail(1)
        rows.append({
            "rolling_window": w,
            "growth_threshold": t,
            "stable_ratio": (final["final_class"]=="stable").mean(),
            "inconsistent_ratio": (final["final_class"]=="inconsistent").mean(),
            "interpretability": 1 - abs(t-0.10) - abs(w-3)*0.05,
        })
        store[(w,t)] = tmp

exp = rank_models(pd.DataFrame(rows), higher_is_better_cols=["stable_ratio","interpretability"], lower_is_better_cols=["inconsistent_ratio"])
# Model/Configuration Ranking:
# Rank tested configurations and select the best one.
# Prefer higher stability and interpretability, with lower inconsistency.
best = exp.iloc[0]
tmp = store[(int(best["rolling_window"]), float(best["growth_threshold"]))]
business_momentum = tmp.groupby(["business_name","sector"], as_index=False).tail(1)[["business_name","sector","week","rolling_engagement","growth","final_class"]].rename(columns={"final_class":"momentum_class","rolling_engagement":"latest_rolling_engagement_rate","growth":"latest_growth"})
business_weekly_trends= tmp[["business_name","sector","week","engagement_rate","rolling_engagement","growth","final_class"]].rename(columns={"final_class":"trend_class"})


# Save Outputs and Insights

business_momentum.to_csv(PROCESSED_DIR / "business_momentum.csv", index=False)
weekly_trends.to_csv(PROCESSED_DIR / "weekly_trends.csv", index=False)
business_weekly_trends.to_csv(PROCESSED_DIR / "business_weekly_trends.csv", index=False)

exp.to_csv(REPORTS_DIR / "momentum_experiments.csv", index=False)
display(exp)
display(business_momentum.head(15))
print("Insight: use declining/inconsistent flags for immediate coaching priorities.")


#  Business Value
# This analysis translates social media metrics into actionable business insights instead of only generating numeric outputs.
# For example:
# - Weekly trends show whether overall engagement is improving or declining.
# - Business momentum identifies which businesses need attention.
# - Momentum classification helps determine whether a business is improving, declining, stable, or inconsistent over time.
# 
# This makes the outputs directly useful for coaching decisions, dashboard alerts, and content strategy evaluation.
