
# 05 Anomaly Detection
# Compare Z-score, Isolation Forest, and LOF for outlier detection.

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


# Load KPI and Build Feature Matrix

if KPI_PATH.exists():
    df = pd.read_csv(KPI_PATH, parse_dates=["post_date"])
else:
    df = engineer_kpis(clean_dataset(load_raw_dataset(RAW_DATA_PATH)))
df.head()



from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

feat = ["engagement_rate","view_rate","comment_rate","like_rate","view_engagement_rate","discount_percent","hashtags_count","emoji_count","caption_length"]
X = StandardScaler().fit_transform(df[feat].fillna(0))


# Experiment Grid and Method Comparison

z_vals, conts, neighs = [2,2.5,3], [0.03,0.05,0.1], [10,20,35]
rows, outputs = [], {}

zmax = np.abs(X).max(axis=1)
for z in z_vals:
    y = np.where(zmax >= z, -1, 1)
    out = df.copy()
    out["anomaly_label"] = y
    out["method"] = f"zscore_{z}"
    out["anomaly_type"] = np.where((y==-1) & (out["engagement_rate"] >= out["engagement_rate"].median()), "positive_anomaly", np.where(y==-1, "negative_anomaly", "normal"))
    outputs[f"zscore_{z}"] = out
    rows.append({"method":"zscore","setting":f"threshold={z}","anomalies":int((y==-1).sum())})

for c in conts:
    y = IsolationForest(contamination=c, random_state=42).fit_predict(X)
    out = df.copy()
    out["anomaly_label"] = y
    out["method"] = f"iforest_{c}"
    out["anomaly_type"] = np.where((y==-1) & (out["engagement_rate"] >= out["engagement_rate"].median()), "positive_anomaly", np.where(y==-1, "negative_anomaly", "normal"))
    outputs[f"iforest_{c}"] = out
    rows.append({"method":"isolation_forest","setting":f"contamination={c}","anomalies":int((y==-1).sum())})

for n in neighs:
    y = LocalOutlierFactor(n_neighbors=n, contamination=0.05).fit_predict(X)
    out = df.copy()
    out["anomaly_label"] = y
    out["method"] = f"lof_{n}"
    out["anomaly_type"] = np.where((y==-1) & (out["engagement_rate"] >= out["engagement_rate"].median()), "positive_anomaly", np.where(y==-1, "negative_anomaly", "normal"))
    outputs[f"lof_{n}"] = out
    rows.append({"method":"lof","setting":f"n_neighbors={n}","anomalies":int((y==-1).sum())})

comp = pd.DataFrame(rows)
comp["interpretability"] = comp["method"].map({"zscore":0.95,"isolation_forest":0.75,"lof":0.7})
comp["anomaly_balance"] = 1 - (comp["anomalies"] - comp["anomalies"].median()).abs() / max(comp["anomalies"].max(),1)
ranked = rank_models(comp, higher_is_better_cols=["interpretability","anomaly_balance"])
best = ranked.iloc[0]

overlap_rows = []
keys = list(outputs.keys())
for m1 in keys:
    s1 = set(outputs[m1].index[outputs[m1]["anomaly_label"] == -1].tolist())
    for m2 in keys:
        s2 = set(outputs[m2].index[outputs[m2]["anomaly_label"] == -1].tolist())
        union = max(len(s1 | s2), 1)
        overlap_rows.append({"method_1": m1, "method_2": m2, "jaccard_overlap": len(s1 & s2) / union})
overlap = pd.DataFrame(overlap_rows)

best_key = next(k for k in outputs if str(best["setting"]).split("=")[1] in k and (k.startswith("zscore") if best["method"]=="zscore" else k.startswith("iforest") if best["method"]=="isolation_forest" else k.startswith("lof")))
anomalies = outputs[best_key].query("anomaly_label == -1")[["business_name","sector","post_date","post_type","engagement_rate","view_rate","comment_rate","method","anomaly_type"]]
# Anomaly Detection Hyperparameters:
# Parameters such as contamination, neighbor count, or z-score thresholds control how strict the anomaly detection is.
# Testing these settings helps balance between detecting too many false anomalies and missing important unusual posts.


# Save Outputs

anomalies.to_csv(PROCESSED_DIR / "anomalies.csv", index=False)
ranked.to_csv(REPORTS_DIR / "anomaly_experiments.csv", index=False)
overlap.to_csv(REPORTS_DIR / "anomaly_overlap_matrix.csv", index=False)
display(ranked)
display(anomalies.head(20))
print("Insight: positive anomalies are replication candidates; negative anomalies are intervention candidates.")


# Business Value:
# Positive anomalies can highlight successful content patterns that should be repeated.
# Negative anomalies can identify posts that need attention, improvement, or strategy adjustment.

