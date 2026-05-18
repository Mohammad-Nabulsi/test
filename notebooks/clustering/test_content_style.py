import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_module_path = ROOT / "notebooks" / "clustering" / "content-style_code.py"
spec = importlib.util.spec_from_file_location("content_style_code", _module_path)
cs = importlib.util.module_from_spec(spec)
sys.modules["content_style_code"] = cs
spec.loader.exec_module(cs)

DATA_PATH = r"C:\Users\hanib\data-mining-project\marketing\data\vanilla_kpi_dataset.json"
CLUSTER_JSON = cs.CONTENT_STYLE_DEFAULT_JSON

df = cs.load_posts_dataset(DATA_PATH)
business_name = df["business_name"].iloc[0]
print(f"Business: {business_name}")
print(f"Total posts: {len(df)}\n")

print("=== Running DBSCAN (eps=2.0, min_samples=25 from JSON) ===")
result = cs.analyze_business_content_style(DATA_PATH, business_name, CLUSTER_JSON)

print("--- Cluster Distribution ---")
print(result["clustered_posts"]["dbscan_fixed_cluster"].value_counts().sort_index().to_string())

print("\n--- Per-Cluster User Averages ---")
for _, row in result["cluster_summary"].iterrows():
    cid = int(row["cluster"])
    print(f"Cluster {cid} ({int(row['user_post_count'])} posts):")
    print(f"  eng_rate: {row['user_avg_engagement_rate']:.6f}  "
          f"view_rate: {row['user_avg_view_rate']:.6f}  "
          f"comment_rate: {row['user_avg_comment_rate']:.6f}")

print("\n--- Best / Worst Posts per Cluster ---")
for _, row in result["best_worst_posts"].iterrows():
    print(f"Cluster {int(row['cluster']):>2} | {row['rank']:<5} | "
          f"eng={row['engagement_rate']:.6f} | "
          f"view={row['view_rate']:.6f} | "
          f"comment={row['comment_rate']:.6f} | "
          f"score={row['combined_score']:.6f} | "
          f"type={row['post_type']}")

print("\n=== Pure Inference (pre-assigned clusters from cached data) ===")
result2 = cs.analyze_business_content_style_pure_inference("Vanilla Palestine")

print("--- Side-by-Side: User vs Cluster ---")
user_s = result2["user_summary"]
overall = result2["cluster_overall"]
merged = user_s.merge(overall, on="cluster", how="left")
for _, row in merged.iterrows():
    cid = int(row["cluster"])
    print(f"\nCluster {cid} ({int(row['user_post_count'])} user / {int(row['cluster_post_count'])} total posts):")
    print(f"  engagement_rate:  user={row['user_avg_engagement_rate']:.6f}  cluster={row['cluster_avg_engagement_rate']:.6f}")
    print(f"  view_rate:        user={row['user_avg_view_rate']:.6f}  cluster={row['cluster_avg_view_rate']:.6f}")
    print(f"  comment_rate:     user={row['user_avg_comment_rate']:.6f}  cluster={row['cluster_avg_comment_rate']:.6f}")

print("\n--- Best / Worst Posts per Cluster (cached) ---")
for _, row in result2["best_worst_posts"].iterrows():
    print(f"Cluster {int(row['cluster']):>2} | {row['rank']:<5} | "
          f"eng={row['engagement_rate']:.6f} | "
          f"view={row['view_rate']:.6f} | "
          f"score={row['combined_score']:.6f} | "
          f"type={row['post_type']}")
