import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import notebooks.clustering.business_clustring_code as bc

data_path = r"C:\Users\hanib\data-mining-project\marketing\data\processed\99streetwear.json"
cluster_json_path = r"C:\Users\hanib\data-mining-project\marketing\notebooks\clustering\artifacts\business_clustering\json\business_cluster_coordinates.json"

try:
    df = pd.read_json(data_path)
except ValueError:
    df = pd.read_json(data_path, lines=True)
biz = bc.preprocess_business_dataset(df)
test_business = biz["business_name"].iloc[0]
print(f"testing business: {test_business}")

out = bc.compare_business_pure_inference(
    biz_features=biz,
    business_name=test_business,
    cluster_json_path=cluster_json_path,
)

print(out[["feature", "business_value", "cluster_avg", "comparison"]].to_string(index=False))
print(f"\nassigned cluster: {out['kmeans_cluster'].iloc[0]}")
print("No KMeans.fit() called. Pure inference via business_clustring_code.")