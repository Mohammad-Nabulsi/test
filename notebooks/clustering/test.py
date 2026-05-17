import sys
from pathlib import Path
import importlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # ...\marketing
sys.path.insert(0, str(ROOT))

import notebooks.clustering.business_clustring_code as bc
importlib.reload(bc)

print("Loaded:", bc.__file__)

# 1) load dataset
data_path = r"C:\Users\hanib\data-mining-project\marketing\data\vanilla_kpi_dataset.json"
try:
    df = pd.read_json(data_path)
    print(f"Loaded JSON without lines=True (Shape: {df.shape})")
except ValueError:
    df = pd.read_json(data_path, lines=True)

# 2) build business features (test preprocessing)
biz = bc.preprocess_business_dataset(df)
print("biz shape:", biz.shape)
print("columns:", biz.columns.tolist())
print(biz.head().to_string(index=False))

# 3) pick one business name to test compare
test_business = biz["business_name"].iloc[0]
print("testing business:", test_business)

# 4) run full compare (uses JSON clusters/averages)
out = bc.compare_business_pure_inference(
    biz_features=biz,
    business_name=test_business,
    cluster_json_path=r"C:\users\hanib\data-mining-project\marketing\notebooks\clustering\artifacts\business_clustering\json\business_cluster_coordinates.json",
)

# 5) show higher/lower/equal result
print(out[["feature", "business_value", "cluster_avg", "comparison"]].to_string(index=False))
