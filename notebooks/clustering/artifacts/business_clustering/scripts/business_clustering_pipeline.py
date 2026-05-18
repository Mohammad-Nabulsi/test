# Auto-exported from 04_business_clustring.ipynb
# Generated from notebook run context

# --- Markdown cell 0 ---
# # 04 Business Clustring
# 
# Goal: cluster businesses by marketing behavior and performance to answer:
# **"What type of marketer is this business?"**
# 

# --- Code cell 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from IPython.display import display
sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', 200)

# --- Markdown cell 2 ---
# ## Load Data
# 

# --- Code cell 3 ---
DATA_PATH = r"C:\users\hanib\data_mining -project\marketing\data\processed\kpi_dataset.json"
#  C:\users\hanib\data_mining -project\marketing\data\processed\kpi_dataset.json
# C:\users\hanib\data_mining -project\marketing\data\data_processed.json

try:
    df = pd.read_json(DATA_PATH)
except ValueError:
    df = pd.read_json(DATA_PATH, lines=True)

print('Shape:', df.shape)
display(df.head(3))

# --- Markdown cell 4 ---
# ## Simple EDA
# 

# --- Code cell 5 ---
print('Columns:')
print(df.columns.tolist())

print('Missing values:')
display(df.isna().sum().sort_values(ascending=False).to_frame('missing'))

print('Sector counts:')
display(df['sector'].value_counts(dropna=False).to_frame('posts_count'))

print('Post type counts:')
display(df['post_type'].value_counts(dropna=False).to_frame('posts_count'))

# --- Markdown cell 6 ---
# ## Prepare Business-Level Features
# 

# --- Code cell 7 ---
df = df[df["business_name"] != "Family Market PS"].copy()

work = df.copy()

# helper to normalize bool-like columns from mixed JSON values

def to_binary_flag(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    if isinstance(x, (int, float)):
        return int(x != 0)
    t = str(x).strip().lower()
    if t in {'1','true','yes','y','on','present'}:
        return 1
    if t in {'0','false','no','n','off','none','nan',''}:
        return 0
    return 1

# numeric conversion
for c in ['followers_count','likes_count','comments_count','views_count','caption_length','hashtags_count','emoji_count','discount_percent' , 'engagement_rate','view_rate','comment_rate' , "engagement" , "post_date" ]:
    if c in work.columns:
        work[c] = pd.to_numeric(work[c], errors='coerce')
    else:
        work[c] = np.nan

# fill numeric missing with safe defaults
# work['followers_count'] = work['followers_count'].fillna(0)
# work['likes_count'] = work['likes_count'].fillna(0)
# work['comments_count'] = work['comments_count'].fillna(0)
# work['views_count'] = work['views_count'].fillna(0)
# work['caption_length'] = work['caption_length'].fillna(work['caption_length'].median())
# work['hashtags_count'] = work['hashtags_count'].fillna(0)
# work['emoji_count'] = work['emoji_count'].fillna(0)
# work['discount_percent'] = work['discount_percent'].fillna(0)

# binary columns
for c in ['CTA_present','promo_post','mentions_location','religious_theme','patriotic_theme','arabic_dialect_style']:
    if c in work.columns:
        work[c] = work[c].apply(to_binary_flag).astype(int)
    else:
        work[c] = 0

# post_type normalization
work['post_type'] = work['post_type'].astype(str).str.strip().str.lower()
work['post_type_std'] = np.select(
    [
        work['post_type'].str.contains('reel', na=False),
        work['post_type'].str.contains('carousel', na=False),
        work['post_type'].str.contains('image|photo|post', na=False),
    ],
    ['reel', 'carousel', 'image'],
    default='other'
)

# core rates at post-level
# work['engagement'] = work['likes_count'] + work['comments_count']
# work['engagement_rate'] = np.where(work['followers_count'] > 0, work['engagement'] / work['followers_count'], np.nan)
# work['view_rate'] = np.where(work['followers_count'] > 0, work['views_count'] / work['followers_count'], np.nan)
# work['comment_rate'] = np.where(work['followers_count'] > 0, work['comments_count'] / work['followers_count'], np.nan)

# date for posting frequency
work['post_date'] = pd.to_datetime(work.get('post_date'), errors='coerce')

# business aggregation
biz_base = (
    work.groupby(['business_name', 'sector'], dropna=False)
    .agg(
        followers_count=('followers_count', 'max'),
        posts_count=('business_name', 'size'),
        avg_engagement_rate=('engagement_rate', 'mean'),
        avg_view_rate=('view_rate', 'mean'),
        avg_comment_rate=('comment_rate', 'mean'),
        avg_caption_length=('caption_length', 'mean'),
        avg_hashtags_count=('hashtags_count', 'mean'),
        avg_emoji_count=('emoji_count', 'mean'),
        percentage_promo_posts=('promo_post', 'mean'),
        percentage_CTA_posts=('CTA_present', 'mean'),
        percentage_location_posts=('mentions_location', 'mean'),
        percentage_religious_theme=('religious_theme', 'mean'),
        percentage_patriotic_theme=('patriotic_theme', 'mean'),
        percentage_arabic_dialect_style=('arabic_dialect_style', 'mean')
    )
    .reset_index()
)

# posting frequency (posts per active day)
active_days = (
    work.groupby('business_name')['post_date']
    .agg(lambda s: (s.max() - s.min()).days + 1 if s.notna().any() else np.nan)
    .rename('active_days')
)

biz = biz_base.merge(active_days, on='business_name', how='left')
biz['posting_frequency'] = np.where(biz['active_days'] > 0, biz['posts_count'] / biz['active_days'], np.nan)

# post type percentages
pt = pd.crosstab(work['business_name'], work['post_type_std'], normalize='index')
for c in ['reel', 'image', 'carousel']:
    if c not in pt.columns:
        pt[c] = 0
pt = pt[['reel','image','carousel']].rename(columns={
    'reel':'percentage_reels',
    'image':'percentage_images',
    'carousel':'percentage_carousels'
}).reset_index()

biz = biz.merge(pt, on='business_name', how='left')

# fill remaining nulls
for c in biz.columns:
    if biz[c].dtype.kind in 'biufc':
        med = biz[c].median()
        if pd.isna(med):
            med = 0
        biz[c] = biz[c].fillna(med)

# convert proportions to percentages for readability
pct_cols = [
    'percentage_reels','percentage_images','percentage_carousels',
    'percentage_promo_posts','percentage_CTA_posts','percentage_location_posts',
    'percentage_religious_theme','percentage_patriotic_theme','percentage_arabic_dialect_style'
]
for c in pct_cols:
    biz[c] = biz[c] * 100

print('Business rows:', len(biz))
display(biz.head(5))

# --- Markdown cell 8 ---
# ## Feature Matrix For Clustering
# 

# --- Code cell 9 ---
biz['log_followers_count'] = np.log1p(biz['followers_count'])

feature_cols = [
    'log_followers_count',
    'avg_engagement_rate',
    'avg_view_rate',
    'avg_comment_rate',
    'posting_frequency',
    'avg_caption_length',
    'avg_hashtags_count',
    'avg_emoji_count',
    'percentage_reels',
    'percentage_images',
    'percentage_carousels',
    'percentage_promo_posts',
    'percentage_CTA_posts',
    'percentage_location_posts',
    'percentage_religious_theme',
    'percentage_patriotic_theme',
    'percentage_arabic_dialect_style'
]

X = biz[feature_cols].copy()
X = X.replace([np.inf, -np.inf], np.nan)
for c in feature_cols:
    X[c] = pd.to_numeric(X[c], errors='coerce')
    X[c] = X[c].fillna(X[c].median() if not pd.isna(X[c].median()) else 0)

print('Any NaN in X:', bool(X.isna().any().any()))
print('X shape:', X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Markdown cell 10 ---
# ## KMeans (Elbow + Silhouette)
# 

# --- Code cell 11 ---
inertias = []
sils = []
K = range(2, min(11, len(X_scaled)))

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=30)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, labels))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(list(K), inertias, marker='o')
ax[0].set_title('Elbow Method')
ax[0].set_xlabel('k')
ax[0].set_ylabel('Inertia')

ax[1].plot(list(K), sils, marker='o', color='darkgreen')
ax[1].set_title('Silhouette by k')
ax[1].set_xlabel('k')
ax[1].set_ylabel('Silhouette')

plt.tight_layout()
plt.show()

best_k = list(K)[int(np.argmax(sils))]
print('Best k (silhouette):', best_k)

# --- Markdown cell 12 ---
# ## Fit KMeans + t-SNE Visualization
# 

# --- Code cell 13 ---
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=30)
biz['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

print('KMeans cluster counts:')
display(biz['kmeans_cluster'].value_counts().sort_index().to_frame('business_count'))

# t-SNE 2D map for business visualization
perplexity = max(5, min(30, len(X_scaled)//5))
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_scaled)

biz['tsne_1'] = X_tsne[:, 0]
biz['tsne_2'] = X_tsne[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=biz, x='tsne_1', y='tsne_2', hue='kmeans_cluster', palette='tab10', s=70, alpha=0.85)
plt.title(f'Business Clusters (KMeans k={best_k}) - t-SNE 2D')
plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()


# --- Markdown cell 14 ---
# ## DBSCAN
# 

# --- Code cell 15 ---
# Manual DBSCAN (set these by hand)
DBSCAN_EPS = 3.1
DBSCAN_MIN_SAMPLES = 4

db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
labels = db.fit_predict(X_scaled)

biz['dbscan_cluster'] = labels

n_clusters = len(set(labels) - {-1})
noise_ratio = float((labels == -1).mean())

sil = np.nan
mask = labels != -1
if n_clusters >= 2 and mask.sum() > n_clusters and len(np.unique(labels[mask])) > 1:
    sil = silhouette_score(X_scaled[mask], labels[mask])

print({'eps': DBSCAN_EPS, 'min_samples': DBSCAN_MIN_SAMPLES})
print('DBSCAN clusters:', n_clusters, '| noise_ratio:', round(noise_ratio, 3), '| silhouette_non_noise:', None if np.isnan(sil) else round(float(sil), 4))
display(biz['dbscan_cluster'].value_counts().sort_index().to_frame('business_count'))

plt.figure(figsize=(10, 6))
sns.scatterplot(data=biz, x='tsne_1', y='tsne_2', hue='dbscan_cluster', palette='tab10', s=70, alpha=0.85)
plt.title(f"Business Clusters (DBSCAN eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}) - t-SNE 2D")
plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()


# --- Markdown cell 16 ---
# ## Cluster Profiling + Sector Distribution
# 

# --- Code cell 17 ---
profile_cols = [
    'avg_engagement_rate','avg_view_rate','avg_comment_rate','posting_frequency',
    'avg_caption_length','avg_hashtags_count','avg_emoji_count',
    'percentage_reels','percentage_images','percentage_carousels',
    'percentage_promo_posts','percentage_CTA_posts','percentage_location_posts',
    'percentage_religious_theme','percentage_patriotic_theme','percentage_arabic_dialect_style'
]

kmeans_profile = biz.groupby('kmeans_cluster')[profile_cols].mean().round(3)
print('KMeans cluster profile:')
display(kmeans_profile)

print('KMeans sector distribution:')
kmeans_sector = pd.crosstab(biz['kmeans_cluster'], biz['sector'], normalize='index').round(3) * 100
display(kmeans_sector)

print('DBSCAN cluster profile (excluding noise -1):')
db_prof = biz[biz['dbscan_cluster'] != -1].groupby('dbscan_cluster')[profile_cols].mean().round(3)
display(db_prof)

print('DBSCAN sector distribution (excluding noise -1):')
db_sec = pd.crosstab(biz[biz['dbscan_cluster'] != -1]['dbscan_cluster'], biz[biz['dbscan_cluster'] != -1]['sector'], normalize='index').round(3) * 100
display(db_sec)

# --- Markdown cell 18 ---
# ## Recommendation Template (Per Cluster)
# 
# - Cluster 0 — Active Sellers
# 
# They post a lot, use many reels, strong CTA, and many promo posts. Focus is to push people to act now.
# 
# -Cluster 1 — Community Voices
# 
# They post less, rarely use promo/CTA, but strongly use local/patriotic themes. Focus is identity and trust, not direct selling.
# 
# - Cluster 2 — Reach Chasers
# 
# They get high views, mostly reels, and use promo often. Focus is visibility and awareness more than deep engagement.
# - Cluster 3 — Brand Storytellers
# 
# They mix reels and images, use longer captions and expressive style. Focus is brand feel and storytelling, softer selling.
# 

# --- Markdown cell 19 ---
# ## Save DBSCAN Artifacts For Inference
# 
# Save all objects needed to assign DBSCAN cluster to new businesses without retraining.
# 

# --- Markdown cell 20 ---
# ## DBSCAN Inference (No Retraining)
# 
# Inference rule: scale input, find training neighbors within `eps`; if neighbors >= `min_samples`, assign dominant non-noise label, else `-1`.
# 

# --- Code cell 21 ---
biz4 = biz[biz["dbscan_cluster"].isin([0, 1, 2, 3])].copy()

# Aggregate on your full business-clustering dataframe
cluster_report = biz4.groupby("dbscan_cluster").agg({
    "business_name": "nunique",
    "posts_count": ["mean", "median"],
    "avg_engagement_rate": ["mean", "median"],
    "avg_view_rate": ["mean", "median"],
    "followers_count": ["mean", "median"],
}).round(4)

display(cluster_report)

# --- Code cell 22 ---
# Top 5 businesses per cluster by engagement rate
top5_per_cluster = (
    biz.sort_values(["dbscan_cluster", "avg_engagement_rate"], ascending=[True, False])
       .groupby("dbscan_cluster", as_index=False)
       .head(5)
)

display(top5_per_cluster[[
    "dbscan_cluster",
    "business_name",
    "sector",
    "avg_engagement_rate",
    "avg_view_rate",
    "posting_frequency",
    "followers_count"
    ]])

# --- Code cell 23 ---
# Compare only clusters 0 and 2 (using your notebook column names)
biz[biz["dbscan_cluster"].isin([0, 2])].groupby("dbscan_cluster").agg({
    "avg_engagement_rate": ["mean", "median"],
    "avg_view_rate": ["mean", "median"],
    "posts_count": ["mean", "median"],
    "followers_count": ["mean", "median"]
}).round(4)

# --- Code cell 24 ---

print("cluster_id" in biz.columns, "avg_engagement_rate" in biz.columns)
print([c for c in biz.columns if "cluster" in c.lower() or "engagement" in c.lower()])

# --- Code cell 25 ---
biz.sort_values(["kmeans_cluster", "avg_engagement_rate"], ascending=[True, False]) \
   .groupby("kmeans_cluster").head(5) 

# --- Markdown cell 26 ---
# | Cluster                                            | Meaning                                                                                                                                                                                                                                                                                                                                                                                    | Recommendation                                                                                                                                                                                                                                                                                                 |
# | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Cluster 0 – Influencer / Media Reel Accounts**   | This cluster contains mostly influencer and media-style accounts with very large follower counts and fully reel-based content. High-performing accounts in this cluster achieve stronger engagement through more emotionally engaging content rather than relying only on patriotic or location-heavy themes.                                                                              | Focus on more engaging storytelling, personal-style reels, and emotionally driven content. Reduce over-reliance on repetitive patriotic/location content. Best posting days appear to be Tuesday, Wednesday, and Thursday.                                                                                     |
# | **Cluster 1 – Mixed Commercial Business Baseline** | This cluster represents the general business baseline and includes fashion pages, gyms, restaurants, and some influencers. Accounts in this cluster rely heavily on reels and promotional content, but performance differs depending on content quality and hashtag usage. High-performing businesses achieve much better reach while using slightly less aggressive promotional behavior. | Continue using reels as the main content type, but improve content quality instead of only increasing promotional posts. Use stronger visual hooks, clearer captions, and better hashtag strategies. Focus more on posting during Monday, Wednesday, Thursday, and Saturday.                                   |
# | **Cluster 2 – Restaurant / Cafe Content Cluster**  | This cluster contains mostly restaurant and cafe pages. The content strategy is more visually oriented, using a mixture of images and reels with high promotional activity. High-performing accounts in this cluster tend to use shorter captions and less aggressive promotional style while still maintaining strong visual appeal.                                                      | Restaurants and cafes should reduce very long promotional captions and focus more on visually appealing food content. Test high-quality image posts alongside reels, since image-based posts showed strong performance in this cluster. Best posting days appear to be Tuesday, Sunday, Friday, and Wednesday. |
# 

# --- Markdown cell 27 ---
# 

# --- Code cell 28 ---
# Export artifacts: coordinates JSON, plots PNGs, notebook as .py
import json
from pathlib import Path

project_root = Path.cwd()
artifacts_dir = project_root / 'notebooks' / 'clustering' / 'artifacts' / 'business_clustering'
json_dir = artifacts_dir / 'json'
scripts_dir = artifacts_dir / 'scripts'
photos_dir = project_root / 'notebooks' / 'clustering' / 'photos' / 'business_clustering'
for d in [artifacts_dir, json_dir, scripts_dir, photos_dir]:
    d.mkdir(parents=True, exist_ok=True)

if 'biz' not in globals():
    raise ValueError('`biz` dataframe not found. Run business clustering cells first.')

kmeans_col = 'kmeans_cluster' if 'kmeans_cluster' in biz.columns else ('cluster_id' if 'cluster_id' in biz.columns else None)
dbscan_col = 'dbscan_cluster' if 'dbscan_cluster' in biz.columns else None
if kmeans_col is None or dbscan_col is None:
    raise ValueError('Missing cluster columns. Ensure KMeans and DBSCAN cells were executed.')

if not {'tsne_1','tsne_2'}.issubset(biz.columns):
    raise ValueError('Missing tsne_1/tsne_2 in biz. Run t-SNE visualization cell first.')

dbscan_eps = globals().get('DBSCAN_EPS', None)
dbscan_min_samples = globals().get('DBSCAN_MIN_SAMPLES', None)
best_k = globals().get('best_k', None)

export_cols = ['business_name','sector','tsne_1','tsne_2',kmeans_col,dbscan_col]
records = biz[export_cols].to_dict(orient='records')
payload = {
  'project':'business_clustering',
  'coordinates_type':'tsne_2d',
  'dbscan_params':{'eps': dbscan_eps, 'min_samples': dbscan_min_samples},
  'kmeans_params':{'n_clusters': best_k},
  'cluster_columns': {'kmeans': kmeans_col, 'dbscan': dbscan_col},
  'n_rows': int(len(biz)),
  'records': records
}
json_path = json_dir / 'business_cluster_coordinates.json'
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved JSON:', json_path)

# Save current figures if open (run right after plotting cells)
try:
    import matplotlib.pyplot as plt
    figs = [plt.figure(i) for i in plt.get_fignums()]
    if len(figs) >= 1:
        figs[-2 if len(figs) >= 2 else -1].savefig(photos_dir / 'business_kmeans_clusters.png', dpi=200, bbox_inches='tight')
    if len(figs) >= 2:
        figs[-1].savefig(photos_dir / 'business_dbscan_clusters.png', dpi=200, bbox_inches='tight')
except Exception as e:
    print('Figure export note:', e)

# Save notebook-as-py
nb_path = project_root / 'notebooks' / 'clustering' / '04_business_clustring.ipynb'
nb = json.loads(nb_path.read_text(encoding='utf-8'))
lines = ['# Auto-exported from 04_business_clustring.ipynb\n', '# Generated from notebook run context\n\n']
for i,cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'markdown':
        lines.append(f'# --- Markdown cell {i} ---\n')
        for ln in ''.join(cell.get('source', [])).split('\n'):
            lines.append('# ' + ln + '\n')
        lines.append('\n')
    elif cell.get('cell_type') == 'code':
        lines.append(f'# --- Code cell {i} ---\n')
        code = ''.join(cell.get('source', []))
        lines.append(code if code.endswith('\n') else code + '\n')
        lines.append('\n')
py_path = scripts_dir / 'business_clustering_pipeline.py'
py_path.write_text(''.join(lines), encoding='utf-8')
print('Saved script:', py_path)

manifest = {
  'project':'business_clustering',
  'artifacts':{
     'json': str(json_path),
     'kmeans_plot': str(photos_dir / 'business_kmeans_clusters.png'),
     'dbscan_plot': str(photos_dir / 'business_dbscan_clusters.png'),
     'script': str(py_path)
  }
}
manifest_path = artifacts_dir / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Saved manifest:', manifest_path)

