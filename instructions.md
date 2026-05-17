# Palestine SME Social Media Intelligence Platform (Student-Project Friendly, Production-Like)

## 1. Project Purpose
This project is an end-to-end analytics platform for Palestinian SMEs' organic social media posts. It supports:
1. Uploading a CSV dataset
2. Validating schema and data quality
3. Cleaning and standardizing data
4. Engineering KPIs for engagement performance
5. Running data mining models and analyses:
   - EDA summaries
   - Post clustering
   - Business clustering
   - PCA (dimensionality reduction)
   - Association rules (Apriori)
   - Weekly trends + simple forecasts
   - Anomaly detection (rules + IsolationForest)
   - Network analysis (co-occurrence graph)
   - Explainable recommendations
6. Saving all outputs to local filesystem storage
7. Exposing results via backend APIs for a React dashboard

## 2. Tech Stack
Backend:
- Python 3.11+
- FastAPI + Uvicorn
- pandas, numpy
- scikit-learn
- mlxtend (association rules)
- networkx (network analysis)
- pydantic (schemas)
- python-multipart (file uploads)
- statsmodels (optional; not required in this starter)

Frontend:
- React + TypeScript + Vite
- Tailwind CSS
- Recharts (charts)
- Axios (API calls)
- React Router (pages)

Storage:
- Local filesystem only
- Uploads and outputs live under `backend/storage/`

## 3. Folder Structure
```
palestine-sme-social-intelligence/
  instructions.md
  README.md
  .gitignore
  docker-compose.yml

  backend/
    Dockerfile
    requirements.txt
    .env.example
    run_backend.sh
    app/
      main.py
      config.py
      schemas.py
      api/
        upload.py
        dashboard.py
        mining.py
        exports.py
      services/
        validation.py
        cleaning.py
        kpi_engineering.py
        eda.py
        clustering.py
        dimensionality.py
        association_rules.py
        time_series.py
        anomaly_detection.py
        network_analysis.py
        recommendations.py
        pipeline.py
      utils/
        file_utils.py
        constants.py
    storage/
      raw/
      cleaned/
      outputs/
      reports/

  frontend/
    Dockerfile
    package.json
    index.html
    vite.config.ts
    tailwind.config.js
    postcss.config.js
    src/
      main.tsx
      App.tsx
      api/client.ts
      components/
        Layout.tsx
        MetricCard.tsx
        DataTable.tsx
        ChartCard.tsx
      pages/
        UploadPage.tsx
        KpiDashboard.tsx
        ContentPerformance.tsx
        ClusteringPage.tsx
        RulesPage.tsx
        TrendsPage.tsx
        NetworkPage.tsx
        RecommendationsPage.tsx
      types.ts

  data/
    sample_synthetic_posts.csv
    data_dictionary.csv
```

## 4. How To Install Backend (Windows PowerShell)
From repo root:
```powershell
cd .\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks venv activation, run PowerShell as Admin once and:
```powershell
Set-ExecutionPolicy RemoteSigned
```

## 5. How To Run Backend
From `backend/` with venv activated:
```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected URLs:
- Backend: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs

## 6. How To Install Frontend
From repo root:
```powershell
cd .\frontend
npm install
```

## 7. How To Run Frontend
From `frontend/`:
```powershell
npm run dev
```

Expected URL:
- Frontend: http://localhost:5173

The frontend reads `VITE_API_BASE_URL` (default: `http://localhost:8000`).

## 8. How To Use The Sample Dataset
1. Start backend and frontend.
2. Open http://localhost:5173
3. Go to **Upload & Data Quality**
4. Upload `data/sample_synthetic_posts.csv`
5. After upload, click **Run Full Pipeline**
6. Browse dashboard pages for results

## 9. Required Dataset Columns
Your CSV must include these raw columns:
- business_name
- sector
- followers_count
- post_date
- posting_hour
- day_of_week
- month
- post_type
- caption_text
- caption_length
- hashtags_count
- emoji_count
- likes_count
- comments_count
- views_count
- language
- CTA_present
- promo_post
- discount_percent
- mentions_location
- religious_theme
- patriotic_theme
- arabic_dialect_style

## 10. How Upload Works
Endpoint: `POST /api/upload`
1. Backend receives a CSV (`multipart/form-data`)
2. Backend generates a `dataset_id` (UUID)
3. Saves raw CSV to: `backend/storage/raw/{dataset_id}/raw.csv`
4. Validates schema + basic numeric constraints
5. Writes validation report JSON to: `backend/storage/reports/{dataset_id}/validation_report.json`
6. Returns:
   - `dataset_id`
   - `validation_report`

## 11. How The Pipeline Works
Endpoint: `POST /api/run-pipeline/{dataset_id}`

The pipeline loads the raw CSV and runs:
1. Validation (again, to guarantee reproducibility)
2. Cleaning
3. KPI engineering
4. EDA summaries
5. Post clustering (KMeans, k=3..6, silhouette selection)
6. Business clustering (aggregated KMeans)
7. PCA for posts and businesses
8. Association rules (Apriori) + business value score
9. Trends + forecast
10. Anomaly detection (rule-based + IsolationForest)
11. Network analysis (co-occurrence graph)
12. Recommendations (explainable, evidence-based)

Outputs are saved under:
`backend/storage/outputs/{dataset_id}/`

If a step cannot run (e.g., too little data), it will return a useful message and save an empty-but-valid output file where applicable, rather than crashing.

## 12. What Each Output File Means
All under `backend/storage/outputs/{dataset_id}/`:
- `clean_dataset.csv`: cleaned and standardized dataset with parsed dates and filled defaults
- `kpis.csv`: dataset with engineered KPI columns
- `eda_summary.json`: grouped engagement summaries and leaderboards
- `post_clusters.csv`: post-level cluster assignments + human-readable labels
- `business_clusters.csv`: business-level cluster assignments + names
- `post_pca.csv`: PCA 2D projection for posts
- `business_pca.csv`: PCA 2D projection for businesses
- `transactions.csv`: transaction items per post used for mining
- `association_rules.csv`: association rules with support/confidence/lift/etc
- `business_value_rules.csv`: rules with an additional business value score
- `weekly_trends.csv`: weekly metrics + shares
- `business_momentum.csv`: momentum per business based on recent vs previous posts
- `forecast.csv`: simple forecasts on weekly metrics
- `anomalies.csv`: flagged outliers/spikes and notes
- `network_nodes.csv`: network node metrics (centralities, pagerank, etc.)
- `network_edges.csv`: network co-occurrence edges with weights
- `network_summary.json`: small network summary
- `recommendations.csv`: explainable recommendations per business
- `pipeline_summary.json`: pipeline run metadata and step statuses

## 13. API Endpoint List
Health:
- `GET /health`

Upload:
- `POST /api/upload`

Pipeline:
- `POST /api/run-pipeline/{dataset_id}`

Dashboard reads:
- `GET /api/dashboard/kpis/{dataset_id}`
- `GET /api/dashboard/content-performance/{dataset_id}`
- `GET /api/dashboard/clustering/{dataset_id}`
- `GET /api/dashboard/rules/{dataset_id}`
- `GET /api/dashboard/trends/{dataset_id}`
- `GET /api/dashboard/network/{dataset_id}`
- `GET /api/dashboard/recommendations/{dataset_id}`

Exports:
- `GET /api/exports/{dataset_id}/{file_name}`

## 14. Dashboard Pages (Frontend)
1. Upload & Data Quality:
   - Upload CSV
   - View validation report
   - Trigger full pipeline
2. KPI Dashboard:
   - KPI metrics, totals, top sectors and businesses
3. Content Performance:
   - Charts comparing performance by post_type, language, CTA, promo, dialect, and posting hour
4. Clustering & PCA:
   - Post clusters table
   - Business clusters table
   - PCA scatter plot (posts)
5. Association Rules:
   - Rules table with confidence/lift/business value score
6. Trends & Forecasting:
   - Weekly trends line charts
   - Business momentum table
   - Forecast chart
7. Network Analysis:
   - Node and edge tables
   - Centrality ranking
   - Network summary
8. Recommendations:
   - Recommendation list/table
   - Priority filters
   - Evidence shown clearly

## 15. Team Split (4 People)
Person 1:
- Data quality (validation + cleaning)
- KPI engineering + EDA
- Upload page + KPI dashboard

Person 2:
- Post clustering
- Business clustering
- PCA
- Clustering dashboard

Person 3:
- Association rules + business value score
- Recommendation generation
- Rules + recommendations pages

Person 4:
- Time-series trends + forecasting
- Anomalies
- Network analysis
- Trends + network pages
- Integration glue, pipeline orchestration

## 16. Four-Week Timeline
Week 1:
- Repo setup, dataset schema, upload flow, validation + cleaning
- KPI engineering + initial EDA
- Frontend layout + routing + upload page

Week 2:
- Clustering (posts + business) + PCA
- Frontend clustering + KPI pages
- Save outputs + ensure export endpoint works

Week 3:
- Association rules + business value scoring
- Recommendations logic
- Frontend rules + recommendations pages

Week 4:
- Trends + forecasting
- Anomalies + network analysis
- End-to-end pipeline QA
- Demo script + report + packaging deliverables

## 17. Troubleshooting
- CORS errors:
  - Ensure backend is running on `http://localhost:8000`
  - Ensure frontend uses `VITE_API_BASE_URL=http://localhost:8000`
- Upload errors:
  - Ensure CSV has all required columns
  - Check validation report in the UI
- Pipeline errors:
  - Re-run with the sample dataset to confirm environment
  - Inspect `backend/storage/outputs/{dataset_id}/pipeline_summary.json`
- Windows path issues:
  - Use PowerShell and run commands from the correct folder
- “Module not found”:
  - Confirm venv is activated and `pip install -r requirements.txt` was run
- Node issues:
  - Delete `frontend/node_modules` and rerun `npm install`

## 18. Git Workflow
Recommended:
1. `main` branch is stable
2. Each person works on a feature branch:
   - `feature/upload-validation`
   - `feature/clustering`
   - `feature/rules-reco`
   - `feature/trends-network`
3. Open PRs, request review, merge to `main`
4. Keep PRs small and testable

## 19. Final Deliverables Checklist
- Working backend API + docs
- Working frontend dashboard with 8 pages
- Sample dataset + data dictionary
- Pipeline outputs saved per dataset_id
- Demo script (upload -> run pipeline -> show dashboards)
- Final report with methods + findings + screenshots

## Exact Commands Summary
Backend install:
```powershell
cd .\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Backend run:
```powershell
cd .\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend install:
```powershell
cd .\frontend
npm install
```

Frontend run:
```powershell
cd .\frontend
npm run dev
```

Expected URLs:
- backend: http://localhost:8000
- API docs: http://localhost:8000/docs
- frontend: http://localhost:5173

