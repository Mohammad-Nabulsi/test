# How To Run

This repo has a Python backend (`backend`) and a Vite frontend (`frontend/connect-depth-main`).

## 1) Setup (Git Bash on Windows)

Run this from the repo root:

```bash
python -m venv .venv && \
source .venv/Scripts/activate && \
python -m pip install --upgrade pip setuptools wheel && \
pip install -r backend/requirements.txt && \
npm --prefix frontend/connect-depth-main install
```

If `python` is not found, use:

```bash
py -3 -m venv .venv
```

## 2) Optional env config (for OpenAI-powered endpoints)

Create `backend/.env`:

```bash
echo 'OPENAI_API_KEY=YOUR_KEY_HERE' > backend/.env
```

## 3) One-liner to run backend + frontend and check both

Run this from the repo root:

```bash
source .venv/Scripts/activate && (cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 >/tmp/backend.log 2>&1) & B=$!; (cd frontend/connect-depth-main && npm run dev -- --host 127.0.0.1 --port 5173 >/tmp/frontend.log 2>&1) & F=$!; sleep 12; curl -fsS http://127.0.0.1:8000/health && echo && curl -fsSI http://127.0.0.1:5173 | head -n 1; kill $B $F
```

Expected checks:
- Backend returns JSON from `/health`.
- Frontend returns `HTTP/1.1 200 OK` (or similar success status line).

## 4) Run normally (without auto-kill)

Terminal 1 (repo root):

```bash
source .venv/Scripts/activate
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 (repo root):

```bash
cd frontend/connect-depth-main
npm run dev -- --host 127.0.0.1 --port 5173
```

## 5) What was configured in this session

- Identified project structure (`backend` + `frontend/connect-depth-main`).
- Confirmed backend dependencies from `backend/requirements.txt`.
- Confirmed backend run command from `backend/run_backend.sh` (`uvicorn app.main:app`).
- Confirmed frontend scripts from `frontend/connect-depth-main/package.json` (`npm run dev`).
- Adjusted virtualenv activation for Git Bash on Windows: `source .venv/Scripts/activate`.
- Provided a single one-liner to start both services, verify both endpoints, then stop both processes.

## 6) Integrated Single-Business APIs (Recommendation, Forecast, Anomalies)

### New endpoints

- `POST /api/generate-rules-single`
- `POST /api/similar-business-recommendations-single`
- `POST /api/benchmark-dashboard-single`
- `POST /api/next-post-recommendations-single`
- `POST /api/forecast/analyze-single`
- `GET /api/forecast/static-single`
- `POST /api/anomalies/analyze-single`
- `GET /api/anomalies/sector-summary-single?sector=restaurants`

### Required files

- `data/vanilla_kpi_dataset.json` (required)
- `data/data_processed.json` (copied when present)
- `data/processed_data.json` (optional, source did not contain it)
- `data/positive_rules.pkl` (recommended)
- `data/negative_rules.pkl` (recommended)
- `backend/_runtime_outputs/forecast_single` (created for forecast outputs)
- `storage/outputs/forecast_single` (created for forecast outputs)

### Smoke test payloads

- Recommendation endpoints:
```json
{"json_path":"data/vanilla_kpi_dataset.json"}
```
- Forecast/Anomalies POST endpoints:
```json
{"uploaded_file_path":"data/vanilla_kpi_dataset.json"}
```

### Stepwise smoke-test checklist

1. Step A (recommendation only):
- Test:
  - `POST /api/generate-rules-single`
  - `POST /api/similar-business-recommendations-single`
  - `POST /api/benchmark-dashboard-single`
  - `POST /api/next-post-recommendations-single`
- Regression re-test:
  - `GET /health`
  - Existing endpoint: `GET /api/recommendations/nonexistent` (expected 404 route-level response)

2. Step B (forecast added):
- Test:
  - `POST /api/forecast/analyze-single`
  - `GET /api/forecast/static-single`
- Regression re-test:
  - All Step A recommendation endpoints
  - `GET /health`
  - `GET /api/recommendations/nonexistent`

3. Step C (anomalies added):
- Test:
  - `POST /api/anomalies/analyze-single`
  - `GET /api/anomalies/sector-summary-single?sector=restaurants`
- Regression re-test:
  - All Step A recommendation endpoints
  - Step B forecast endpoints
  - `GET /health`
  - `GET /api/recommendations/nonexistent`
