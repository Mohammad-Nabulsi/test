# Project README

## Quick Start

### Run backend + frontend together (Git Bash)
```bash
(cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000) & (cd frontend/connect-depth-main && npm run dev -- --host 127.0.0.1 --port 5173) & wait
```

### Health checks
- Backend: `http://127.0.0.1:8000/health`
- Frontend: `http://127.0.0.1:5173`

## Safe Merge Flow (Keep Local On Conflict)

Use this when pulling a remote branch while preserving your current local work:

```bash
git branch checkpoint-before-merge-YYYY-MM-DD
git stash push -u -m "before merge <remote-branch>"
git fetch origin
git merge -X ours origin/<remote-branch>
git stash pop
git diff --name-only --diff-filter=U
git status
```

## Conflict Notes

- If `git diff --name-only --diff-filter=U` prints nothing, there are no unresolved conflicts.
- `LF will be replaced by CRLF` is a line-ending warning, not a merge conflict.

## Current Verified State (2026-05-18)

- `origin/dareen+main` merged into `main`.
- `origin/Hana-work` merged into `main`.
- Local uncommitted changes remain in working tree by design.
