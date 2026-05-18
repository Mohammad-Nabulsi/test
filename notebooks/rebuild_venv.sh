#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v py.exe >/dev/null 2>&1; then
  PYTHON_CMD=(py.exe -3)
elif command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_CMD=(python.exe)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
else
  echo "No Windows Python found. Install Python and make sure py.exe or python.exe is in PATH." >&2
  exit 1
fi

if [ -d ".venv" ]; then
  echo "Removing existing .venv ..."
  rm -rf .venv
fi

echo "Creating fresh .venv ..."
"${PYTHON_CMD[@]}" -m venv .venv

VENV_PY=".venv/Scripts/python.exe"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt
"$VENV_PY" -m ipykernel install --user --name dm-notebooks --display-name "Python (.venv - notebooks)"

echo "Done. New environment is ready at $SCRIPT_DIR/.venv"
