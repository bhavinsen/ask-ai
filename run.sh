#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.10}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON=python3
fi

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt
  python -m src.ingest
else
  source .venv/bin/activate
fi

streamlit run app.py
