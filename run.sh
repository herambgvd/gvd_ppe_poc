#!/usr/bin/env bash
# Launch the AI-Powered PPE Detection System on the GPU.
# The venv holds torch 2.11.0+cu128 (Blackwell/RTX 5070, sm_120 compatible).
set -e
cd "$(dirname "$0")"

# Force GPU 0 explicitly (config default "auto" also resolves to cuda:0).
export PPE_DEVICE="${PPE_DEVICE:-0}"
export PPE_HOST="${PPE_HOST:-0.0.0.0}"
export PPE_PORT="${PPE_PORT:-5000}"

exec .venv/bin/python app.py
