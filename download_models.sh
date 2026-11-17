#!/bin/bash
# Downloads all models referenced by the manifest (v2 active + v1 for rollback) into the HF cache.
set -e; cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true; . .venv/bin/activate
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements.txt
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["fakespot-ai/roberta-base-ai-text-detection-v1", "intfloat/e5-small-v2",            # v2 active
          "distilgpt2", "Hello-SimpleAI/chatgpt-detector-roberta", "sentence-transformers/all-MiniLM-L6-v2"]:  # v1 (rollback)
    print("downloading", m); snapshot_download(m)
PY
