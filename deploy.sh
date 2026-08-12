#!/usr/bin/env bash
# Pod bootstrap — run on a fresh RunPod PyTorch pod after cloning the repo.
#   export HF_TOKEN=... ANTHROPIC_API_KEY=... OPENAI_API_KEY=...   (or scp .env)
#   bash deploy.sh
set -euo pipefail
cd "$(dirname "$0")"

# load ../.env or ./.env if present (keys never printed)
for f in ../.env .env; do
  if [ -f "$f" ]; then
    set -a; . <(sed -E 's/^ *(export +)?([A-Z_]+)[:= ] *"?([^"]*)"?$/\2=\3/' "$f"); set +a
  fi
done
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-${CLAUDE_API_KEY:-}}"

echo "[1/4] installing deps ..."
pip install -q -r requirements.txt

echo "[2/4] checking HF access to gated Llama ..."
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.list_repo_files("meta-llama/Llama-3.1-8B-Instruct")
print("      Llama access OK, logged in as", api.whoami()["name"])
PY

echo "[3/4] checking judge APIs ..."
python - <<'PY'
import os
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY missing"
assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY missing"
print("      keys present")
PY

echo "[4/4] GPU smoke test on Model A (downloads ~16GB on first run) ..."
python scripts/smoke_test.py --model-key A

echo ""
echo "DEPLOY OK. Next: python scripts/m1_gate.py --model-key A"
