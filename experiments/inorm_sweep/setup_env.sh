#!/usr/bin/env bash
# One-off: create the Python environment for pRNN + I-Norm on the cluster.
# Run on a login node from anywhere:
#
#   bash experiments/inorm_sweep/setup_env.sh
#
# Optional env vars: REPO_ROOT (default: this repo), VENV (default $REPO_ROOT/.venv).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV="${VENV:-$REPO_ROOT/.venv}"

cd "$REPO_ROOT"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 "$VENV"
  PIP=(uv pip install --python "$VENV/bin/python")
else
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  PIP=("$VENV/bin/python" -m pip install)
fi

# setup.py pins numpy<2, pynapple, gymnasium; the rest is needed by trainNet/W&B
"${PIP[@]}" -e . torch torchvision minigrid wandb tqdm seaborn ratinabox pillow

"$VENV/bin/python" - <<'EOF'
import torch, minigrid, pynapple, wandb
from prnn.utils.predictiveNet import PredictiveNet, CELL_TYPES
assert "INormRNNCell" in CELL_TYPES
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("environment OK")
EOF
