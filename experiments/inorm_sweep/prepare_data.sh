#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --output=sbatch_out/prnn_data_%j.out
#SBATCH --error=sbatch_err/prnn_data_%j.err
#SBATCH --job-name=prnn_data
#
# Generate the LRoom trajectory dataset once, before launching sweep agents
# (otherwise parallel jobs would all try to write it at the same time).
#
#   sbatch experiments/inorm_sweep/prepare_data.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/pRNN}"
VENV="${VENV:-$REPO_ROOT/.venv}"

cd "$REPO_ROOT"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python experiments/inorm_sweep/prepare_data.py --datadir Data --dataNtraj 10240 --seqdur 500
