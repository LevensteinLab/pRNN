#!/usr/bin/env bash
#SBATCH --array=0-59%10     # 60 sweep runs, at most 10 at a time
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|lovelace|hopper"
#SBATCH --mem=16GB
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00
#SBATCH --output=sbatch_out/prnn_sweep_%A_%a.out
#SBATCH --error=sbatch_err/prnn_sweep_%A_%a.err
#SBATCH --job-name=prnn_sweep
#
# One W&B sweep run per array task. Works for sweep_inorm.yaml and
# sweep_layernorm_baseline.yaml alike:
#
#   SWEEP_ID=<entity>/<project>/<sweep_id> sbatch experiments/inorm_sweep/sweep_agent.sh
#
# Optional env vars: REPO_ROOT (default $HOME/pRNN), VENV (default $REPO_ROOT/.venv).
# Run experiments/inorm_sweep/prepare_data.sh once first, so the trajectory
# dataset in $REPO_ROOT/Data already exists and jobs only copy it.
set -euo pipefail

: "${SWEEP_ID:?Set SWEEP_ID=<entity>/<project>/<sweep_id> (printed by 'wandb sweep')}"
REPO_ROOT="${REPO_ROOT:-$HOME/pRNN}"
VENV="${VENV:-$REPO_ROOT/.venv}"

cd "$REPO_ROOT"
source "$VENV/bin/activate"

# trainNet.py copies the dataset to fast node-local storage before training
export PRNN_DATA_TMPDIR="${SLURM_TMPDIR:-}"
export MPLBACKEND=Agg

echo "sweep=$SWEEP_ID task=${SLURM_ARRAY_TASK_ID:-0} host=$(hostname)"
wandb agent --count 1 "$SWEEP_ID"
