#!/usr/bin/env bash
#SBATCH --array=0-3
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|lovelace|hopper"
#SBATCH --mem=16GB
#SBATCH --cpus-per-gpu=4
#SBATCH --time=12:00:00
#SBATCH --output=sbatch_out/prnn_pilot_%A_%a.out
#SBATCH --error=sbatch_err/prnn_pilot_%A_%a.err
#SBATCH --job-name=prnn_pilot
#
# Full-size pilot before the sweep (6 epochs each):
#   0: I-Norm, default inhibitory lrs (mean 3e-3 / var 1e-1)
#   1: I-Norm, fast inhibition (mean 1e-2 / var 1)
#   2: I-Norm, RNN_EI_Network fmnist defaults (1e-5 / 1e-6): inhibition lags
#   3: LayerNorm baseline
# Compare epoch_loss, epoch_aux_loss, epoch_ln_cos and epoch_*_time_s in W&B
# (group "pilot") to check the inhibition keeps up and to set --time for the sweep.
#
#   sbatch experiments/inorm_sweep/pilot.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/pRNN}"
VENV="${VENV:-$REPO_ROOT/.venv}"
WANDB_PROJECT="${WANDB_PROJECT:-prnn-inorm}"
NUMEPOCHS="${NUMEPOCHS:-6}"

cd "$REPO_ROOT"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PRNN_DATA_TMPDIR="${SLURM_TMPDIR:-}"
export MPLBACKEND=Agg
export WANDB_RUN_GROUP=pilot

i=${SLURM_ARRAY_TASK_ID:-0}
case $i in
  0) cell=INormRNNCell;     extra=(--lr_norm_mean=3e-3 --lr_norm_var0=1e-1); name=inorm_default ;;
  1) cell=INormRNNCell;     extra=(--lr_norm_mean=1e-2 --lr_norm_var0=1);    name=inorm_fast ;;
  2) cell=INormRNNCell;     extra=(--lr_norm_mean=1e-5 --lr_norm_var0=1e-6); name=inorm_slow ;;
  3) cell=LayerNormRNNCell; extra=();                                       name=layernorm ;;
  *) echo "unknown task $i"; exit 1 ;;
esac
echo "pilot task=$i cell=$cell ${extra[*]:-}"

python examples/trainNet.py \
  --pRNNtype=Masked --cell="$cell" --k=5 --hidden_size=500 \
  --env=LRoom-18x18-v0 --envPackage=farama-minigrid --actenc=SpeedHD \
  --numepochs="$NUMEPOCHS" --numtrials=1024 --seed=8 \
  --savefolder=pilot/ --namext="$name" --no-saveTrainData \
  --wandb --wandb_project="$WANDB_PROJECT" \
  ${extra[@]+"${extra[@]}"}
