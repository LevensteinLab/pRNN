#!/bin/bash
#SBATCH --job-name=pRNN_thcycRNN_5win_full
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Load modules
module --force purge
module load python/3.10 cuda/12.6.0

# Activate venv
source ~/venvs/venv-pRNN/bin/activate

# Run training script and use positional arg 1 as pRNNtype
export PRNNTYPE=$1
export SAVE_FOLDERNAME=nets
uv run --active trainNet.py --pRNNtype $PRNNTYPE --savefolder $PRNNTYPE --noDataLoader

mv $SAVE_FOLDERNAME/$PRNNTYPE $SCRATCH/pRNN/$SAVE_FOLDERNAME/$PRNNTYPE

# Deactivate venv
deactivate
