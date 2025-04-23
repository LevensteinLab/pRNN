#!/bin/bash
#SBATCH --job-name=prnn_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=$HOME/prnn/logs/pipeline_%j.out
#SBATCH --error=$HOME/prnn/logs/pipeline_%j.err

# 1) Activate your venv
VENV_DIR=$HOME/venvs/PredictiveReplay_39
if [ -d "$VENV_DIR" ]; then
  echo "Activating venv at $VENV_DIR"
  source "$VENV_DIR/bin/activate"
else
  echo "ERROR: venv not found at $VENV_DIR" >&2
  exit 1
fi

# 2) (Optional) ensure we can import your prnn package
export PYTHONPATH=$HOME/prnn:$PYTHONPATH

# 3) Go to your project directory
cd $HOME/prnn

# 4) Make sure the logs folder exists
mkdir -p logs

# 5) Run your wrapper (make sure it's executable: chmod +x run_full_pipeline.py)
./run_pipeline.py

