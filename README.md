# Predictive Replay

This project is focused on modelling rat exploratory behavior in the
Novel Object Recognition (NOR) paradigm, leveraging curiosity as an
intrinsic reward signal for RL.

## Project Setup

This project is managed using [`uv`](https://docs.astral.sh/uv/).
See [workspace documentation](https://docs.astral.sh/uv/concepts/projects/workspaces).

This project utilizes a custom [Minigrid](https://github.com/SabrinaDu7/minigrid) package.

```bash
# Clone the pRNN repo and the Minigrid repo as a submodule
# Use git@github.com:SabrinaDu7/pRNN.git if SSH enabled.
git clone --recurse-submodules -j8 https://github.com/SabrinaDu7/pRNN.git
cd pRNN/

# Create and activate venv
uv venv —python 3.10
source .venv/bin/activate

# Download dependencies from pyproject.toml
uv sync

# Test imports (test from Dan's original repo)
uv run test/test_imports.py
```

## Running training

Default values for the training pipeline can be found in ```trainNet.py```. Here is an example run:

```bash
uv run trainNet.py --pRNNtype thcycRNN_5win_full --savefolder thcycRNN_5win_full --noDataLoader
```
