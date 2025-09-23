# Predictive Replay

This project is focused on modelling rat exploratory behavior in the
Novel Object Recognition (NOR) paradigm, leveraging curiosity as an
intrinsic reward signal for RL.

## Project Setup

This project is managed using [`uv`](https://docs.astral.sh/uv/).
See [workspace documentation](https://docs.astral.sh/uv/concepts/projects/workspaces).
A `justfile` is defined to automate common tasks like running files with [`just`](https://github.com/casey/just).

This project utilizes a custom [Minigrid](https://github.com/SabrinaDu7/minigrid) package.

```bash
# Clone the pRNN repo and the Minigrid repo as a submodule
# Use git@github.com:SabrinaDu7/pRNN.git if SSH enabled.
git clone --recurse-submodules -j8 https://github.com/SabrinaDu7/pRNN.git
cd pRNN/

# Create and activate venv
uv venv --python 3.10
source .venv/bin/activate

# Download dependencies from pyproject.toml
uv sync

# Test imports (test from Dan's original repo)
uv run test/test_imports.py
```

## Running pRNN training

Default values for the training pipeline can be found in ```trainNet.py```. Here is an example run:

```bash
uv run trainNet.py --pRNNtype thcycRNN_5win_full --savefolder thcycRNN_5win_full --noDataLoader
```

Recipes in the `justfile` can also be used to run training scripts. Note that there are recipes that
are meant to be used on the Mila Cluster as well. For example:

```bash
just thRNN-mila
```

## Running RL training

Possible inputs the agent can receive:

- FO: full observation (often used as a positive control)
- PO: partial observation (the same type of input as the pRNN)
- h: the hidden state of the pRNN
- h+PO: the hidden state of the pRNN and a partial observation

## Setup on Mila's cluster

On the login node:

1. Clone the repo: ```git clone --recurse-submodules -j8 https://github.com/SabrinaDu7/pRNN.git```
2. Create a virtual environment in your desired directory. (Ex: ```uv venv —python 3.10 ~/venvs/venv-pRNN```)
3. Activate and sync the venv: ```source ~/venvs/venv-pRNN/bin/activate``` then ```uv sync --active```
4. Venv is ready to be used on compute nodes. You can deactivate it for now: ```deactivate```

When using ```salloc/srun``` or ```sbatch```, you must activate the venv on the compute node
and use the option ```--active``` to use the active venv. Example run command:

```bash
uv run trainNet.py --active --pRNNtype thcycRNN_5win_full --savefolder thcycRNN_5win_full --noDataLoader
```
