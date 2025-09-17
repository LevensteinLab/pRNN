# Training pRNNs
thRNN-local:
    uv run trainNet.py --pRNNtype thRNN_5win --savefolder thRNN_5win --noDataLoader

thRNN-mila:
    sbatch --output=$SCRATCH/pRNN/%x.out --error=$SCRATCH/pRNN/%x.err pRNN.sh thRNN_5win

thcycRNN-local:
    uv run trainNet.py --pRNNtype thcycRNN_5win --savefolder thcycRNN_5win --noDataLoader

thcycRNN-mila:
    sbatch --output=$SCRATCH/pRNN/%x.out --error=$SCRATCH/pRNN/%x.err pRNN.sh thcycRNN_5win

# Formatting
lint:
    uv run ruff format .