##
# TRAINING
##
thRNN ENV FOLDER:
    uv run trainNet.py --pRNNtype thRNN_5win --env {{ENV}} --savefolder {{FOLDER}} --noDataLoader

thcycRNN FOLDER:
    uv run trainNet.py --pRNNtype thcycRNN_5win --savefolder {{FOLDER}} --noDataLoader

# The first positional arg (e.g., thRNN_5win)  is the pRNN type
thRNN-mila:
    sbatch --output=$SCRATCH/pRNN/%x.out --error=$SCRATCH/pRNN/%x.err pRNN.sh thRNN_5win

thcycRNN-mila:
    sbatch --output=$SCRATCH/pRNN/%x.out --error=$SCRATCH/pRNN/%x.err pRNN.sh thcycRNN_5win

##
# FORMATTING AND TESTING
##
lint:
    uv run ruff format .

test:
    uv run -m pytest -m "not slow"