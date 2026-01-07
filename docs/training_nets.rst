Training & Loading pRNNs on a Cluster
=====================================

We have also provided an example script that allows you to train a pRNN from the command line, namely ``trainNet.py``. This script accepts all arguments that specify a network itself. It also accepts flags to specify whether you'd like to train a new network or load an existing one, as well as whether or not to use a Dataloader. You'll want to modify this to fit the specific needs of your project.

Bash scripts may call this training script to automate training of networks on a high performance computing (HPC) cluster. This will be required for larger networks. For example: 

.. code-block:: bash

    #!/bin/bash

    #SBATCH --job-name=prnn_training
    #SBATCH --output=logs/trainNet/prnn_%j.out
    #SBATCH --error=logs/trainNet/prnn_%j.err
    #SBATCH --time=24:00:00
    #SBATCH --partition=gpu
    #SBATCH --gpus=1
    #SBATCH --mem=64GB


    cd ~
    module load miniconda
    conda activate base39
    source ~/venvs/PredictiveReplay_39/bin/activate

    cd project/pRNN
    python trainNet.py --savefolder='examplenet/' --lr=2e-3 --numepochs=500 --numtrials=1024 --batchsize=16 --pRNNtype='Masked' --actenc='SpeedHD' --k=5 

Training with 500 epochs and 1024 trials may take several hours... Make sure to use a GPU partition if available. Model checkpoints, figures, and training curves will be saved to the specified ``savefolder``. You may also choose to lower the number of epochs and trials for quicker testing.