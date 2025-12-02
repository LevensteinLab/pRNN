Training & Loading pRNNs on a Cluster
=====================================

We have also provided a script that allows you to train a pRNN from the command line, namely ``trainNet.py``. This script accepts all arguments that specify a network itself. It also accepts flags to specify whether you'd like to train a new network or load an existing one, as well as whether or not to use a Dataloader. 

Bash scripts may call this training script to automate training of networks on a high performance computing (HPC) cluster. This will be required for larger networks. For example: 

.. code-block:: bash

    #!/bin/bash

    #SBATCH --job-name=prnn_arch
    #SBATCH --output=logs/trainNet/prnn_%j.out
    #SBATCH --error=logs/trainNet/prnn_%j.err
    #SBATCH --time=3:00:00

    cd ~
    module load miniconda
    conda activate base39
    source ~/venvs/PredictiveReplay_39/bin/activate

    cd project/pRNN
    python trainNet.py --savefolder='examplenet/' --lr=2e-3 --numepochs=6 --batchsize=16 --pRNNtype='thRNN_5win' --actenc='SpeedHD'