#%load_ext autoreload
#%autoreload 2

import numpy as np
import pandas as pd
import torch
import random
import math
import matplotlib
matplotlib.use('Agg')

from torch import nn

from prnn.utils.data import generate_trajectories, create_dataloader
from prnn.utils.env import make_env
from prnn.utils.agent import RandomActionAgent, RandomHDAgent, RatInABoxAgent
from prnn.utils.predictiveNet import PredictiveNet

print("matplotlib version: ", matplotlib.__version__)

# env_key is the one you put in RatEnvironment.py
# package is the one you put in env.py

env = make_env(env_key='cheeseboard', package='ratinabox_colors_Reward', act_enc='ContSpeedOnehotHD', FoV_params={"spatial_resolution": 0.05,
                           "angle_range": [0, 30],
                           "distance_range": [0.0, 1.2],
                           "beta": 10,
                           "walls_occlude": False
                           }) #add FoV here
agent = RatInABoxAgent('_')


# to test if you collect observations correctly
# obs should be a tuple of two tensors
# act should be a tensor

obs, act, state, render = env.collectObservationSequence(agent, 10)

prednet = PredictiveNet(env, pRNNtype='multRNN_5win_i01_o01')

# to test if pRNN works correctly with these observations

obs_pred, _, _ = prednet.predict(obs, act)


prednet.plotObservationSequence(obs, render, obs_pred, state, timesteps=range(4,10))

# when you're sure that everything's set, you may want to generate some data with this
# it will save a bunch of trajectories in the folder you specify, and then you can train pRNNs faster
# withput having to collect data every time

#generate_trajectories(env, agent, n_trajs=10240, seq_length=1000, folder='Data') #TODO: update folder

%run trainNet_prnn.py --savefolder='test/' --pRNNtype='multRNN_5win_i01_o01' \
        --sparsity=0.1 --mean_std_ratio=1 --eg_weight_decay=1e-8 --eg_lr=2e-3 \
        --env='cheeseboard' --env_package='ratinabox_colors_Reward' --agent='RatInABoxAgent' \
        --seqdur=1000 --lr=2e-3 --numepochs=6 --numtrials=1024 --hiddensize=500 --noisestd=0.05 \
        --bias_lr=0.2 --trainBias --ntimescale=2 --actenc='ContSpeedOnehotHD' --batch_size=8 --datasetSize=10240 \
        --datasetfolder='/Data' --namext='ContSpeedOnehotHD' -s=8 --saveTrainData #TODO: update data
