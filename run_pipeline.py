#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch
import random
import math
import matplotlib
matplotlib.use('Agg')

from torch import nn

from prnn.utils.data       import generate_trajectories, create_dataloader
from prnn.utils.env        import make_env
from prnn.utils.agent      import RandomActionAgent, RandomHDAgent, RatInABoxAgent
from prnn.utils.predictiveNet import PredictiveNet

#print("matplotlib version: ", matplotlib.__version__)

# 1) Build your env + agent + quick smoke test
env = make_env(env_key='cheeseboard',
               package='ratinabox_colors_Reward',
               act_enc='ContSpeedOnehotHD',
               FoV_params={
                 "spatial_resolution": 0.05,
                 "angle_range": [0, 30],
                 "distance_range": [0.0, 1.2],
                 "beta": 10,
                 "walls_occlude": False,
               })
agent = RatInABoxAgent('_')

obs, act, state, render = env.collectObservationSequence(agent, 10)
prednet = PredictiveNet(env, pRNNtype='multRNN_5win_i01_o01')
obs_pred, _, _ = prednet.predict(obs, act)
prednet.plotObservationSequence(obs, render, obs_pred, state, timesteps=range(4,10))

# 2) (Optional) pre-generate trajectories
generate_trajectories(env, agent, n_trajs=10240, seq_length=1000, folder='Data')

# 3) Now kick off the real training script
import subprocess
cmd = [
    "python", "trainNet_prnn.py",
    "--savefolder",     "test/",
    "--pRNNtype",       "multRNN_5win_i01_o01",
    "--sparsity",       "0.1",
    "--mean_std_ratio", "1",
    "--eg_weight_decay","1e-8",
    "--eg_lr",          "2e-3",
    "--env",            "cheeseboard",
    "--env_package",    "ratinabox_colors_Reward",
    "--agent",          "RatInABoxAgent",
    "--seqdur",         "1000",
    "--lr",             "2e-3",
    "--numepochs",      "6",
    "--numtrials",      "1024",
    "--hiddensize",     "500",
    "--noisestd",       "0.05",
    "--bias_lr",        "0.2",
    "--trainBias",
    "--ntimescale",     "2",
    "--actenc",         "ContSpeedOnehotHD",
    "--batch_size",     "8",
    "--datasetSize",    "10240",
    "--datasetfolder",  "./Data",
    "--namext",         "ContSpeedOnehotHD",
    "-s",               "8",
    "--saveTrainData",
]
subprocess.run(cmd, check=True)
