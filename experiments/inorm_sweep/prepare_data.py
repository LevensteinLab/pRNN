"""
Generate the trajectory dataset used by examples/trainNet.py's DataLoader, in the
same folder layout (<datadir>/<env>-<agent>-<actenc>), so that sweep runs only
need to read it.
"""

import argparse

import numpy as np
import torch

from prnn.utils.agent import create_agent
from prnn.utils.data import create_dataloader
from prnn.utils.env import make_env

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="LRoom-18x18-v0", type=str)
parser.add_argument("--envPackage", default="farama-minigrid", type=str)
parser.add_argument("--agent", default="RandomActionAgent", type=str)
parser.add_argument("--actenc", default="SpeedHD", type=str)
parser.add_argument("--datadir", default="Data", type=str)
parser.add_argument("--dataNtraj", default=10240, type=int)
parser.add_argument("--seqdur", default=500, type=int)
parser.add_argument("--seed", default=8, type=int)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

env = make_env(args.env, args.envPackage, args.actenc)
agent = create_agent(args.env, env, args.agent)
create_dataloader(env, agent, args.dataNtraj, args.seqdur, args.datadir, generate=True)
