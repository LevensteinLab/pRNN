import math
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from PIL import Image

from prnn.utils.general import saveFig
from prnn.examples.Miniworld.VAE import run_random_walk
# from prnn.examples.RatEnvironment import make_rat_env

env = gym.make(
            "MiniWorld-LRoom-v0",
            view="agent",
            render_mode="rgb_array",
            obs_width=64,
            obs_height=64,
            window_width=64,
            window_height=64,
            max_episode_steps=math.inf,
        )

# riab_env = make_rat_env("RiaB-LRoom")
# agent = MiniworldRandomAgent(riab_env)

env.reset()
obs = env.render_obs() #.transpose(2, 0, 1)
render = env.render()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.imshow(obs)
saveFig(fig, "obs", savepath='Figures/test')

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.imshow(render)
saveFig(fig, "render", savepath='Figures/test')


# render = renders[0]
# for i in range(len(renders)-1):
#     ag = renders[i+1][..., 0] > 140
#     render[ag] = renders[i+1][ag]

# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# agent.plot_trajectory(t_start=0, t_end=200, fig=fig, ax=ax,color="changing")

# saveFig(fig, "riab", savepath='Figures/test')

# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# ax.imshow(render)

# saveFig(fig, "topview", savepath='Figures/test')