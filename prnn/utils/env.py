#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 22:00:57 2021

@author: dl2820
"""
import numpy as np
import matplotlib.pyplot as plt

from prnn.utils.Shell import *
from prnn.examples.RatEnvironment import make_rat_env



def make_env(env_key, package='gym-minigrid', act_enc='OnehotHD',
             speed=0.2, thigmotaxis=0.2, HDbins=12, wrap=True, seed=42):

    # For different types/names of the env, creates the env, makes necessary adjustments, then wraps it in a corresponding shell
    if package=='gym-minigrid':
        import gym
        import gym_minigrid
        from gym_minigrid.wrappers import RGBImgPartialObsWrapper_HD
        if wrap:
            env = RGBImgPartialObsWrapper_HD(gym.make(env_key),tile_size=1)
        else:
            env = gym.make(env_key)
        env.reset()
        env = GymMinigridShell(env, act_enc, env_key)

    elif package=='farama-minigrid':
        import gymnasium as gym
        import minigrid
        from minigrid.wrappers import RGBImgPartialObsWrapper_HD
        if wrap:
            env = RGBImgPartialObsWrapper_HD(gym.make(env_key),tile_size=1)
        else:
            env = gym.make(env_key)
        env.reset(seed=seed)
        env = FaramaMinigridShell(env, act_enc, env_key)

    elif package=='ratinabox':        
        env = make_rat_env(env_key)
        env = RatInABoxShell(env, act_enc, env_key, speed, thigmotaxis, HDbins)

    elif package=='ratinabox_remix':        
        env = make_rat_env(env_key)
        env = RiaBRemixColorsShell(env, act_enc, env_key, speed, thigmotaxis, HDbins)

    else:
        raise NotImplementedError('Package is not supported yet or its name is incorrect')
    
    return env


# TODO: is obsolete? Remove and then remove the notion of highlight from render?
def plot_env(env, highlight=True):
    
    gridView = env.render(highlight=highlight)
    
    plt.figure()
    plt.imshow(gridView)
    plt.xticks([])
    plt.yticks([])