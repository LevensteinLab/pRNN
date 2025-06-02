#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 22:00:57 2021

@author: dl2820
"""
import numpy as np
import matplotlib.pyplot as plt

from gymnasium import spaces
from gymnasium.core import ObservationWrapper

from prnn.utils.Shell import *
from prnn.examples.RatEnvironment import make_rat_env



def make_env(env_key, package='gym-minigrid', act_enc='OneHotHD',
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
        import prnn.examples.Lroom
        if wrap:
            env = RGBImgPartialObsWrapper_HD_Farama(gym.make(env_key),tile_size=1)
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


class RGBImgPartialObsWrapper_HD_Farama(ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.
    Including direction information (HD)
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype='uint8'
        )
        self.observation_space.spaces['direction'] = spaces.Discrete(4)

            
    def observation(self, obs):
        env = self.unwrapped

        rgb_img_partial = self.get_frame(tile_size=self.tile_size, agent_pov=True)

        return {
            'mission': obs['mission'],
            'image': rgb_img_partial,
            'direction': obs['direction']
        }