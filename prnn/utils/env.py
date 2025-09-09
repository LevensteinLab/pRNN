#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 22:00:57 2021

@author: dl2820
"""
import numpy as np
import math
import matplotlib.pyplot as plt

from gymnasium import spaces
from gymnasium.core import ObservationWrapper

from prnn.utils.Shell import *
from prnn.examples.RatEnvironment import make_rat_env, FoV_params_default, Grid_params_default

def make_env(env_key, package='gym-minigrid', act_enc='OnehotHD',
             speed=0.2, thigmotaxis=0.2, HDbins=12, wrap=True,
             seed=42, FoV_params=FoV_params_default,
             Grid_params=Grid_params_default, encoder=None):

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

    elif package=='ratinabox_vision':        
        env = make_rat_env(env_key)
        env = RiaBVisionShell(env, act_enc, env_key, speed,
                              thigmotaxis, HDbins, FoV_params)

    elif package=='ratinabox_remix':        
        env = make_rat_env(env_key)
        env = RiaBRemixColorsShell(env, act_enc, env_key, speed,
                                   thigmotaxis, HDbins, FoV_params)

    elif package=='ratinabox_grid':        
        env = make_rat_env(env_key)
        env = RiaBGridShell(env, act_enc, env_key, speed,
                            thigmotaxis, HDbins, Grid_params)

    elif package=='ratinabox_colors_grid':        
        env = make_rat_env(env_key)
        env = RiaBColorsGridShell(env, act_enc, env_key, speed,
                                  thigmotaxis, HDbins, FoV_params, Grid_params)
        
    elif package=='miniworld_vae':
        import gymnasium as gym
        import miniworld
        env = gym.make(
                    env_key,
                    view="agent",
                    render_mode="rgb_array",
                    obs_width=64,
                    obs_height=64,
                    window_width=64,
                    window_height=64,
                    max_episode_steps=math.inf,
        )
        env.reset(seed=seed)
        env = MiniworldVAEShell(env, act_enc, env_key,
                                encoder, HDbins)
        
    elif package=='miniworld_prnn_ae':
        import gymnasium as gym
        import miniworld
        env = gym.make(
                    env_key,
                    view="agent",
                    render_mode="rgb_array",
                    obs_width=64,
                    obs_height=64,
                    window_width=64,
                    window_height=64,
                    max_episode_steps=math.inf,
        )
        env.reset(seed=seed)
        env = MiniworldShell(env, act_enc, env_key, HDbins)

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