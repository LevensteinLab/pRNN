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
from prnn.environments.RatEnvironment import make_rat_env, config_default


def make_env(env_key, package='gym-minigrid', act_enc='OneHotHD',
             riab_cfg=config_default, HDbins=12, wrap=True,
             seed=42, encoder=None):


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
        import prnn.environments.Lroom
        if wrap:
            env = RGBImgPartialObsWrapper_HD_Farama(gym.make(env_key),tile_size=1)
        else:
            env = gym.make(env_key)
        env.reset(seed=seed)
        env = FaramaMinigridShell(env, act_enc, env_key)

    elif package=='ratinabox_vision':        
        env = make_rat_env(env_key)
        env = RiaBVisionShell(env, act_enc, env_key, HDbins=HDbins,
                              speed=riab_cfg['speed'],
                              thigmotaxis=riab_cfg['thigmotaxis'],
                              FoV_params=riab_cfg['FoV_params'],)

    elif package=='ratinabox_remix':        
        env = make_rat_env(env_key)
        env = RiaBRemixColorsShell(env, act_enc, env_key, HDbins=HDbins,
                                   speed=riab_cfg['speed'],
                                   thigmotaxis=riab_cfg['thigmotaxis'],
                                   FoV_params=riab_cfg['FoV_params'],)

    elif package=='ratinabox_grid':        
        env = make_rat_env(env_key)
        env = RiaBGridShell(env, act_enc, env_key, HDbins=HDbins,
                            speed=riab_cfg['speed'],
                            thigmotaxis=riab_cfg['thigmotaxis'],
                            Grid_params=riab_cfg['Grid_params'],)

    elif package=='ratinabox_colors_grid':        
        env = make_rat_env(env_key)
        env = RiaBColorsGridShell(env, act_enc, env_key, HDbins=HDbins,
                                  speed=riab_cfg['speed'],
                                  thigmotaxis=riab_cfg['thigmotaxis'],
                                  FoV_params=riab_cfg['FoV_params'],
                                  Grid_params=riab_cfg['Grid_params'],)
        
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

def episode_video_trigger(episode, vid_n_episodes):
    return episode % vid_n_episodes == 0


def make_minigrid_env(
    env_key: str,
    input_type: str,
    agent_start_pos: "tuple[int, int] | None" = None,
    agent_start_dir: "int | None" = None,
    seed: int = 0,
    vid_folder: str = "",
    vid_n_episodes: int = 0,
    wrapper: "str | None" = None,
    render_mode: str = "rgb_array",
    act_enc: "str | None" = None,
    **kwargs,  # e.g., subroom_size and open_all_paths for FourRooms, size for LRoom
):
    """Farama-minigrid env factory used by the RL_for_pRNN stack.

    Ported from SabrinaDu7/pRNN (there named make_env); renamed here to avoid
    clobbering the package-dispatch make_env above, which upstream examples,
    tests, and ObjectMemoryTask depend on.
    """
    import gymnasium as gym
    from gymnasium.wrappers.record_video import RecordVideo
    from functools import partial
    from minigrid.wrappers import (
        FullyObsWrapper,
        RGBImgPartialObsWrapper_HD,
    )

    from prnn.utils.enums import MinigridEnvNames, ActionEncodingsEnum, AgentInputType

    assert input_type in AgentInputType
    assert env_key in MinigridEnvNames
    assert act_enc in ActionEncodingsEnum

    env = gym.make(
        env_key,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        render_mode=render_mode,
        **kwargs,
    )

    if input_type == "Visual_FO":
        # Not RGB one here because we want RL agent to have as much info as possible
        env = FullyObsWrapper(env)
    elif "pRNN" in input_type or "PO" in input_type:
        # The same RGB wrapper is used for comparability whenever partial observation is needed
        env = RGBImgPartialObsWrapper_HD(env, tile_size=1)
    else:
        # For the cases without any visual input
        env = HDObsWrapper(env)

    if wrapper:
        import minigrid.wrappers as mgw

        env = getattr(mgw, wrapper)(env, **kwargs)

    if vid_n_episodes:
        trigger_func = partial(episode_video_trigger, vid_n_episodes=vid_n_episodes)
        env = RecordVideo(env, video_folder=vid_folder, episode_trigger=trigger_func)

    env.reset(seed=seed)
    env = FaramaMinigridShell(env, act_enc, env_key)
    return env


class HDObsWrapper(ObservationWrapper):
    """
    Wrapper exposing direction information (HD) alongside the mission string,
    for agent-input types without any visual input.
    """

    def __init__(self, env):
        super().__init__(env)
        HD_space = spaces.Discrete(4)

        if isinstance(self.observation_space, spaces.Dict):
            self.observation_space = spaces.Dict(
                {**self.observation_space.spaces, "HD": HD_space}
            )
        else:
            self.observation_space = spaces.Dict(
                {"mission": self.observation_space, "HD": HD_space}
            )

    def observation(self, observation):
        return {"mission": observation["mission"], "HD": observation["direction"]}
