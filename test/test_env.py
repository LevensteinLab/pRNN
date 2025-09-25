import os
import numpy as np

from minigrid.envs import LEnv_18_goal, LEnv_16_goal
from prnn.utils.env import make_env
from prnn.utils.predictiveNet import PredictiveNet

ENV_NAME = "MiniGrid-LRoom_Goal-18x18-v0"
ENV_CLASS = LEnv_18_goal if '18' in ENV_NAME else LEnv_16_goal
NET_NAME = "thRNN_5winthRNN_5win--s8_old"

def test_import_lenv_goal():
    assert ENV_CLASS is not None
    assert hasattr(ENV_CLASS, '__init__')

def test_create_lroom_goal_env():
    env = make_env(ENV_NAME)
    
    # Basic checks
    assert env is not None
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    assert hasattr(env, 'observation_space')
    assert hasattr(env, 'action_space')
    
    while hasattr(env, 'env'):
        env = env.env
    assert isinstance(env, ENV_CLASS)

def test_env_reset():
    """Test that the environment can be reset successfully."""
    env = make_env(ENV_NAME)
    
    # Should return observation and info
    result = env.reset()
    assert isinstance(result, dict)
    assert len(result) == 3
    
    mission, image, direction = result
    assert isinstance(mission, str) # TODO: Fishy
    assert isinstance(image, str)
    assert isinstance(direction, str)

def test_environment_step():
    """Test that the environment can perform a step."""
    env = make_env(ENV_NAME)
    env.reset()
    
    # Take a random action (0 should be valid for most MiniGrid envs)
    action = 0
    result = env.step(action)
    
    # Should return obs, reward, terminated, truncated, info
    assert isinstance(result, tuple)
    assert len(result) == 5
    
    obs, reward, terminated, truncated, info = result
    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# Test for network loading
def test_predictive_net_import():
    """Test that PredictiveNet can be imported."""
    assert PredictiveNet is not None
    assert hasattr(PredictiveNet, 'loadNet')

def test_load_pretrained_network():
    """Test loading a pre-trained network."""

    assert os.path.isfile(f"nets/{NET_NAME}.pkl")
    predictiveNet = PredictiveNet.loadNet(NET_NAME)

    assert predictiveNet is not None
    assert hasattr(predictiveNet, 'pRNN')
    assert hasattr(predictiveNet, 'EnvLibrary')
    assert hasattr(predictiveNet, 'env_shell')
    assert hasattr(predictiveNet, 'predict')
    assert hasattr(predictiveNet, 'addEnvironment')


def test_network_with_environment():
    """Test that a loaded network can work with the LRoom environment."""
    network_name = "thRNN_5winthRNN_5win--s8_old"
    
    predictiveNet = PredictiveNet.loadNet(network_name)
    env = make_env(ENV_NAME)
    predictiveNet.addEnvironment(env)
    
    # Basic functionality test
    assert len(predictiveNet.EnvLibrary) >= 2  # Original env + new env

test_env_reset()