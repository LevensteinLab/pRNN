"""calculateSpatialMetrics: SI/sRSA/SWdist from precomputed pooled activity."""

import numpy as np
import torch

from prnn.utils.env import make_minigrid_env as make_env
from prnn.utils.enums import AgentInputType, MinigridEnvNames
from prnn.utils.predictiveNet import PredictiveNet
from prnn.utils.agent import RandomActionAgent

SEED = 11


def _setup():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env = make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc="SpeedHD",
        seed=SEED,
    )
    pN = PredictiveNet(
        env,
        hidden_size=32,
        pRNNtype="thRNN_5win",
        trainNoiseMeanStd=(0, 0.03),
        wandb_log=False,
    )
    pN.pRNN.eval()
    agent = RandomActionAgent(env.action_space, np.array([0.15, 0.15, 0.6, 0.1]))
    return env, pN, agent


def test_metrics_from_pooled_trajectories():
    env, pN, agent = _setup()

    # pool 3 short trajectories, mimicking the RL repo's collection
    onset = 5
    h_rows, pos_rows = [], []
    for _ in range(3):
        obs, act, state, _ = pN.collectObservationSequence(env, agent, 60, discretize=True)
        with torch.no_grad():
            _, _, h = pN.predict(obs, act)
        h_mean = torch.mean(h, dim=0)
        h_rows.append(h_mean[onset:])
        pos_rows.append(state["agent_pos"][onset:-1, :])

    h_pool = torch.cat(h_rows)
    pos_pool = np.concatenate(pos_rows)

    metrics = pN.calculateSpatialMetrics(
        h_pool, pos_pool, env, sleep_timesteps=50, active_time_threshold=10
    )

    assert set(metrics) == {"SI", "sRSA", "SWdist"}
    assert np.isfinite(metrics["sRSA"])
    assert np.isfinite(metrics["SWdist"]) and metrics["SWdist"] >= 0
    assert len(metrics["SI"]) == pN.hidden_size
    assert np.isfinite(np.nanmean(np.asarray(metrics["SI"]["SI"], dtype=float)))


def test_metrics_shape_assert():
    env, pN, agent = _setup()
    bad_pos = np.zeros((7, 2))
    h = np.zeros((10, pN.hidden_size), dtype=np.float32)
    try:
        pN.calculateSpatialMetrics(h, bad_pos, env)
        raise AssertionError("expected shape assert to fire")
    except AssertionError as e:
        assert "must be (N, H) and (N, 2)" in str(e)
