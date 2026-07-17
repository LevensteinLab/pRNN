"""predict(batched=True) must match per-sequence serial predicts.

The batched input prep historically produced 3-D (L, X, B) where the forward
needs 4-D (1, L, X, B) and crashed; fixed 2026-07-14. Serial/batched equality
is float32 reduction-order noise, not bitwise (wider matmuls).
"""

import numpy as np
import torch

from prnn.utils.env import make_minigrid_env as make_env
from prnn.utils.enums import AgentInputType, MinigridEnvNames
from prnn.utils.predictiveNet import PredictiveNet

B, L = 3, 40


def _net():
    torch.manual_seed(0)
    np.random.seed(0)
    env = make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc="SpeedHD",
        seed=0,
    )
    pN = PredictiveNet(env, hidden_size=32, pRNNtype="thRNN_5win",
                       trainNoiseMeanStd=(0, 0), wandb_log=False)
    pN.pRNN.eval()  # dropout off; zero noise -> deterministic
    return pN


def test_predict_batched_matches_serial():
    pN = _net()
    obs = torch.rand(B, L + 1, pN.obs_size)
    act = torch.rand(B, L, pN.act_size)

    yb, tb, hb = pN.predict(obs, act, batched=True)
    assert yb.shape == (1, L, pN.obs_size, B)

    for b in range(B):
        y, t, h = pN.predict(obs[b : b + 1], act[b : b + 1])
        assert torch.allclose(y[0], yb[0, ..., b], atol=1e-5)
        assert torch.allclose(h[0], hb[0, ..., b], atol=1e-5)
        assert torch.equal(t[0], tb[0, ..., b])


def test_trainstep_batched_runs_and_loss_matches_serial_mean():
    """One batched trainStep's predloss equals the mean of the per-segment
    losses (equal-length segments, elementwise MSE) - checked on identical
    weights via deep copies since trainStep updates parameters."""
    pN = _net()
    obs = torch.rand(B, L + 1, pN.obs_size)
    act = torch.rand(B, L, pN.act_size)

    serial = [pN.copy() for _ in range(B)]
    batched = pN.copy()

    losses = [serial[b].trainStep(obs[b : b + 1], act[b : b + 1])[0] for b in range(B)]
    loss_batched = batched.trainStep(obs, act, batched=True)[0]
    assert np.isfinite(loss_batched)
    assert abs(loss_batched - float(np.mean(losses))) < 1e-5
