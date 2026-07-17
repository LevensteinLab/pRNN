"""Pin the temporal-alignment contract the RL_for_pRNN stack depends on.

For the masked (thRNN_*win) architectures, predict() returns obs_pred[t]
targeting obs[t] - the SAME index. Future prediction comes from inMask
zeroing the observation input on k of k+1 steps, NOT from a +1 shift
(predOffset stays 0). Downstream curiosity-reward alignment breaks silently
if this ever changes.
"""

import numpy as np
import torch

from prnn.utils.env import make_minigrid_env as make_env
from prnn.utils.enums import AgentInputType, MinigridEnvNames
from prnn.utils.predictiveNet import PredictiveNet


def _net(pRNNtype: str = "thRNN_5win") -> PredictiveNet:
    torch.manual_seed(0)
    np.random.seed(0)
    env = make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc="SpeedHD",
        seed=0,
    )
    return PredictiveNet(env, hidden_size=16, pRNNtype=pRNNtype,
                         trainNoiseMeanStd=(0, 0), wandb_log=False)


def test_masked_nets_predict_same_index():
    pN = _net()
    assert pN.pRNN.predOffset == 0
    assert pN.phase_k == 6


def test_masks_zero_future_obs():
    pN = _net()
    inMask = list(pN.pRNN.inMask)
    assert inMask == [True, False, False, False, False, False]
    # outMask keeps every prediction; actions are unmasked for plain thRNN_5win
    assert list(pN.pRNN.outMask) == [True] * 6
    assert list(pN.pRNN.actMask) == [True] * 6


def test_prevact_variant_shares_alignment():
    pN = _net("thRNN_5win_prevAct")
    assert pN.pRNN.predOffset == 0
    assert pN.pRNN.actOffset == 1
    assert list(pN.pRNN.inMask) == [True, False, False, False, False, False]


def test_predict_output_len_matches_input():
    """predOffset=0 => obs_pred has one prediction per input timestep."""
    pN = _net()
    pN.pRNN.eval()
    L = 24
    obs = torch.rand(1, L + 1, pN.obs_size)
    act = torch.rand(1, L, pN.act_size)
    obs_pred, obs_next, h = pN.predict(obs, act, randInit=False)
    assert obs_pred.shape[1] == obs_next.shape[1]
    assert obs_pred.shape[-1] == pN.obs_size
