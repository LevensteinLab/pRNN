"""
Tests for the inhibitory normalization cell (INormRNNCell).
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from prnn.utils.predictiveNet import PredictiveNet, CELL_TYPES, netOptions
from prnn.utils.env import make_env
from prnn.utils.Architectures import MaskedRNN
from prnn.utils.thetaRNN import INormRNNCell


@pytest.fixture
def mock_env():
    return make_env("LRoom-18x18-v0", "farama-minigrid", "SpeedHD")


@pytest.fixture
def inorm_net(mock_env):
    torch.manual_seed(0)
    np.random.seed(0)
    return PredictiveNet(mock_env, pRNNtype="inormRNN_mask", hidden_size=64)


def _drive(cell, x, h):
    return x @ cell.weight_ih.T + h @ cell.weight_hh.T


def test_registered():
    assert CELL_TYPES["INormRNNCell"] is INormRNNCell
    assert "inormRNN_mask" in netOptions


@pytest.mark.parametrize("batch", [1, 8])
def test_cell_matches_layernorm_at_init(batch):
    """At init the inhibitory pathways reproduce LayerNorm exactly."""
    np.random.seed(0)
    torch.manual_seed(0)
    cell = INormRNNCell(20, 50)
    assert (cell.weight_ih >= 0).all() and (cell.weight_hh >= 0).all()
    x = torch.rand(batch, 20)
    h = torch.relu(torch.randn(batch, 50))
    pre = _drive(cell, x, h)
    z, aux = cell.paramln(pre, x, h)
    torch.testing.assert_close(z, F.layer_norm(pre, (50,)), atol=1e-3, rtol=1e-3)
    assert aux.item() < 1e-4


def test_cell_forward_shapes_and_aux():
    np.random.seed(0)
    cell = INormRNNCell(20, 50)
    x = torch.rand(4, 20)
    h = torch.relu(torch.randn(4, 50))
    hy, state = cell(x, torch.zeros(4, 50), (h,))
    assert hy.shape == (4, 50) and state[0] is hy
    assert len(cell._aux_losses) == 1
    with torch.no_grad():
        cell(x, torch.zeros(4, 50), (h,))
    assert len(cell._aux_losses) == 1  # nothing stored without grad
    assert cell.pop_aux_loss() is not None
    assert cell.pop_aux_loss() is None


def test_prnn_reinit_after_identity(mock_env):
    """pRNN adds (1-1/tau)I to W_hh; the norm must be refit to the final weights."""
    np.random.seed(0)
    torch.manual_seed(0)
    net = MaskedRNN(mock_env.getObsSize(), mock_env.getActSize(), 64, cell=INormRNNCell, k=5)
    cell = net.rnn.cell
    assert (cell.weight_hh.diagonal() > 0).all()
    x = torch.rand(3, cell.input_size)
    h = torch.relu(torch.randn(3, 64))
    pre = _drive(cell, x, h)
    z, _ = cell.paramln(pre, x, h)
    torch.testing.assert_close(z, F.layer_norm(pre, (64,)), atol=1e-3, rtol=1e-3)


def test_optimizers(inorm_net):
    assert inorm_net.optimizer_norm is not None
    norm_ids = {id(p) for g in inorm_net.optimizer_norm.param_groups for p in g["params"]}
    main_ids = {id(p) for g in inorm_net.optimizer.param_groups for p in g["params"]}
    assert norm_ids.isdisjoint(main_ids)
    # the divisive readout (I -> E) is frozen
    assert not any(p.requires_grad for p in inorm_net.pRNN.rnn.cell.paramln.var_net[2].parameters())


def test_train_step(inorm_net, mock_env):
    from prnn.utils.agent import create_agent

    agent = create_agent("LRoom-18x18-v0", mock_env, "RandomActionAgent")
    cell = inorm_net.pRNN.rnn.cell
    # Push a weight negative to check the projection after the step
    with torch.no_grad():
        cell.weight_hh[0, 1] = -1.0
    mean_before = cell.paramln.mean_net.weight.detach().clone()
    var_before = cell.paramln.var_net[0].weight.detach().clone()
    obs, act, _, _ = inorm_net.collectObservationSequence(mock_env, agent, 50)
    steploss, _, _ = inorm_net.trainStep(obs, act)
    assert np.isfinite(steploss)
    assert (cell.weight_ih >= 0).all() and (cell.weight_hh >= 0).all()
    assert not torch.equal(cell.paramln.mean_net.weight, mean_before) or not torch.equal(
        cell.paramln.var_net[0].weight, var_before
    )
    for key in ["aux_loss", "ln_mse", "ln_cos"]:
        assert key in inorm_net.TrainingSaver
    assert len(cell._aux_losses) == 0 and cell._ln_n == 0
    # right after init the inhibition still matches LayerNorm closely
    assert inorm_net.TrainingSaver["ln_cos"].iloc[-1] > 0.99


def test_default_norm_lrs(inorm_net):
    lrs = [g["lr"] for g in inorm_net.optimizer_norm.param_groups]
    assert lrs == [3e-3, 1e-1]


def test_epoch_stops_on_divergence(inorm_net, mock_env):
    from prnn.utils.agent import create_agent

    agent = create_agent("LRoom-18x18-v0", mock_env, "RandomActionAgent")
    with torch.no_grad():
        inorm_net.pRNN.W_out.fill_(float("nan"))
    inorm_net.trainingEpoch(mock_env, agent, sequence_duration=20, num_trials=5)
    assert inorm_net.diverged
    assert len(inorm_net.TrainingSaver) == 1  # stopped after the first non-finite trial


def test_save_load(inorm_net):
    inorm_net.saveNet("tmp/test_inorm")
    loaded = PredictiveNet.loadNet("tmp/test_inorm")
    assert isinstance(loaded.pRNN.rnn.cell, INormRNNCell)
    torch.testing.assert_close(loaded.pRNN.W, inorm_net.pRNN.W)
    assert loaded.optimizer_norm is not None
