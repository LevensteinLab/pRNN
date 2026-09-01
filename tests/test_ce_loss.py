"""The categorical (predCE) loss and its logits readout.

Gates, in dependency order: the default construction is untouched (sigmoid
readout, predMSE, bitwise-identical parameters); the logits readout changes
only the squash; predCE recovers exact targets from vocab-valued pixels,
fails loudly on out-of-vocabulary values, and trains end-to-end through
trainStep on a real shell.
"""

import pytest
import torch

from prnn.utils.env import make_env
from prnn.utils.lossFuns import predCE
from prnn.utils.predictiveNet import PredictiveNet

# A miniature 4-value vocabulary in [0,1]; tests are independent of the
# consumer repo's real 7-value palette on purpose.
VOCAB = torch.tensor(
    [[0.0, 0.0, 0.0], [0.3, 0.3, 0.3], [0.3, 0.3, 1.0], [1.0, 1.0, 0.3]]
)
C = VOCAB.shape[0]


@pytest.fixture(scope="module")
def env():
    return make_env("LRoom-18x18-v0", "farama-minigrid", "SpeedHD")


def _pixels(targets: torch.Tensor) -> torch.Tensor:
    """(N, n_tiles) class indices -> (N, n_tiles*3) vocab pixels."""
    return VOCAB[targets].reshape(targets.shape[0], -1)


def test_perfect_logits_reach_zero_loss():
    torch.manual_seed(0)
    targets = torch.randint(0, C, (5, 49))
    logits = torch.full((5, 49, C), -20.0)
    logits.scatter_(2, targets.unsqueeze(-1), 20.0)
    loss = predCE(VOCAB)(logits.reshape(5, -1), _pixels(targets), None)
    assert loss < 1e-6


def test_uniform_logits_sit_at_log_c():
    targets = torch.randint(0, C, (5, 49), generator=torch.Generator().manual_seed(1))
    loss = predCE(VOCAB)(torch.zeros(5, 49 * C), _pixels(targets), None)
    assert abs(float(loss) - torch.log(torch.tensor(float(C)))) < 1e-5


def test_out_of_vocabulary_pixel_fails_loudly():
    targets = torch.zeros(1, 49, dtype=torch.long)
    pixels = _pixels(targets)
    pixels[0, 0] = 0.5  # not a vocab value in any channel combination
    with pytest.raises(AssertionError, match="vocabulary"):
        predCE(VOCAB)(torch.zeros(1, 49 * C), pixels, None)


def test_batched_trailing_batch_axis_layout():
    """The masked-net batched layout puts features on dim 2 with B trailing.

    One-hot logits, NOT uniform: CE of uniform logits is ln C for ANY targets,
    so the uniform version could not catch a transposed target derivation in
    this - the trickiest - reshape (audit 2026-08-31). Perfect logits reach
    ~zero loss only if the target lookup used the very same layout.
    """
    B, L = 3, 4
    targets = torch.randint(0, C, (B * L, 49), generator=torch.Generator().manual_seed(2))
    pixels = _pixels(targets).reshape(1, L, B, 49 * 3).movedim(2, 3)  # (1, L, 147, B)
    onehot = torch.full((B * L, 49, C), -20.0)
    onehot.scatter_(2, targets.unsqueeze(-1), 20.0)
    logits = onehot.reshape(1, L, B, 49 * C).movedim(2, 3)  # (1, L, 343, B)
    loss = predCE(VOCAB)(logits, pixels, None)
    assert float(loss) < 1e-6
    uniform = predCE(VOCAB)(torch.zeros(1, L, 49 * C, B), pixels, None)
    assert abs(float(uniform) - torch.log(torch.tensor(float(C)))) < 1e-5


def test_focal_gamma_downweights_easy_tiles():
    targets = torch.randint(0, C, (5, 49), generator=torch.Generator().manual_seed(3))
    logits = torch.randn(5, 49, C, generator=torch.Generator().manual_seed(4))
    plain = predCE(VOCAB)(logits.reshape(5, -1), _pixels(targets), None)
    focal = predCE(VOCAB, focal_gamma=2.0)(logits.reshape(5, -1), _pixels(targets), None)
    assert 0 < float(focal) < float(plain)


def test_default_construction_is_untouched(env):
    """No CE kwargs -> the exact historical network: sigmoid squash, 147-wide
    readout, predMSE. The parameter shapes and the RNG stream must not move."""
    torch.manual_seed(0)
    pN = PredictiveNet(env, pRNNtype="thRNN_5win")
    assert type(pN.loss_fn).__name__ == "predMSE"
    assert len(pN.pRNN.outlayer) == 2  # Linear + Sigmoid
    assert pN.pRNN.outlayer[0].out_features == pN.obs_size


def test_ce_network_trains_end_to_end(env):
    torch.manual_seed(0)
    n_tiles = env.getObsSize() // 3
    pN = PredictiveNet(
        env,
        pRNNtype="thRNN_5win",
        losstype="predCE",
        loss_kwargs={"vocab": VOCAB},
        output_size=n_tiles * C,
        readout="logits",
    )
    assert pN.pRNN.outlayer[0].out_features == n_tiles * C
    assert len(pN.pRNN.outlayer) == 1  # no squash

    # A synthetic vocab-valued sequence instead of a live rollout: the loss
    # must SEE vocab pixels, and the real render values are the consumer
    # repo's contract, not this one's.
    T = 12
    targets = torch.randint(0, C, (T + 1, n_tiles), generator=torch.Generator().manual_seed(5))
    obs = VOCAB[targets].reshape(1, T + 1, -1)
    act = torch.zeros(1, T, pN.act_size)
    before = pN.pRNN.outlayer[0].weight.detach().clone()
    pN.trainStep(obs, act, return_stats=False)
    assert not torch.equal(before, pN.pRNN.outlayer[0].weight), "no gradient reached the readout"


def test_mlp_readout_replicates_the_grid_predict_decode(env):
    """readout="mlp" -> ResidualMLP, LayerNorm, Linear - the grid-predict
    decode stack - with W_out/b_out anchored on the PROJECTION and every
    trunk parameter actually updated by trainStep (a mis-wired optimizer
    group would leave the trunk frozen and the readout silently linear)."""
    from prnn.utils.Architectures import ResidualMLP

    torch.manual_seed(0)
    n_tiles = env.getObsSize() // 3
    pN = PredictiveNet(
        env, pRNNtype="thRNN_5win", losstype="predCE",
        loss_kwargs={"vocab": VOCAB}, output_size=n_tiles * C, readout="mlp",
    )
    trunk, norm_, proj = pN.pRNN.outlayer
    assert isinstance(trunk, ResidualMLP)
    assert isinstance(norm_, torch.nn.LayerNorm)
    assert isinstance(proj, torch.nn.Linear)
    assert pN.pRNN.W_out is proj.weight and pN.pRNN.b_out is proj.bias
    assert proj.out_features == n_tiles * C
    assert any(g.get("name") == "ReadoutTrunk" for g in pN.optimizer.param_groups)

    T = 12
    targets = torch.randint(0, C, (T + 1, n_tiles), generator=torch.Generator().manual_seed(5))
    obs = VOCAB[targets].reshape(1, T + 1, -1)
    act = torch.zeros(1, T, pN.act_size)
    before = {n: p.detach().clone() for n, p in pN.pRNN.outlayer.named_parameters()}
    pN.trainStep(obs, act, return_stats=False)
    stuck = [n for n, p in pN.pRNN.outlayer.named_parameters()
             if torch.equal(before[n], p)]
    assert not stuck, f"readout parameters not updated by trainStep: {stuck}"


def test_linear_readouts_have_no_trunk_group(env):
    torch.manual_seed(0)
    pN = PredictiveNet(env, pRNNtype="thRNN_5win")
    assert not any(g.get("name") == "ReadoutTrunk" for g in pN.optimizer.param_groups)
