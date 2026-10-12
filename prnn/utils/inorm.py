"""
Inhibitory normalization (I-Norm) building blocks for the INormRNNCell.

Ported from RNN_EI_Network (branch param-layer-norm-neurogym-integration,
commit 9f8269c): ``inhibition/normalization.py`` and ``inhibition/init.py``.

The excitatory RNN drive is normalized by learned inhibition instead of an
exact LayerNorm:
    - subtractive inhibition: ``mean_net`` predicts the mean of the drive,
    - divisive inhibition:    ``var_net`` (Linear -> square -> average) predicts
      its variance.
Both predictors are initialized to reproduce LayerNorm exactly and are kept
LayerNorm-like during training by an auxiliary moment-matching loss. The task
gradient is routed through a configurable LayerNorm-like backward ("GradNorm").
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def calc_ln_mu_sigma(mean, var):
    """Parameters of a lognormal with the given mean and variance."""
    mu_ln = np.log(mean**2 / np.sqrt(mean**2 + var))
    sigma_ln = np.sqrt(np.log(1 + (var / mean**2)))
    return mu_ln, sigma_ln


def excitatory_weight(Wex, numerator=2, k=1):
    """Non-negative lognormal init for an excitatory weight matrix (in place)."""
    ne, n_input = Wex.shape
    target_std_wex = np.sqrt(numerator * ne / (n_input * (ne - 1)))
    mu, sigma = calc_ln_mu_sigma(target_std_wex * k, target_std_wex**2)
    Wex_np = np.random.lognormal(mu, sigma, size=(ne, n_input))
    Wex.data = torch.from_numpy(Wex_np).float()
    Wex.clamp = True


def parametrized_ln_mean_weight(mean_linear, W_XE, W_EE, bias=None):
    """Subtractive pathway: predicts the mean of W_XE x + W_EE h (+ bias)."""
    W_XE_np = W_XE.detach().cpu().to_dense().numpy()
    W_EE_np = W_EE.detach().cpu().to_dense().numpy()
    W_full = np.concatenate([W_XE_np, W_EE_np], axis=1)
    mean_row = W_full.mean(axis=0, keepdims=True)
    mean_linear.weight.data = torch.from_numpy(mean_row).float()
    bias_mean = float(bias.detach().cpu().mean()) if bias is not None else 0.0
    mean_linear.bias.data = torch.tensor([bias_mean], dtype=torch.float32)
    mean_linear.weight.clamp = True
    mean_linear.bias.clamp = True


def parametrized_ln_var_weight(var_first, var_second, W_XE, W_EE, bias=None, n_div=None):
    """Divisive pathway: SVD of the centered weights so that
    ``var_second((var_first(feat))**2)`` equals the variance of the drive."""
    ne, n_in = W_XE.shape
    _, n_h = W_EE.shape
    feat_dim = n_in + n_h
    W_XE_np = W_XE.detach().cpu().to_dense().numpy()
    W_EE_np = W_EE.detach().cpu().to_dense().numpy()
    if bias is not None:
        bias_col = bias.detach().cpu().numpy().reshape(-1, 1)
    else:
        bias_col = np.zeros((ne, 1))
    W_full = np.concatenate([W_XE_np, W_EE_np, bias_col], axis=1)
    W_IX = W_full.mean(axis=0, keepdims=True)
    W_eff = W_full - np.ones((ne, 1)) @ W_IX
    _, S, V_T = np.linalg.svd(W_eff)
    if n_div is None:
        n_div = ne
    k = min(n_div, S.shape[0])
    B_IX = np.diag(S[:k]) @ V_T[:k, :]
    B_IX_w = B_IX[:, :feat_dim]
    B_IX_b = B_IX[:, feat_dim]
    # var_first may have more rows than k (if n_div > rank); zero the rest.
    w = np.zeros((var_first.out_features, feat_dim))
    b = np.zeros(var_first.out_features)
    w[:k], b[:k] = B_IX_w, B_IX_b
    var_first.weight.data = torch.from_numpy(w).float()
    var_first.bias.data = torch.from_numpy(b).float()
    var_first.weight.clamp = False
    var_first.bias.clamp = False
    B_EI = np.ones((1, var_first.out_features)) / ne
    var_second.weight.data = torch.from_numpy(B_EI).float()
    var_second.bias.data = torch.zeros(1, dtype=torch.float32)
    var_second.weight.clamp = True
    var_second.bias.clamp = True


# ---------------------------------------------------------------------------
# Normalization modules
# ---------------------------------------------------------------------------


class Square(nn.Module):
    """Point-wise square used in the divisive-inhibition path."""

    def forward(self, x):
        return x * x


class LayerNormalizeFunctionFA(torch.autograd.Function):
    """Identity forward with configurable LayerNorm-like backward ablations."""

    @staticmethod
    def forward(ctx, x, weights, no_backward, ln_feedback):
        epsilon = 1e-5
        ctx.no_backward = no_backward
        ctx.ln_feedback = ln_feedback
        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean
        actual_var = x.var(dim=-1, keepdim=True, unbiased=False)
        ctx.actual_std = torch.sqrt(actual_var + epsilon)
        ctx.save_for_backward(x_centered, weights.to(x.device))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.no_backward:
            return grad_output, None, None, None

        x_centered, weights = ctx.saved_tensors
        D = x_centered.shape[-1]
        grad_input = grad_output

        if ctx.ln_feedback == "full":
            grad_mean = grad_output.mean(dim=-1, keepdim=True)
            x_hat = x_centered / ctx.actual_std
            dot = (grad_output * x_hat).mean(dim=-1, keepdim=True)
            grad_input = (grad_output - grad_mean - x_hat * dot) / ctx.actual_std
        elif ctx.ln_feedback == "center":
            grad_mean = grad_output.mean(dim=-1, keepdim=True)
            grad_input = grad_output - grad_mean
        elif ctx.ln_feedback == "fa_center":
            grad_mean = (weights * grad_output).sum(dim=-1, keepdim=True)
            grad_input = grad_output - grad_mean
        elif ctx.ln_feedback == "scale":
            grad_input = grad_output / ctx.actual_std
        elif ctx.ln_feedback == "decorrelate":
            dot = (grad_output * x_centered).sum(dim=-1, keepdim=True)
            grad_input = grad_output - x_centered * dot / D

        return grad_input, None, None, None


LN_FEEDBACK_OPTIONS = ("full", "center", "fa_center", "scale", "decorrelate")


class LayerNormFeedbackAblation(nn.Module):
    """Forward: identity. Backward: (part of) the LayerNorm Jacobian, chosen by
    ``ln_feedback``. ``no_backward=True`` passes the gradient straight through."""

    def __init__(self, ln_feedback="full", no_backward=False):
        super().__init__()
        if ln_feedback not in LN_FEEDBACK_OPTIONS:
            raise ValueError(f"ln_feedback must be one of {LN_FEEDBACK_OPTIONS}")
        self.ln_feedback = ln_feedback
        self.no_backward = no_backward

    def forward(self, x, weights=None):
        if weights is None:
            weights = torch.ones(x.shape[-1], device=x.device) / x.shape[-1]
        return LayerNormalizeFunctionFA.apply(x, weights, self.no_backward, self.ln_feedback)


class ParametrizedLayerNorm(nn.Module):
    """Predict scalar mean/variance from ``(x_t, h_prev)`` and normalize ``pre_act``.

    Forward: ``z = (pre_act - pred_mean) / sqrt(pred_var)`` with stop-gradiented
    statistics, followed by the GradNorm backward. Also returns the aux moment
    loss, which only trains the predictors (``mean_net`` and ``var_net[0]``).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        eps=1e-5,
        stats_hidden_size=None,
        freeze_ei=True,
        ln_feedback="full",
        gradient_norm=True,
    ):
        super().__init__()
        self.eps = eps
        self.ln_feedback = ln_feedback
        self.gradient_norm = gradient_norm
        self.grad_norm = LayerNormFeedbackAblation(
            ln_feedback=ln_feedback, no_backward=not gradient_norm
        )
        # Fixed, random, positive weights summing to 1: only used by ``fa_center``.
        if ln_feedback == "fa_center":
            nu = torch.rand(hidden_size)
        else:
            nu = torch.ones(hidden_size)
        self.register_buffer("fa_weights", nu / nu.sum())

        feat_dim = input_size + hidden_size
        stats_hidden_size = stats_hidden_size or hidden_size  # number of divisive units
        self.mean_net = nn.Linear(feat_dim, 1)
        self.var_net = nn.Sequential(
            nn.Linear(feat_dim, stats_hidden_size),
            Square(),
            nn.Linear(stats_hidden_size, 1),
        )
        self.register_buffer("fanout", torch.ones(hidden_size, 1))

        # The divisive readout (I -> E) is a fixed uniform average.
        if freeze_ei:
            for p in self.var_net[2].parameters():
                p.requires_grad_(False)

    def init_from_rnn_weights(self, W_XE, W_EE, bias=None):
        """Make the predicted-stats norm equal LayerNorm(pre_act) at init."""
        n_div = self.var_net[0].out_features
        parametrized_ln_mean_weight(self.mean_net, W_XE, W_EE, bias)
        parametrized_ln_var_weight(self.var_net[0], self.var_net[2], W_XE, W_EE, bias, n_div=n_div)

    def clamp_weights(self):
        with torch.no_grad():
            for p in self.parameters():
                if getattr(p, "clamp", False):
                    p.clamp_(min=0)

    def predict_stats(self, x_t, h_prev):
        feat = torch.cat([x_t, h_prev], dim=-1)
        pred_mean = self.mean_net(feat) @ self.fanout.T
        pred_var = (self.var_net(feat) + self.eps) @ self.fanout.T
        return pred_mean, pred_var

    def aux_loss(self, pred_mean, pred_var, pre_act):
        """I-Norm moment-matching loss: E[mean(z)^2 + (var(z) - 1)^2]."""
        x = pre_act.detach()
        z = (x - pred_mean) / torch.sqrt(pred_var)
        mean = z.mean(dim=-1)
        var = z.var(dim=-1, unbiased=False)
        return (mean.pow(2) + (var - 1).pow(2)).mean()

    def forward(self, pre_act, x_t, h_prev):
        pred_mean, pred_var = self.predict_stats(x_t.detach(), h_prev.detach())
        aux = self.aux_loss(pred_mean, pred_var, pre_act)
        z = (pre_act - pred_mean.detach()) / torch.sqrt(pred_var.detach())
        out = self.grad_norm(z, weights=self.fa_weights)
        return out, aux
