#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 16:37:45 2022

@author: dl2820
"""
import torch
from torch import nn

#All Losses forward should have inputs (obs_pred,obs_next,h)

class LPLLoss(nn.Module):
    
    def __init__(self, lambda_hebb=0.0001, lambda_decorr=0.01, epsilon = 1e-4):
        #LPL paper defaults: lambda_hebb=1, lambda_decorr=10, epsilon = 1e-4
        #~even: lambda_hebb=0.0002, lambda_decorr=0.02,
        super(LPLLoss, self).__init__()
        
        self.eps = epsilon
        self.l_he = lambda_hebb
        self.l_de = lambda_decorr


    def forward(self, obs_pred,obs_next,z):
        loss = self.L_pred(z) + self.l_he*self.L_hebb(z) + self.l_de*self.L_decorr(z)
        return loss
    
    
    def L_pred(self, z):
        z_SG = z.detach()
        loss = 0.5*(z[:,1:,:] - z_SG[:,:-1,:]).square().mean()
        return loss
    
    def L_hebb(self, z):
        z_center = z.mean(dim=(0,1)).detach()
        variance = ((z - z_center) ** 2).sum(dim=(0,1)) / (z.shape[0] + z.shape[1] - 1)
        loss = -torch.log(variance + self.eps).mean()
        return loss
    
    def L_decorr(self, z):
        z_mean = z.mean(dim=(0,1)).detach()
        z_centered = (z - z_mean).reshape(1,-1,z.size(2))
        #cov = torch.einsum('ij,ik->jk', a_centered, a_centered).fill_diagonal_(0) / (a.shape[0] - 1)
        #loss = torch.sum(cov ** 2) / (cov.shape[0] ** 2 - cov.shape[0])
        cov = z_centered[0,:,:].T.cov().fill_diagonal_(0)
        loss = torch.sum(cov ** 2) / (cov.shape[0])
        return loss
    
    
class predRMSE(nn.Module):
    def __init__(self, eps=1e-8, **kwargs):
        super(predRMSE,self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, obs_pred,obs_next,z):
        loss = torch.sqrt(self.mse(obs_pred,obs_next)+self.eps)
        return loss

    
class predCE(nn.Module):
    """Per-tile cross-entropy on a discrete tile vocabulary.

    The observation space at tile_size=1 IS categorical - each tile is one of
    a small closed set of RGB values - so CE is the correct likelihood where
    MSE-on-sigmoid was a Gaussian pasted onto discrete data (and measurably
    hedged toward the pixel mean). Requires the matching `readout="logits"`
    architecture kwarg: obs_pred carries n_tiles*C logits where obs_next
    carries n_tiles*3 pixels.

    vocab: (C, 3) float tensor in [0, 1], the exact tile values, an EXPLICIT
    constructor input (never inferred from a batch - class indices must be
    stable across rooms, runs and checkpoints). Targets are nearest-vocab
    per tile with a closed-set assert: an unseen tile value fails loudly
    rather than training on a wrong label. The assert is skipped while a CUDA
    graph is capturing (a host sync cannot run there); the eager warmup
    passes that precede every capture do run it.

    focal_gamma: optional focal reweighting ((1-pt)^gamma * ce, as in
    ../grid-predict/src/train). None = plain CE. Contingency only.
    """

    def __init__(self, vocab, focal_gamma=None, **kwargs):
        super().__init__()
        self.register_buffer("vocab", torch.as_tensor(vocab, dtype=torch.float32))
        self.n_classes = self.vocab.shape[0]
        self.n_channels = self.vocab.shape[1]
        self.focal_gamma = focal_gamma

    def forward(self, obs_pred, obs_next, z):
        if self.vocab.device != obs_next.device:
            self.vocab = self.vocab.to(obs_next.device)
        C = self.n_classes
        # feature axis: obs_next has n_tiles*3 pixels there, obs_pred n_tiles*C logits
        n_tiles = None
        for ax, s in enumerate(obs_next.shape):
            if s % self.n_channels == 0 and obs_pred.shape[ax] == (s // self.n_channels) * C and s != obs_pred.shape[ax]:
                n_tiles, pix_ax = s // self.n_channels, ax
                break
        assert n_tiles is not None, (
            f"no axis pairs obs_next {tuple(obs_next.shape)} with obs_pred "
            f"{tuple(obs_pred.shape)} as pixels vs {C}-class logits"
        )
        logits = obs_pred.movedim(pix_ax, -1).reshape(-1, n_tiles, C)
        pixels = obs_next.movedim(pix_ax, -1).reshape(-1, n_tiles, self.n_channels)
        dist = (pixels.unsqueeze(-2) - self.vocab).abs().sum(-1)
        mindist, targets = dist.min(-1)
        if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            assert float(mindist.max()) < 1e-3, (
                "tile value outside the committed vocabulary - "
                "rebuild envs/palette.py::TILE_VOCABULARY"
            )
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, C), targets.reshape(-1), reduction="none"
        )
        if self.focal_gamma is not None:
            pt = torch.exp(-ce)
            ce = ((1 - pt) ** self.focal_gamma) * ce
        return ce.mean()

    def render(self, obs_pred):
        """Logits rows -> displayable pixel rows in [0,1] (argmax colour).

        Trailing-feature layouts (…, n_tiles*C), which is what the serial
        predict path and every figure consumer hold. The ONE home for
        "what does a categorical prediction look like": plotSampleTrajectory
        and the RL adapter's render_prediction_rows both call this.
        """
        C = self.n_classes
        assert obs_pred.shape[-1] % C == 0, (
            f"trailing dim {obs_pred.shape[-1]} is not a multiple of {C} classes"
        )
        vocab = self.vocab.to(obs_pred.device)
        n_tiles = obs_pred.shape[-1] // C
        classes = obs_pred.reshape(*obs_pred.shape[:-1], n_tiles, C).argmax(-1)
        return vocab[classes].reshape(*obs_pred.shape[:-1], n_tiles * self.n_channels)


class predMSE(nn.Module):
    def __init__(self, **kwargs):
        super(predMSE,self).__init__()
        self.loss_fn = nn.MSELoss()
        
    def forward(self, obs_pred,obs_next,z):
        loss = self.loss_fn(obs_pred,obs_next)
        return loss

#Note - will need to update predMSE to match output (total,pred)
class predMSE_reg(nn.Module):
    def __init__(self, beta_energy=0, **kwargs):
        super(predMSE, self).__init__()
        
        self.beta_energy = beta_energy
        self.loss_fn = nn.MSELoss()
    
    def forward(self, obs_pred,obs_next,z):
        predloss = self.loss_fn(obs_pred,obs_next) 
        energyloss = self.beta_energy*torch.linalg.vector_norm(z).sum() #Check dimension here
        totalloss = predloss+energyloss
        return totalloss, predloss

#https://arxiv.org/png/2105.04906.png
class VICReg(nn.Module):
    def __init__(self):
        super(VICReg,self).__init__()

#%%
# loss = LPLLoss()
# #%%
# x = torch.rand(1,100,10,requires_grad=True)
# for ll in range(100):
#     lloss = loss(x)
#     lloss.backward()
#     x = x - x.grad
# #%%
# import torch
# x = torch.rand(1,4,5,requires_grad=True)

# x_mean = x.mean(dim=(0,1)).detach()
# x_centered = (x - x_mean).reshape(1,-1,x.size(2))
# x_centered[0,:,:].T.cov()

# torch.einsum('ij,ik->jk', x_centered[0,:,:], x_centered[0,:,:]).fill_diagonal_(0) / (x.shape[0] + x.shape[1] - 1)