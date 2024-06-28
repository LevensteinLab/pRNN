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

#https://arxiv.org/pdf/2105.04906.pdf
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