#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:17:33 2022

@author: dl2820
"""

""" Preditive net modules below """

from torch import nn
import torch
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import norm
from prnn.utils.thetaRNN import thetaRNNLayer, RNNCell, LayerNormRNNCell, AdaptingLayerNormRNNCell, AdaptingRNNCell, SparseGatedRNNCell, LogNRNNCell

from prnn.utils.pytorchInits import CANN_


class pRNN(nn.Module):
    """
    A general predictive RNN framework that takes observations and actions, and
    returns predicted observations, as well as the actual observations to train
    Observations are inputs that are predicted. Actions are inputs that are
    relevant to prediction but not predicted.

    predOffset: the output at time t s matched to obs t+predOffset (defulat: 1)
    actOffset:  the action input at time t is the action took at t-actOffset (default:0)
    inMask:     boolean list, length corresponding to the prediction cycle period. (default: [True])
    outMask, actMask: list, same length as inMask (default: None)

    All inputs should be tensors of shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
    """
    def __init__(self, obs_size, act_size, hidden_size=500,
                 cell=RNNCell,  dropp=0, bptttrunc=50, k=0, f=0.5,
                 predOffset=1, inMask=[True], outMask=None, 
                 actOffset=0, actMask=None, neuralTimescale=2,
                 continuousTheta=False,
                 **cell_kwargs):
        super(pRNN, self).__init__()

        #pRNN architecture parameters
        self.predOffset = predOffset
        self.actOffset = actOffset
        self.actpad = nn.ConstantPad1d((0,0,self.actOffset,0),0)
        self.batched_actpad = nn.ConstantPad1d((0,0,0,0,self.actOffset,0),0)
        self.inMask = inMask
        if outMask is None:
            outMask = [True for i in inMask]
        if actMask is None:
            actMask = [True for i in inMask]
        self.outMask = outMask
        self.actMask = actMask
        self.hidden_size = hidden_size
        
        self.droplayer = nn.Dropout(p=dropp)
        #self.droplayer_act = nn.Dropout(p=dropp_act)
        
        self.create_layers(obs_size, act_size, hidden_size,
                           cell, bptttrunc, continuousTheta,
                           k, f, **cell_kwargs)

        self.W_in = self.rnn.cell.weight_ih
        self.W = self.rnn.cell.weight_hh
        self.W_out = self.outlayer[0].weight
        self.bias = self.rnn.cell.bias

        if hasattr(self.rnn.cell,'weight_is'):
            self.W_is = self.rnn.cell.weight_is
            self.W_sh = self.rnn.cell.weight_sh
            self.W_hs = self.rnn.cell.weight_hs
        
        self.neuralTimescale = neuralTimescale

        with torch.no_grad():
            self.W.add_(torch.eye(hidden_size).mul_(1-1/self.neuralTimescale).to_sparse())

    
    def create_layers(self, obs_size, act_size, hidden_size,
                      cell, bptttrunc, continuousTheta,
                      k, f, **cell_kwargs):
        #Sparsity via layernorm subtraction
        mu = norm.ppf(f)
        musig = [mu,1]

        #TODO: add cellparams input to pass through
        #Consider putting the sigmoid outside this layer...
        input_size = obs_size + act_size
        self.rnn = thetaRNNLayer(cell, bptttrunc, input_size, hidden_size,
                                 defaultTheta=k, continuousTheta=continuousTheta,
                                 musig=musig, **cell_kwargs)
        self.outlayer = nn.Sequential(
            nn.Linear(hidden_size, obs_size, bias=False),
            nn.Sigmoid()
            )


    def forward(self, obs, act, noise_params=(0,0), state=torch.tensor([]), theta=None,
                single=False, mask=None, batched=False, fullRNNstate=False):
        #Determine the noise shape
        k=0
        if hasattr(self,'k'):
            k= self.k
        if batched:
            noise_shape = (k+1, obs.size(1), self.hidden_size, obs.size(-1))
        else:
            noise_shape = (k+1, obs.size(1), self.rnn.cell.hidden_size)
            #^^^for backwards compadiblity. change to self.hidden_size later

        noise_t = self.generate_noise(noise_params, noise_shape)

        if single:
            x_t = torch.cat((obs,act), 2)
            h_t,_ = self.rnn(x_t, internal=noise_t, state=state, theta=theta)
            if not fullRNNstate: 
                h_t = h_t[:,:,:self.hidden_size] #For RNNcells that output more than the hidden RNN units
            y_t = None
            obs_target = None
        else:
            x_t, obs_target, outmask = self.restructure_inputs(obs,act,batched)
            #x_t = self.droplayer(x_t) # (should it count action???) dropout with action
            h_t,_ = self.rnn(x_t, internal=noise_t, state=state,
                             theta=theta, mask=mask, batched=batched)
            if not fullRNNstate: 
                h_t = h_t[:,:,:self.hidden_size] #For RNNcells that output more than the hidden RNN units (ugly)
            if batched:
                h_t = h_t.permute(-1,*[i for i in range(len(h_t.size())-1)])
                allout = self.outlayer(h_t[:,:,:,:self.hidden_size])
                allout = allout.permute(*[i for i in range(1,len(allout.size()))],0)
                h_t = h_t.permute(*[i for i in range(1,len(h_t.size()))],0)
            else:
                allout = self.outlayer(h_t[:,:,:self.hidden_size])

            #Apply the mask to the output
            y_t = torch.zeros_like(allout)
            y_t[:,outmask,:] = allout[:,outmask,:] #The predicted outputs.
        return y_t, h_t, obs_target

   #TODO: combine forward and internal?
    def internal(self, noise_t, state=torch.tensor([])):
        h_t,_ = self.rnn(internal=noise_t, state=state, theta=0)
        y_t = self.outlayer(h_t)
        return y_t, h_t

    def generate_noise(self, noise_params, shape):
        if noise_params != (0,0):
            noise = noise_params[0] + noise_params[1]*torch.randn(shape, device=self.W.device)
        else:
            noise = torch.zeros(shape, device=self.W.device)

        return noise

    def restructure_inputs(self, obs, act, batched=False):
        """
        Join obs and act into a single input tensor shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
        obs should be one timestep longer than act, for the [t+1] observation
        after the last action
        """

        #Apply the action and prediction offsets
        if batched:
            act = self.batched_actpad(act)
        else:
            act = self.actpad(act)
        
        if self.actOffset:
                act = act[:,:-self.actOffset,...]

        obs_target = obs[:,self.predOffset:,:]

        #Make everything the same size
        minsize = min(obs.size(1),act.size(1),obs_target.size(1))
        obs, act = obs[:,:minsize,:], act[:,:minsize,:]
        obs_target = obs_target[:,:minsize,:]

        #Apply the masks (this is ugly.)
        actmask = np.resize(np.array(self.actMask),minsize)
        outmask = np.resize(np.array(self.outMask),minsize)
        obsmask = np.resize(np.array(self.inMask),minsize)

        obs_out = torch.zeros_like(obs, requires_grad=False)
        act_out = torch.zeros_like(act, requires_grad=False)
        obs_target_out = torch.zeros_like(obs_target, requires_grad=False)

        obs_out[:,obsmask,:] = obs[:,obsmask,:]
        act_out[:,actmask,:] = act[:,actmask,:]
        obs_target_out[:,outmask,:] = obs_target[:,outmask,:]
        
        obs_out = self.droplayer(obs_out) #dropout without action
        #act_out = self.droplayer_act(act_out) #dropout without action

        #Concatenate the obs/act into a single input
        x_t = torch.cat((obs_out,act_out), 2)
        return x_t, obs_target_out, outmask


    def spontaneous(self, timesteps, noisemean, noisestd, wgain=1,
                    agent=None, randInit=True, env=None):
        device = self.W.device
        #Noise
        noise_params = (noisemean,noisestd)
        #for backwards compadibility. change to self.hidden_size later
        noise_shape = (1,timesteps,self.rnn.cell.hidden_size) 
        noise_t = self.generate_noise(noise_params, noise_shape)
        if randInit:
            noise_shape = (1,1,self.rnn.cell.hidden_size)
            state = self.generate_noise(noise_params, noise_shape)
            state = self.rnn.cell.actfun(state)
        else:
            state = torch.tensor([])
                
        #Weight Gain
        with torch.no_grad():
            offdiags = self.W.mul(1-torch.eye(self.rnn.cell.hidden_size))
            self.W.add_(offdiags*(wgain-1))
            
        #Action
        if agent is not None:
            obs,act,_,_ = env.collectObservationSequence(agent, timesteps)
            obs,act = obs.to(device),act.to(device)
            obs = torch.zeros_like(obs)
            
            obs_pred, h_t, _ = self.forward(obs, act, noise_t=noise_t, state=state, theta=0)
            noise_t = (noise_t,act)
        else:
            obs_pred,h_t = self.internal(noise_t, state=state)
        
        with torch.no_grad():
            self.W.subtract_(offdiags*(wgain-1))
            
        return obs_pred,h_t,noise_t



class pRNN_th(pRNN):    
    def __init__(self, obs_size, act_size, k, hidden_size=500,
                 cell=RNNCell,  dropp=0, bptttrunc=50, f=0.5,
                 predOffset=0, inMask=[True], outMask=None,
                 actOffset=0, actMask=None, neuralTimescale=2,
                continuousTheta=False, actionTheta=False,
                **cell_kwargs):
        super(pRNN_th, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                cell=cell,  dropp=dropp, bptttrunc=bptttrunc, k=k, f=f,
                predOffset=predOffset, inMask=inMask, outMask=outMask,
                actOffset=actOffset, actMask=actMask,
                neuralTimescale=neuralTimescale,
                continuousTheta=continuousTheta,
                **cell_kwargs)
        
        self.k = k
        self.actionTheta = actionTheta
        self.obspad=(0,0,0,0,0,k)
        self.batched_obspad=(0,0,0,0,0,0,0,k)
        
    def restructure_inputs(self, obs, act, batched=False):
        """
        Join obs and act into a single input tensor shape (N,L,H)
        N: Batch size/Theta Size
        L: timesamps
        H: input_size
        obs should be one timestep longer than act, for the [t+1] observation
        after the last action
        """
        #Apply the action and prediction offsets
        if batched:
            act = self.batched_actpad(act)
            obspad = self.batched_obspad
        else:
            act = self.actpad(act)
            if ~hasattr(self,'obspad'):   #For backwards compadibility - remove later
                self.obspad = (0,0,0,0,0,self.k)
            obspad = self.obspad
        obs_target = obs[:,self.predOffset:,:]
        
        #Apply the theta prediction for target observation
        theta_idx = np.flip(toeplitz(np.arange(self.k+1),
                                     np.arange(obs_target.size(1))),0)
        theta_idx = theta_idx[:,self.k:,]
        obs_target = obs_target[:,theta_idx.copy()]
        obs_target = torch.squeeze(obs_target,0)
        
        if self.actionTheta == 'hold':
            size = [x for x in act.size()] # So it works with batched and non-batched
            size[0] = self.k+1
            act = act.expand(*size)
            obs = nn.functional.pad(input=obs, pad=obspad, 
                                    mode='constant', value=0)


        elif self.actionTheta is True:
            theta_idx = np.flip(toeplitz(np.arange(self.k+1),
                                         np.arange(act.size(1))),0)
            theta_idx = theta_idx[:,self.k:,]
            act = act[:,theta_idx.copy()]
            act = torch.squeeze(act,0)
            obs = nn.functional.pad(input=obs, pad=obspad, 
                                    mode='constant', value=0)
            
        
        #Make everything the same size
        minsize = min(obs.size(1),act.size(1),obs_target.size(1))
        obs, act = obs[:,:minsize,:], act[:,:minsize,:]
        obs_target = obs_target[:,:minsize,:]
        
        #No masks for theta net
        obs_out = torch.zeros_like(obs, requires_grad=False)
        act_out = torch.zeros_like(act, requires_grad=False)
        obs_target_out = torch.zeros_like(obs_target, requires_grad=False)
        outmask = True
        
        obs_out[:] = obs[:]
        act_out[:] = act[:]
        obs_target_out[:] = obs_target[:]

        obs = self.droplayer(obs) #dropout without action
        
        #Concatenate the obs/act into a single input
        x_t = torch.cat((obs_out,act_out), 2)
        return x_t, obs_target_out, outmask


class pRNN_multimodal(pRNN):
    """
    A predictive RNN framework that allows segregation between different types of observations.
    They can be included only in inputs, in outputs, or both.

    obs_size: tuple of sizes for each observation type
    inIDs: tuple of indices of observations to include in input
    outIDs: tuple of indices of observations to include in output

    All inputs should be tensors of shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
    """
    def __init__(self, obs_size, act_size, hidden_size=500,
                 cell=RNNCell,  dropp=0, bptttrunc=50, k=0, f=0.5,
                 predOffset=1, inMask=[True], outMask=None, 
                 actOffset=0, actMask=None, neuralTimescale=2,
                 continuousTheta=False, inIDs=None, outIDs=None,
                 **cell_kwargs):
        self.obs_size = obs_size
        self.inIDs = inIDs
        self.outIDs = outIDs
        n=0
        self.obs_out_slices = []
        for i in outIDs:
            self.obs_out_slices.append(slice(n,n+obs_size[i]))
            n += obs_size[i]

        super(pRNN_multimodal, self).__init__(obs_size, act_size, hidden_size,
                                              cell,  dropp, bptttrunc, k, f,
                                              predOffset, inMask, outMask, 
                                              actOffset, actMask, neuralTimescale,
                                              continuousTheta,
                                              **cell_kwargs)

    
    def create_layers(self, obs_size, act_size, hidden_size,
                      cell, bptttrunc, continuousTheta,
                      k, f, **cell_kwargs):
        #Sparsity via layernorm subtraction
        mu = norm.ppf(f)
        musig = [mu,1]

        self.input_size = 0
        output_size = 0
        for i in self.inIDs:
                self.input_size += obs_size[i]
        for i in self.outIDs:
                output_size += obs_size[i]
        self.input_size += act_size

        self.rnn = thetaRNNLayer(cell, bptttrunc, self.input_size, hidden_size,
                                 defaultTheta=k, continuousTheta=continuousTheta,
                                 musig=musig, **cell_kwargs)
        self.outlayer = nn.Sequential(
            nn.Linear(hidden_size, output_size, bias=False),
            nn.Sigmoid()
            )


    def forward(self, obs, act, noise_params=(0,0), state=torch.tensor([]), theta=None,
                single=False, mask=None, batched=False, fullRNNstate=False):
        #Determine the noise shape
        k=0
        if hasattr(self,'k'):
            k= self.k
        if batched:
            noise_shape = (k+1, act.size(1)+1, self.hidden_size, act.size(-1))
        else:
            noise_shape = (k+1, act.size(1)+1, self.hidden_size)

        noise_t = self.generate_noise(noise_params, noise_shape)

        if single:
            x_t = []
            for i in self.inIDs:
                x_t.append(obs[i])
            x_t = torch.cat((*x_t,act), 2)
            h_t,_ = self.rnn(x_t, internal=noise_t, state=state, theta=theta)
            if not fullRNNstate: 
                h_t = h_t[:,:,:self.hidden_size] #For RNNcells that output more than the hidden RNN units
            y_t = None
            obs_target = None
        else:
            x_t, obs_target, outmask = self.restructure_inputs(obs,act,batched)
            #x_t = self.droplayer(x_t) # (should it count action???) dropout with action
            h_t,_ = self.rnn(x_t, internal=noise_t, state=state,
                             theta=theta, mask=mask, batched=batched)
            if not fullRNNstate: 
                h_t = h_t[:,:,:self.hidden_size] #For RNNcells that output more than the hidden RNN units (ugly)
            if batched:
                h_t = h_t.permute(-1,*[i for i in range(len(h_t.size())-1)])
                allout = self.outlayer(h_t[:,:,:,:self.hidden_size])
                allout = allout.permute(*[i for i in range(1,len(allout.size()))],0)
                h_t = h_t.permute(*[i for i in range(1,len(h_t.size()))],0)
            else:
                allout = self.outlayer(h_t[:,:,:self.hidden_size])

            #Apply the mask to the output
            pred = torch.zeros_like(allout)
            pred[:,outmask,:] = allout[:,outmask,:] #The predicted outputs.
            y_t = [] # Outputs disentangled
            for i in set((*self.inIDs, *self.outIDs)):
                if i in set(self.inIDs)-set(self.outIDs):
                    y_t.append(torch.zeros_like(obs[i]))
                else:
                    y_t.append(pred[...,self.obs_out_slices[self.outIDs.index(i)]])
            y_t = (pred, y_t)
        return y_t, h_t, obs_target

    def restructure_inputs(self, obs, act, batched=False):
        #Apply the action and prediction offsets
        if batched:
            act = self.batched_actpad(act)
        else:
            act = self.actpad(act)
        
        if self.actOffset:
                act = act[:,:-self.actOffset,...]

        # Specify inputs and outputs in the observation
        obs_in = []
        for i in self.inIDs:
            obs_in.append(obs[i])
        obs_in = torch.cat(obs_in, 2)
        obs_target = []
        for i in self.outIDs:
            obs_target.append(obs[i][:,self.predOffset:,:])
        obs_target = torch.cat(obs_target, 2)

        #Make everything the same size
        minsize = min(obs_in.size(1),act.size(1),obs_target.size(1))
        obs_in, act = obs_in[:,:minsize,:], act[:,:minsize,:]
        obs_target = obs_target[:,:minsize,:]

        #Apply the masks (this is ugly.)
        actmask = np.resize(np.array(self.actMask),minsize)
        outmask = np.resize(np.array(self.outMask),minsize)
        obsmask = np.resize(np.array(self.inMask),minsize)

        obs_out = torch.zeros_like(obs_in, requires_grad=False)
        act_out = torch.zeros_like(act, requires_grad=False)
        obs_target_out = torch.zeros_like(obs_target, requires_grad=False)

        obs_out[:,obsmask,:] = obs_in[:,obsmask,:]
        act_out[:,actmask,:] = act[:,actmask,:]
        obs_target_out[:,outmask,:] = obs_target[:,outmask,:]
        
        obs_out = self.droplayer(obs_out) #dropout without action
        #act_out = self.droplayer_act(act_out) #dropout without action

        #Concatenate the obs/act into a single input
        x_t = torch.cat((obs_out,act_out), 2)
        return x_t, obs_target_out, outmask
        
        
        



    
    

class vRNN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell):
        super(vRNN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=None, actMask=None)

class thRNN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell):
        super(thRNN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[False,True], actMask=None)

class vRNN_0win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_0win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)

class vRNN_1win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_1win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,True], outMask=[True,True], actMask=None)

class vRNN_2win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_2win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=None)

class vRNN_3win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_3win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=None)

class vRNN_4win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_4win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=None)

class vRNN_5win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_5win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)

class vRNN_1win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_1win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,False])

class vRNN_2win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_2win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=[True,False,False])

class vRNN_3win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_3win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True], 
                          actMask=[True,False,False,False])

class vRNN_4win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_4win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=[True,False,False,False,False])

class vRNN_5win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_5win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=[True,False,False,False,False,False])



        
        
class thRNN_0win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_0win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)

class thRNN_1win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_1win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=None)

class thRNN_2win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_2win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=None)

class thRNN_3win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_3win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=None)

class thRNN_4win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_4win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=None)

class thRNN_5win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_5win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_6win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_6win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True],
                          actMask=None)
        
        
        
        
        




class thRNN_0win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_0win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)

class thRNN_1win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_1win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=None)

class thRNN_2win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_2win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=None)

class thRNN_3win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_3win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=None)

class thRNN_4win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_4win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=None)

class thRNN_5win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(thRNN_5win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_6win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_6win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_7win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_7win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_8win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_8win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_9win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_9win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_10win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_10win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
        

class thRNN_1win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thRNN_1win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,False])

class thRNN_2win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thRNN_2win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=[True,False,False])

class thRNN_3win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thRNN_3win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=[True,False,False,False])

class thRNN_4win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thRNN_4win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=[True,False,False,False,False])

class thRNN_5win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thRNN_5win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=[True,False,False,False,False,False])

class thRNN_0win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_0win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True], outMask=[True], actMask=None)

class thRNN_1win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_1win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False], outMask=[True,True],
                          actMask=None)

class thRNN_2win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_2win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=None)

class thRNN_3win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_3win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=None)

class thRNN_4win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_4win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=None)

class thRNN_5win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_5win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_6win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_6win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_7win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_7win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_8win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_8win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_9win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_9win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_10win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_10win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True,True,True],
                          actMask=None)






class AutoencoderFF(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=None):
        super(AutoencoderFF, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()
        
class AutoencoderRec(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=None):
        super(AutoencoderRec, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderPred(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=None):
        super(AutoencoderPred, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderFFPred(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=None):
        super(AutoencoderFFPred, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()
        
        
        
class AutoencoderFF_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderFF_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()
        
class AutoencoderRec_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderRec_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderPred_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(AutoencoderPred_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                                                 f=f,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderFFPred_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15, f=0.5):
        super(AutoencoderFFPred_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                                                    cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, 
                                                    dropp=dropp,f=f,
                                                    predOffset=1, actOffset=0,
                                                    inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()

class AutoencoderMaskedO(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderMaskedO, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[False,True],
                          actMask=[True,True])
        
class AutoencoderMaskedOA(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderMaskedOA, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[False,True],
                          actMask=[True,False])

class AutoencoderMaskedO_noout(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderMaskedO_noout, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,True])
        
class AutoencoderMaskedOA_noout(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderMaskedOA_noout, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,False])


        

        
        
class thcycRNN_3win(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thcycRNN_3win, self).__init__(obs_size, act_size,  k=3, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       predOffset=0, actOffset=0,
                                      )
        
class thcycRNN_4win(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thcycRNN_4win, self).__init__(obs_size, act_size,  k=4, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       predOffset=0, actOffset=0,
                                      )
        
class thcycRNN_5win(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thcycRNN_5win, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True)
        
        
        
class thcycRNN_5win_holdc(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_holdc, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta='hold')
        
class thcycRNN_5win_fullc(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_fullc, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta=True)
        
class thcycRNN_5win_firstc(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_firstc, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta=False)
        
        
class thcycRNN_5win_hold(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_hold, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta='hold')
        
class thcycRNN_5win_full(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(thcycRNN_5win_full, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta=True)
        
class thcycRNN_5win_first(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_first, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta=False)
        
        
     
        
class thcycRNN_5win_holdc_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_holdc_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta='hold')
        
class thcycRNN_5win_fullc_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_fullc_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta=True)
        
class thcycRNN_5win_firstc_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_firstc_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta=False)
        
        
class thcycRNN_5win_hold_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_hold_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta='hold')
        
class thcycRNN_5win_full_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_full_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta=True)
        
class thcycRNN_5win_first_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_first_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta=False)
        
        
class thcycRNN_5win_holdc_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_holdc_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=True, actionTheta='hold')
        
class thcycRNN_5win_fullc_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_fullc_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=True, actionTheta=True)
        
class thcycRNN_5win_firstc_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_firstc_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=True, actionTheta=False)
        
        
class thcycRNN_5win_hold_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_hold_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=False, actionTheta='hold')
        
class thcycRNN_5win_full_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_full_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=False, actionTheta=True)
        
class thcycRNN_5win_first_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_first_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=False, actionTheta=False)
        


class lognRNN_rollout(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LogNRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(lognRNN_rollout, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta=True,
                                       **cell_kwargs)

class lognRNN_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LogNRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(lognRNN_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None,
                          **cell_kwargs)
        
        
        
              
        

class vRNN_LayerNorm(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_LayerNorm, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=LayerNormRNNCell)


class thRNN_LayerNorm(thRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(thRNN_LayerNorm, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=LayerNormRNNCell)


class vRNN_LayerNormAdapt(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_LayerNormAdapt, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=AdaptingLayerNormRNNCell)



class vRNN_CANN(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_CANN, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=LayerNormRNNCell)

        #TODO Clean this
        size = [20,20,20]
        Nmaps = 1
        self.locations = CANN_(self.W, size, Nmaps, selfconnect=False)
        with torch.no_grad():
            self.W.add_(torch.eye(hidden_size).mul_(0.5))



class vRNN_adaptCANN(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_adaptCANN, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=AdaptingLayerNormRNNCell)

        #TODO Clean this
        size = [15,15,15]
        Nmaps = 1
        self.locations = CANN_(self.W, size, Nmaps, selfconnect=False)
        with torch.no_grad():
            self.W.add_(torch.eye(hidden_size).mul_(0.5))




class vRNN_CANN_FFonly(vRNN_CANN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_CANN_FFonly, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias)

        rootk = np.sqrt(1/hidden_size)
        with torch.no_grad():
            self.W.add_(torch.rand(hidden_size, hidden_size)*0.5*rootk)
        self.W.requires_grad=False


class vRNN_adptCANN_FFonly(vRNN_adaptCANN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_adptCANN_FFonly, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias)

        rootk = np.sqrt(1/hidden_size)
        with torch.no_grad():
            self.W.add_(torch.rand(hidden_size, hidden_size)*0.2*rootk)
        self.W.requires_grad=False



class sgpRNN_5win(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                 f=0.5,
                 sparse_size=1000, sparse_beta=1,
                 lambda_direct=1, lambda_context=1, lambda_sparse=1):
        super(sgpRNN_5win, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=SparseGatedRNNCell,
                                       continuousTheta=False, actionTheta=True, 
                                       bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       sparse_size=sparse_size, sparse_beta=sparse_beta,
                                       lambda_direct=lambda_direct, lambda_context=lambda_context,
                                       lambda_sparse=lambda_sparse)



class multRNN_5win_i01_o01(pRNN_multimodal):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(multRNN_5win_i01_o01, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None,
                          inIDs=(0,1), outIDs=(0,1),
                          )



class multRNN_5win_i1_o0(pRNN_multimodal):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(multRNN_5win_i1_o0, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None,
                          inIDs=(1,), outIDs=(0,),
                          )



class multRNN_5win_i01_o0(pRNN_multimodal):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(multRNN_5win_i01_o0, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None,
                          inIDs=(0,1), outIDs=(0,),
                          )



class multRNN_5win_i0_o1(pRNN_multimodal):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(multRNN_5win_i0_o1, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None,
                          inIDs=(0,), outIDs=(1,),
                          )