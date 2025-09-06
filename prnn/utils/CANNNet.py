#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
import torch
from prnn.utils.pytorchInits import CANN_, init_
from prnn.utils.predictiveNet import PredictiveNet
from scipy.spatial.distance import cdist
import numpy as np

#from utils.LayerNormRNN import RNNLayer, RNNCell, LayerNormRNNCell, AdaptingLayerNormRNNCell
from prnn.utils.thetaRNN import thetaRNNLayer, LayerNormRNNCell


class CANNnet(PredictiveNet):
    def __init__(self, env, hidden_size=300, mapsize = [18,18], selfconnect=False,
                 neuralTimescale = 1, cell=LayerNormRNNCell, connectionWidth=1,
                 inputWidth=3, inputStrength=6, untunedInput=5,
                 trainNoiseMeanStd = (0,0), inhibition=0):
        self.trainNoiseMeanStd = trainNoiseMeanStd
        
        self.suppObs = None
        #not actually used, but needed for some placeholders
        self.EnvLibrary = []
        self.env_shell = env
        self.act_size = env.getActSize()
        self.obs_size = env.getObsSize()
        self.obs_shape = env.obs_shape
        self.addEnvironment(env)
        
        self.hidden_size = hidden_size
        self.pRNN = CANNRNN(hidden_size=hidden_size, mapsize = mapsize,
                            cell=cell, neuralTimescale=neuralTimescale,
                            selfconnect=selfconnect, connectionWidth=connectionWidth,
                            inputWidth=inputWidth, inputStrength=inputStrength,
                            untunedInput=untunedInput, inhibition=inhibition)
        
        self.current_state = torch.tensor([])
        
    def predict(self, obs, act, state):
        #A full behavioral sequence of state (obs/act aren't used, but are here to 
        #keep I/O same as PredictiveNet class)
        device = self.pRNN.W.device
        state = self.env2pred_state(state)
        if hasattr(self,'trainNoiseMeanStd') and self.trainNoiseMeanStd != (0,0):
            noise = self.trainNoiseMeanStd
            timesteps = obs.size(1)
            noise_t = noise[0] + noise[1]*torch.randn((1,timesteps,self.hidden_size),
                                                                    device=device)
        else:
            noise_t = torch.tensor([])
    
        obs_pred, h, obs_next, self.current_state = self.pRNN(state, None, noise_t, self.current_state)
        
        return obs_pred, obs_next, h
    
    
    def env2pred_state(self, state):
        """
        Convert state input from gym environment format to pytorch
        arrays for input to the CANN net, tensor of shape (B,L,D)
        B: Batch
        L: timesamps
        D: dimension of the CANN
        https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        """
        if isinstance(state, dict):
            state = torch.tensor(state['agent_pos'][:-1,:],dtype=torch.float,requires_grad=False)
        elif isinstance(state, tuple):
            state = torch.tensor(state,dtype=torch.float, requires_grad=False, device=self.pRNN.W.device)[None]
        else:
            state = torch.tensor(state[:-1,:],dtype=torch.float,requires_grad=False)
        #state = torch.unsqueeze(state, dim=0)
        return state
    

class CANNRNN(nn.Module):
    def __init__(self, hidden_size=300, mapsize = [18,18], selfconnect=False,
                 cell=LayerNormRNNCell, neuralTimescale=1, connectionWidth=1,
                 inputWidth=3, inputStrength=6, untunedInput=5, peakWeight=None, inhibition=0):
        super(CANNRNN, self).__init__()
        
        self.rnn = thetaRNNLayer(cell, 1e8, hidden_size, hidden_size)
        
        self.W_in = self.rnn.cell.weight_ih
        self.W = self.rnn.cell.weight_hh
        
        self.neuralTimescale = neuralTimescale
        self.inputWidth = inputWidth
        self.inputStrength = inputStrength
        self.untunedInput = untunedInput
        self.connectionWidth = connectionWidth
        
        #Initialize input connections to unity 
        #(will calculate input based on position)
        init_(self.W_in,torch.eye(hidden_size))
        
        #Initalize Recurrent connections
        Nmaps = 1
        self.locations = CANN_(self.W, mapsize, Nmaps, selfconnect=selfconnect, width=connectionWidth, peak=peakWeight, inh=inhibition)
        with torch.no_grad():
            self.W.add_(torch.eye(hidden_size).mul_(1-1/self.neuralTimescale))
            
    def forward(self, position, act, noise_t=torch.tensor([]), state=torch.tensor([])):
        #Note: action isn't use
        x_t = self.position2input(position)
        h_t, s_t = self.rnn(x_t, internal=noise_t, state=state)
        obs_pred = None
        obs_next = None
        return obs_pred, h_t, obs_next, s_t
    
    def internal(self, noise_t):
        h_t,_ = self.rnn(internal=noise_t)
        return _, h_t
    
    def position2input(self, position):
        #From Tsodyks1999
        #I0: untuned input
        #Lambda: tuned input gain
        #Tuning_width: width of input tuning (recall, recurrent width=1...)
        I0 = self.untunedInput
        lambd = self.inputStrength
        tuning_width = self.inputWidth 
        
        cellinput = I0+lambd*np.exp(-cdist(position,self.locations[0])
                                       /tuning_width)
        cellinput = torch.tensor(cellinput,dtype=torch.float,requires_grad=False)
        cellinput = torch.unsqueeze(cellinput, dim=0)
        return cellinput