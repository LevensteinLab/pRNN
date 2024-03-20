#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:01:17 2022

modeled after: 
https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py

@author: dl2820
"""

import torch.nn as nn
from torch.nn import Parameter
import torch
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor
import numbers
import numpy as np

#class RNNLayer(jit.ScriptModule):
class RNNLayer(nn.Module):    
    """
    A wrapper for customized RNN layers... inputs should match torch.nn.RNN
    conventions for batch_first=True
    """
    def __init__(self, cell, trunc, *cell_args):
        super(RNNLayer, self).__init__()
        self.cell = cell(*cell_args)
        self.trunc = trunc
        

    #@jit.script_method
    def forward(self, input: Tensor=torch.tensor([]),
                internal: Tensor=torch.tensor([]),
                state: Tensor=torch.tensor([])) -> Tuple[Tensor, Tensor]:
        
        if input.size(0)==0:
            input = torch.zeros(internal.size(0),internal.size(1),self.cell.input_size,
                                device=self.cell.weight_hh.device)
        if state.size(0)==0:
            state = torch.zeros(1,input.size(0),self.cell.hidden_size,
                                device=self.cell.weight_hh.device)
        if internal.size(0)==0: # TODO: check this
            internal = torch.zeros(1,input.size(1),
                                   device=self.cell.weight_hh.device)
            
        inputs = input.unbind(1)
        internals = internal.unbind(1)
        state = (torch.squeeze(state,0),0) #To match RNN builtin
        #outputs = torch.jit.annotate(List[Tensor], [])
        outputs = []
        
        for i in range(len(inputs)):
            if hasattr(self,'trunc') and np.mod(i,self.trunc)==0:
                state = (state[0].detach(),) #Truncated BPTT
                
            out, state = self.cell(inputs[i], internals[i], state)
            #Here: loop theta (don't change state... consider: loop above but then how not change state?)
            #Question: what to do about internals?...
            outputs += [out]
            
            # TODO: adjust out, state, outputs inputs, and return to match nn.RNN
        state = torch.unsqueeze(state[0],0) #To match RNN builtin
        return torch.stack(outputs,1), state
        
        
        

#class RNNCell(jit.ScriptModule):
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, musig=None):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #Default Initalization 
        rootk = np.sqrt(1/hidden_size)
        self.weight_ih = Parameter(torch.rand(hidden_size, input_size)*2*rootk-rootk)
        self.weight_hh = Parameter(torch.rand(hidden_size, hidden_size)*2*rootk-rootk)
        # The layernorms provide learnable biases
        
        #TODO: Add option for torch.sigmoid or torch.tanh
        self.actfun = torch.nn.ReLU()

    # TODO: with and without history (-h) 
    #@jit.script_method
    def forward(self, input: Tensor, internal: Tensor, state: Tensor) -> Tensor:
        hx = state[0]
        i_input = torch.mm(input, self.weight_ih.t())
        h_input = torch.mm(hx, self.weight_hh.t())
        x = i_input + h_input
        hy = self.actfun(x + internal)
        return hy, (hy,)



#class LayerNormRNNCell(jit.ScriptModule):
class LayerNormRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, musig=[0,1]):
        super(LayerNormRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #Default Initalization 
        rootk = np.sqrt(1/hidden_size)
        self.weight_ih = Parameter(torch.rand(hidden_size, input_size)*2*rootk-rootk)
        self.weight_hh = Parameter(torch.rand(hidden_size, hidden_size)*2*rootk-rootk)
        # The layernorms provide learnable biases
        
        self.layernorm = LayerNorm(hidden_size,musig)
        #TODO: Add option for torch.sigmoid or torch.tanh
        self.actfun = torch.nn.ReLU()

    # TODO: with and without history (-h) 
    #@jit.script_method
    def forward(self, input: Tensor, internal: Tensor, state: Tensor) -> Tensor:
        hx = state[0]
        i_input = torch.mm(input, self.weight_ih.t())
        h_input = torch.mm(hx, self.weight_hh.t())
        x = self.layernorm(i_input + h_input)
        hy = self.actfun(x + internal)
        return hy, (hy,)



class AdaptingLayerNormRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, musig=[0,1]):
        super(AdaptingLayerNormRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #Default Initalization 
        rootk = np.sqrt(1/hidden_size)
        self.weight_ih = Parameter(torch.rand(hidden_size, input_size)*2*rootk-rootk)
        self.weight_hh = Parameter(torch.rand(hidden_size, hidden_size)*2*rootk-rootk)
        self.b = Parameter(torch.ones(1)*1)
        self.tau_a = Parameter(torch.ones(1)*4)
        # The layernorms provide learnable biases
        
        self.layernorm = LayerNorm(hidden_size,musig)
        #TODO: Add option for torch.sigmoid or torch.tanh
        self.actfun = torch.nn.ReLU()

    # TODO: with and without history (-h) 
    #@jit.script_method
    def forward(self, input: Tensor, internal: Tensor, state: Tensor) -> Tensor:
        hx, ax = state
        i_input = torch.mm(input, self.weight_ih.t())
        h_input = torch.mm(hx, self.weight_hh.t())
        x = i_input + h_input
        x = self.layernorm(i_input + h_input)
        # TODO check time indices
        ay = ax * (1-1/self.tau_a) + self.b/self.tau_a *hx
        hy = self.actfun(x + internal - ax)
        return hy, (hy,ay)



#class LayerNorm(jit.ScriptModule):
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, musig):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1

        self.mu = musig[0]
        self.sig = musig[1]
        self.normalized_shape = normalized_shape

    #@jit.script_method
    def compute_layernorm_stats(self, input):
        mu = input.mean(-1, keepdim=True)
        sigma = input.std(-1, keepdim=True, unbiased=False) + 0.0001
        return mu, sigma

    #@jit.script_method
    def forward(self, input):
        mu, sigma = self.compute_layernorm_stats(input)
        return (input - mu) / sigma * self.sig + self.mu
    
    
    
#Unit Tests
#Tested shapes and RNN without norm........ need way to test LayerNormRNNCell
def test_script_rnn_layer(seq_len, batch, input_size, hidden_size, trunc):
    inp = torch.randn(batch,seq_len,input_size)
    state = torch.randn(1, batch, hidden_size)
    internal = torch.zeros(batch,seq_len, hidden_size)
    rnn = RNNLayer(RNNCell, trunc, input_size, hidden_size)
    out, out_state = rnn(inp, internal, state)

    # Control: pytorch native LSTM
    rnn_ctl = nn.RNN(input_size, hidden_size, 
                     batch_first=True, bias=False, nonlinearity = 'relu')

    for rnn_param, custom_param in zip(rnn_ctl.all_weights[0], rnn.parameters()):
        assert rnn_param.shape == custom_param.shape
        with torch.no_grad():
            rnn_param.copy_(custom_param)
    rnn_out, rnn_out_state = rnn_ctl(inp, state)

    assert (out - rnn_out).abs().max() < 1e-5
    assert (out_state - rnn_out_state).abs().max() < 1e-5


test_script_rnn_layer(5, 2, 3, 7, 2)
