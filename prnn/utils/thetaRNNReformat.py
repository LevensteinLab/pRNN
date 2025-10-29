#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:23:45 2025

modeled after: 
https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py

@author: vv266, dl2820
"""

import torch.nn as nn
from torch.nn import Parameter, ReLU, Sigmoid, Tanh
from torch import Tensor
#import torch.jit as jit
from typing import List, Tuple, String
from torch import Tensor
import numbers
import numpy as np
import math

import torch.nn.utils.prune as prune


activations = {
    "relu": torch.nn.ReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh()
}

from abc import ABC, abstractmethod # abstract-base classes in python

class BaseCell(nn.Module, ABC):
    """
    Abstract class that defines a generic cell; contains an class initialization, 
    a weight initialization function, and a forward/state update method.
    """    
    def __init__(self, input_size: int, hidden_size: int, actfun: str = "relu", *args, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if actfun not in activations:
            raise ValueError(f"Unknown activation '{actfun}'.")
        self.actfun = activations[actfun]

    @abstractmethod
    def initialize_weights(self) -> None:
        "Initialize weights, likely with Xavier (Glorot) scheme"
        raise NotImplementedError

    #does not need to be implemented in all custom implementations
    def update_preactivation(self, input: Tensor, hx: Tensor, *args, **kwargs) -> Tensor: #just x
        "Use input at this time step and current h to generate x, before normalization, bias, and activation"
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, input: Tensor, state: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]: #tuple of hy
        "Apply some/all of the following to x: bias, noise, adaptation, activation function"
        raise NotImplementedError

class RNNCell(BaseCell):
    """
    Parent class that defines the base recurrent cell in the predictive RNN...
    This class initializes two weight matrices (one from input --> hidden and one from hidden --> hidden),
    a bias term, and an activation function of choice.

    Args:
        input_size (int): length of input vector
        hidden_size (int): length of hidden state
        musig (Tuple[float, float]): length of state
    Returns:
        Tuple(Tensor, Tuple(Tensor)): updated hidden state, hy
    """
    def __init__(self, input_size: int, hidden_size: int, actfun: str = "relu", *args, **kwargs):
        super().__init__(input_size, hidden_size, actfun)
        self.initialize_weights(input_size = input_size, hidden_size = hidden_size)    


    def initialize_weights(self, input_size: int, hidden_size: int):
        rootk_h = np.sqrt(1./hidden_size)
        rootk_i = np.sqrt(1./input_size)

        # initialize weights with Xavier/Glorot scheme (equiv to the default PyTorch?)
        self.weight_ih = Parameter(torch.rand(hidden_size, input_size)*2*rootk_i-rootk_i) #scales uniform dist : W ~ U(- sqrt(1/n_size), + sqrt(1/n_size)
        self.weight_hh = Parameter(torch.rand(hidden_size, hidden_size)*2*rootk_h-rootk_h)
        self.bias = Parameter(torch.zeros(hidden_size))
    
    def update_preactivation(self, input, hx, *args, **kwargs) -> Tensor:
        i_input = torch.mm(input, self.weight_ih.t())
        h_input = torch.mm(hx, self.weight_hh.t())        
        x = i_input + h_input
        return x
        
    def forward(self, input: Tensor, state: Tensor, internal: Tensor) -> Tuple[Tensor, Tensor]: #tuple of hy
        hx = state[0]
        x = self.update_preactivation(input, hx)
        hy = self.actfun(x + internal + self.bias) # canonical RNN hidden state update (hx --> hy) but with an *internal* term
        return hy, (hy,)

class AdaptingRNNCell(RNNCell):

    def __init__(self, input_size: int, hidden_size: int, actfun: str = "relu", b = 0.3, tau_a = 8., *args, **kwargs):
        # initialize class attributes and weights
        super().__init__(input_size, hidden_size, actfun) 
        
        self.layernorm.mu = Parameter(torch.zeros(self.hidden_size)+self.layernorm.mu)

        self.b = b #gain in adapatation
        self.tau_a = tau_a #decay in adaptation
    
    def forward(self, input: Tensor, state: Tensor, internal: Tensor) -> Tuple[Tensor, Tensor]:
        hx, ax = state
        x = super().update_preactivation(input, hx)

        ay = ax * (1-1/self.tau_a) + (self.b/self.tau_a) *hx #adaptation variable, a_{t} dependent on both h_{t-1} and a_{t-1}
        hy = self.actfun(x + internal + self.bias - ax)
        return hy, (hy,ay)

class LayerNormRNNCell(RNNCell):
    def __init__(self, input_size: int, hidden_size: int, actfun: str = "relu", mu = 0, sig = 1, *args, **kwargs):
        super().__init__(input_size, hidden_size, actfun)
        
        #set up layernorm
        self.layernorm = LayerNorm(hidden_size, mu, sig)
        self.layernorm.mu = Parameter(torch.zeros(self.hidden_size)+self.layernorm.mu)
        self.bias = self.layernorm.mu
    
    def forward(self, input:Tensor, state:Tensor, internal:Tensor):
        hx = state[0]
        x = self.layernorm(super().update_preactivation(input, hx)) #apply layernorm to output of preactivation
        hy = self.actfun(x + internal)
        return hy, (hy, )

# inherits from both adapting and layernorm
class AdaptingLayerNormRNNCell(AdaptingRNNCell, LayerNormRNNCell):
    def __init__(self, input_size, hidden_size, actfun = "relu", b=0.3, tau_a=8, *args, **kwargs):
        #set up both adaptation and layernorm stuff
        AdaptingRNNCell.__init__(input_size, hidden_size, actfun, b, tau_a, *args, **kwargs)
        LayerNormRNNCell.__init__(input_size, hidden_size, actfun)
    
    def forward(self, input: Tensor, state:Tensor, internal: Tensor):
        hx, ax = state
        x = self.layernorm(super().update_preactivation(input, hx))
        ay = ax * (1-1/self.tau_a) + self.b/self.tau_a *hx
        hy = self.actfun(x + internal - ax)
        return hy, (hy,ay)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, mu, sig):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1

        self.mu = mu
        self.sig = sig
        self.normalized_shape = normalized_shape

    def compute_layernorm_stats(self, input):
        mu = input.mean(-1, keepdim=True)
        sigma = input.std(-1, keepdim=True, unbiased=False) + 0.0001
        return mu, sigma

    def forward(self, input):
        mu, sigma = self.compute_layernorm_stats(input)
        return (input - mu) / sigma * self.sig + self.mu
    
