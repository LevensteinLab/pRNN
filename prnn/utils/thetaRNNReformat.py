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

    @abstractmethod
    def forward(self, input: Tensor, state: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]: #tuple of hy
        "Define update of hidden state, h, and return."
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
        
    #@jit.script_method
    def forward(self, input: Tensor, state: Tensor, internal: Tensor) -> Tuple[Tensor, Tensor]: #tuple of hy
        """ Update of hidden state, h, of the model (hx --> hy, i.e. h_{t-1} --> h_{t})
        """
        hx = state[0]
        i_input = torch.mm(input, self.weight_ih.t()) #TODO: is matrix multiply necessary here, or can we use @
        h_input = torch.mm(hx, self.weight_hh.t())

        x = i_input + h_input
        hy = self.actfun(x + internal + self.bias) # canonical RNN hidden state update (hx --> hy) but with an *internal* term
        return hy, (hy,)