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
    def __init__(self, input_size: int, hidden_size: int, actfun: str = "relu", **kwargs) -> None:
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
    def forward(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]: #tuple of hy
        "Define update of hidden state, h, and return."
        raise NotImplementedError


