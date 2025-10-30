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
from typing import List, Tuple
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

initializations = {
    "xavier": xavier_init(),
    "log_normal": sparse_lognormal_()
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
    def __init__(self, input_size: int, hidden_size: int, actfun: str = "relu", init: str = "xavier", *args, **kwargs):
        super().__init__(input_size, hidden_size, actfun)

        self.initialize_weights(input_size = input_size, hidden_size = hidden_size, init = init, **kwargs)

    def initialize_weights(self, input_size: int, hidden_size: int, init: str, **kwargs):
        self.weight_i: Tensor;
        self.weight_hh: Tensor;
        self.bias: Tensor;
        
        #initialize IN PLACE
        if init == "xavier": 
            xavier_init(input_size, hidden_size, self.weight_ih, self.weight_hh, self.bias)

        if init == "log_normal": 
            #pull out extra keyword args needed for log normal init
            mean_std_ratio = kwargs["mean_std_ratio"] if "mean_std_ratio" in kwargs else 1.
            sparsity = kwargs["sparsity"] if "sparsity" in kwargs else 1.

            sparse_lognormal_(self.weight_ih, mean_std_ratio=mean_std_ratio, sparsity=sparsity)
            sparse_lognormal_(self.weight_hh, mean_std_ratio=mean_std_ratio, sparsity=sparsity)
    
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

    def __init__(self, input_size: int, hidden_size: int, actfun: str = "relu", init = "xavier", b = 0.3, tau_a = 8., *args, **kwargs):
        # initialize class attributes and weights
        super().__init__(input_size, hidden_size, actfun, init) 
        
        self.b = b #gain in adapatation
        self.tau_a = tau_a #decay in adaptation
    
    def forward(self, input: Tensor, state: Tensor, internal: Tensor) -> Tuple[Tensor, Tensor]:
        hx, ax = state
        x = super().update_preactivation(input, hx)

        ay = ax * (1-1/self.tau_a) + (self.b/self.tau_a) *hx #adaptation variable, a_{t} dependent on both h_{t-1} and a_{t-1}
        hy = self.actfun(x + internal + self.bias - ax)
        return hy, (hy,ay)

class LayerNormRNNCell(RNNCell):
    def __init__(self, input_size: int, hidden_size: int, actfun: str = "relu", init = "xavier", mu = 0, sig = 1, *args, **kwargs):
        super().__init__(input_size, hidden_size, actfun, init)
        
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
    def __init__(self, input_size, hidden_size, actfun = "relu", init = "xavier", b=0.3, tau_a=8, *args, **kwargs):
        #set up both adaptation and layernorm stuff
        AdaptingRNNCell.__init__(self, input_size, hidden_size, actfun, init, b, tau_a, *args, **kwargs)
        LayerNormRNNCell.__init__(self, input_size, hidden_size, actfun, init)
    
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
    

## CUTTING OUT SPARSE GATED for now... 

#INIT FUNCTIONS

def xavier_init(input_size, hidden_size, W_ih, W_hh, b):
    """ Modify in place!!
    """
    rootk_h = np.sqrt(1./hidden_size)
    rootk_i = np.sqrt(1./input_size)

    # initialize weights with Xavier/Glorot scheme (equiv to the default PyTorch?)
    W_ih = Parameter(torch.rand(hidden_size, input_size)*2*rootk_i-rootk_i) #scales uniform dist : W ~ U(- sqrt(1/n_size), + sqrt(1/n_size)
    W_hh = Parameter(torch.rand(hidden_size, hidden_size)*2*rootk_h-rootk_h)
    b = Parameter(torch.zeros(hidden_size))

def calc_ln_mu_sigma(mean, var):
    "Given desired mean and var returns ln mu and sigma"
    mu_ln = math.log(mean ** 2 / math.sqrt(mean ** 2 + var))
    sigma_ln = math.sqrt(math.log(1 + (var / mean ** 2)))
    return mu_ln, sigma_ln

def sparse_lognormal_(
    tensor: Tensor,
    gain: float = 1.0,
    mode: str = "fan_in",
    mean_std_ratio: float = 1,
    sparsity: float = 1,
    **ignored
):
    """
    Initializes the tensor with a log normal distribution * {1,-1}. 

    Arguments:
        tensor: torch.Tensor, the tensor to initialize
        gain: float, the gain to use for the initialization stddev calulation.
        mode: str, the mode to use for the initialization. Options are 'fan_in', 'fan_out'
        generator: optional torch.Generator, the random number generator to use. 
        mean_std_ratio: float, the ratio of the mean to std for log_normal initialization.

    Note this function draws from a log normal distribution with mean = mean_std_ratio * std
    and then multiplies the tensor by a random Rademacher dist. variable (impl. with bernoulli). 
    This induces the need to correct the ln std dev, as the final symmetrical distribution
    will have variance = mu^2 + sigma^2 = (1 + mean_std_ratio^2) * sigma^2. Where sigma, mu are
    the log normal distribution parameters.
    """
    if sparsity == 0:
        with torch.no_grad():
            tensor.mul_(0.)
        return

    fan = torch.nn.init._calculate_correct_fan(tensor, mode) * sparsity
    std = gain / math.sqrt(fan)
    #std /= (1+mean_std_ratio**2)**0.5 # Adjust for multiplication with bernoulli  
    mu, sigma = calc_ln_mu_sigma(std * mean_std_ratio, std ** 2)
    with torch.no_grad():
        tensor.log_normal_(mu, sigma)
        #tensor.mul_(2 * torch.bernoulli(0.5 * torch.ones_like(tensor), generator=generator) - 1)  
        if sparsity < 1:
            sparse_mask = torch.rand_like(tensor) < sparsity  
            tensor.mul_(sparse_mask)
            #tensor = tensor.to_sparse()



#Unit Tests

import torch.nn.functional as F
def test_script_thrnn_layer(seq_len, input_size, hidden_size, trunc, theta):
    inp = torch.randn(1,seq_len,input_size)
    inp = F.pad(inp,(0,0,0,theta))
    state = torch.randn(1, 1, hidden_size)
    internal = torch.zeros(theta+1 ,seq_len+theta, hidden_size)
    rnn = thetaRNNLayer(RNNCell, trunc, input_size, hidden_size)
    out, out_state = rnn(inp, internal, state, theta=theta)

    # Control: pytorch native LSTM
    rnn_ctl = nn.RNN(input_size, hidden_size, 
                     batch_first=True, bias=False, nonlinearity = 'relu')

    for rnn_param, custom_param in zip(rnn_ctl.all_weights[0], rnn.parameters()):
        assert rnn_param.shape == custom_param.shape
        with torch.no_grad():
            rnn_param.copy_(custom_param)
    rnn_out, rnn_out_state = rnn_ctl(inp, state)

    #Check the output matches rnn default for theta=0
    assert (out[0,:,:] - rnn_out).abs().max() < 1e-5
    assert (out_state - rnn_out_state).abs().max() < 1e-5
    
    #Check the theta prediction matches the rnn output when input is withheld
    assert (out[:,-theta-1,0] - rnn_out[0,-theta-1:,0]).abs().max() < 1e-5

    return out,rnn_out,inp,rnn

test_script_thrnn_layer(5, 3, 7, 10, 4)