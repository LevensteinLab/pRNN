#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:23:45 2025

modeled after: 
https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py

@author: vv266, dl2820
"""

import torch
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

from abc import ABC, abstractmethod

#INIT FUNCTIONS

def xavier_init(input_size: int, hidden_size:int, W_ih: Tensor, W_hh: Tensor, b: float):
    """Function to initialize weights. 
    Weights are drawn uniformly with bounds proportional to number of inputs/outputs.
    """
    rootk_h = np.sqrt(1./hidden_size)
    rootk_i = np.sqrt(1./input_size)

    # initialize weights with Xavier/Glorot scheme (equiv to the default PyTorch?)
    W_ih = Parameter(torch.rand(hidden_size, input_size)*2*rootk_i-rootk_i) #scales uniform dist : W ~ U(- sqrt(1/n_size), + sqrt(1/n_size)
    W_hh = Parameter(torch.rand(hidden_size, hidden_size)*2*rootk_h-rootk_h)
    b = Parameter(torch.zeros(hidden_size))

    return W_ih, W_hh, b

def calc_ln_mu_sigma(mean: float, var: float):
    """Calculates mu and sigma parameters for use by lognormal initialization.
    """
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

    return tensor

activations = {
    "relu": torch.nn.ReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh()
}

class BaseCell(nn.Module, ABC):
    """
    Abstract Base Class (ABC) that defines a generic cell.
    Contains a constructor, a weight initialization function, and a forward/state update method.
    Optionally, a function to transform inputs before application of an activation function.
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
    def forward(self, input: Tensor, internal: Tensor, state: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]: #tuple of hy
        "Apply some/all of the following to x: bias, noise, adaptation, activation function"
        raise NotImplementedError

class RNNCell(BaseCell):
    """
    Parent class that defines the base recurrent cell in the predictive RNN
    This class initializes two weight matrices (one from input --> hidden and one from hidden --> hidden),
    a bias term, and an activation function of choice.
    """
    def __init__(self, input_size: int, hidden_size: int, actfun: str = "relu", init: str = "xavier", *args, **kwargs):
        """
        Args:
            input_size (int): length of flattened input vector
            hidden_size (int): number of hidden units
            actfun (str, optional): Choice of activation function. Options: "relu", "sigmoid", "tanh". Defaults to "relu".
            init (str, optional): Initialization scheme. Options: "xavier", "log_normal". Defaults to "xavier".
        """
        super().__init__(input_size, hidden_size, actfun)
        init = kwargs["init"] if "init" in kwargs else init
        self.initialize_weights(input_size = input_size, hidden_size = hidden_size, init = init, **kwargs)


    def initialize_weights(self, input_size: int, hidden_size: int, init: str, **kwargs):
        """
        Initialize weights for input, recurrent, and bias parameters

        Args:
            input_size (int): Length of flattened input vector
            hidden_size (int): Number of hidden units.
            init (str): Initialization scheme. Options: "xavier", "log_normal".
            **kwargs: Additional params for init
                For "log_normal":
                    mean_std_ratio: Controls log-normal mean/std balance. Default 1.
                    sparsity: Fraction of nonzero connections in initialization. Default 1. 
        """
        #init weights
        self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = Parameter(torch.zeros(hidden_size))
        
        #initialize 
        if init == "xavier": 
            self.weight_ih, self.weight_hh, self.bias = xavier_init(input_size, hidden_size, self.weight_ih, self.weight_hh, self.bias)

        if init == "log_normal": 
            print('hello')
            #pull out extra keyword args needed for log normal init
            mean_std_ratio = kwargs["mean_std_ratio"] if "mean_std_ratio" in kwargs else 1.
            sparsity = kwargs["sparsity"] if "sparsity" in kwargs else 1.

            self.weight_ih = sparse_lognormal_(self.weight_ih, mean_std_ratio=mean_std_ratio, sparsity=sparsity)
            self.weight_hh = sparse_lognormal_(self.weight_hh, mean_std_ratio=mean_std_ratio, sparsity=sparsity)
    
    def update_preactivation(self, input, hx, *args, **kwargs) -> Tensor:
        """
        Apply weight matricies to input and previous state.
        """

        i_input = torch.mm(input, self.weight_ih.t())
        h_input = torch.mm(hx, self.weight_hh.t())        
        x = i_input + h_input
        return x
        
    def forward(self, input: Tensor, internal: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]: #tuple of hy
        """
        Add bias and apply activation function.
        """
        hx = state[0]
        x = self.update_preactivation(input, hx)
        hy = self.actfun(x + internal + self.bias) # canonical RNN hidden state update (hx --> hy) but with an *internal* term
        return hy, (hy,)

class AdaptingRNNCell(RNNCell):
    """
    RNN Cell that uses an adaptation variable ("ay") dependent on both h_{t-1} and a_{t-1}.
    Inherits from RNNCell.
    """

    def __init__(self, input_size: int, hidden_size: int, actfun: str = "relu", init = "xavier", b = 0.3, tau_a = 8., *args, **kwargs):
        """
        Args:
            Inherits input_size, hidden_size, actfun, init from RNNCell.
            b (float, optional): Gain in adaptation. Defaults to 0.3.
            tau_a (_type_, optional): Decay in adaptation. Defaults to 8..
        """
        # initialize class attributes and weights
        super().__init__(input_size, hidden_size, actfun, init) 
        
        self.b = b
        self.tau_a = tau_a
    
    def forward(self, input: Tensor, internal: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Calculate adaptation term, apply bias and adaptation, apply activation function.
        """
        hx, ax = state
        x = super().update_preactivation(input, hx)

        ay = ax * (1-1/self.tau_a) + (self.b/self.tau_a) *hx #adaptation variable, a_{t} dependent on both h_{t-1} and a_{t-1}
        hy = self.actfun(x + internal + self.bias - ax)
        return hy, (hy,ay)

class LayerNormRNNCell(RNNCell):
    """
    RNN Cell that applies LayerNorm before activation function.
    Inherits from RNNCell.
    """
    def __init__(self, input_size: int, hidden_size: int, actfun: str = "relu", init = "xavier", mu = 0, sig = 1, *args, **kwargs):
        """Constructor.

        Args:
            Inherits input_size, hidden_size, actfun, init from RNNCell.
            mu (int, optional): Mean for LayerNorm. Defaults to 0.
            sig (int, optional): Std dev for LayerNorm. Defaults to 1.
        """
        super().__init__(input_size, hidden_size, actfun, init)
        
        #set up layernorm
        self.layernorm = LayerNorm(hidden_size, mu, sig)
        self.layernorm.mu = Parameter(torch.zeros(self.hidden_size)+self.layernorm.mu)
        self.bias = self.layernorm.mu
    
    def forward(self, input:Tensor, internal:Tensor, state:Tensor):
        """
        Apply LayerNorm before activation function. No adaptation.
        """
        hx = state[0]
        x = self.layernorm(super().update_preactivation(input, hx)) #apply layernorm to output of preactivation
        hy = self.actfun(x + internal)
        return hy, (hy, )

# inherits from both adapting and layernorm
class AdaptingLayerNormRNNCell(AdaptingRNNCell, LayerNormRNNCell):
    """
    RNN Cell that BOTH applies LayerNorm and adds an adaptation term before activation function.
    Multiple inheritance from AdaptingRNNCell and LayerNormRNNCell.
    """
    def __init__(self, input_size, hidden_size, actfun = "relu", init = "xavier", b=0.3, tau_a=8, *args, **kwargs):
        #set up both adaptation and layernorm stuff
        AdaptingRNNCell.__init__(self, input_size, hidden_size, actfun, init, b, tau_a, *args, **kwargs)
        LayerNormRNNCell.__init__(self, input_size, hidden_size, actfun, init)
    
    def forward(self, input: Tensor, internal:Tensor, state: Tensor):
        hx, ax = state
        x = self.layernorm(super().update_preactivation(input, hx))
        ay = ax * (1-1/self.tau_a) + self.b/self.tau_a *hx
        hy = self.actfun(x + internal - ax)
        return hy, (hy,ay)


class LayerNorm(nn.Module):
    """
    Class for LayerNorm object.
    """
    def __init__(self, normalized_shape, mu, sig):
        """Constructor.

        Args:
            normalized_shape (int): Size of LayerNorm ? #TODO Confirm with Dan
            mu (float): Mean
            sig (float): Std.Dev
        """
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
    
class thetaRNNLayer(nn.Module):    
    """
    A wrapper for customized RNN layers... inputs should match torch.nn.RNN
    conventions for batch_first=True, with theta taking the place of the batch dimension
    theta x sequence x neuron 
    #in the future, can make batch x sequence x neuron x theta...
    
    forward method inputs:
    input  [1 x sequence x neuron] (optional if internal provided)
    internal [theta x sequence x neuron] (optional if input provided)
    state [#TODO: DL fill this in] (optional)
    """
    def __init__(self, cell, trunc, *cell_args, defaultTheta=0, continuousTheta=False, **cell_kwargs): #the init scheme (xavier vs lognormal) will get propagated through here
        super(thetaRNNLayer, self).__init__()
        self.cell = cell(*cell_args,**cell_kwargs)
        self.trunc = trunc
        self.theta = defaultTheta
        self.continuousTheta = continuousTheta #whether or not to go from last theta = Theta to t+1 time step, v.s. return back to theta = 0 to t+1 
        
        #self.trunc = cell_kwargs["bptttrunc"] if "bptttrunc" in cell_kwargs else 50

    def preprocess_inputs(self, input, internal, state, theta, batched: bool):
        """
        Perform input validation, initialize vectors to store input, state, and internal vectors, if not done already.
        Also add padding for batches.
        """
        #ensure that there's at least one: input sequences or noise (internal) driving the network 
        assert not(input.size(0)==0 and internal.size(0)==0), "RNN should be driven by input and/or noise."
        
        #if any vars are missing, init with zeros and correct size
        if input.size(0)==0:
            input = torch.zeros(internal.size(0),internal.size(1),self.cell.input_size,
                                device=self.cell.weight_hh.device)
        if state.size(0)==0:
            state = torch.zeros(1,1,self.cell.hidden_size,
                                device=self.cell.weight_hh.device)
        if internal.size(0)==0:
            internal = torch.zeros(theta+1,input.size(1),self.cell.hidden_size,
                                   device=self.cell.weight_hh.device)
        
        #add padding for batches
        if input.size(0)<theta+1: # For "first-only" action inputs
            if batched:
                pad=(0,0,0,0,0,0,0,theta)
            else:
                pad=(0,0,0,0,0,theta)
            input = nn.functional.pad(input=input, pad=pad, 
                                    mode='constant', value=0)   
        
        return input, internal, state

    #@jit.script_method
    def forward(self, input: Tensor=torch.tensor([]),
                internal: Tensor=torch.tensor([]),
                state: Tensor=torch.tensor([]),
                theta=None,
                mask=None,
                batched=False) -> Tuple[Tensor, Tensor]:
        """Defines forward through the RNN Layer.
        Loops thorugh inputs (regardless of batching), and for each input,
        Loops through the rollouts. ("theta" here is "k" in Architectures.)

        Args:
            input (Tensor, optional): Input Tensor. Defaults to torch.tensor([]).
            internal (Tensor, optional): Noise Tensor/"internal replay(?)". Defaults to torch.tensor([]).
            state (Tensor, optional): Recurrent units. Defaults to torch.tensor([]).
            theta (int, optional): Number of rollouts. Defaults to None.
            mask (list(boolean), optional): Masks to cover input observations. Defaults to None.
            batched (bool, optional): Inputs batched? Defaults to False.

        Returns:
            Tuple[Tensor, Tensor]: _description_
        """
        if theta is None:
            theta = self.theta

        input, internal, state = self.preprocess_inputs(input, internal, state, theta, batched)
        
        #Consider unbind twice, rather than indexing later...
        inputs = input.unbind(1)
        internals = internal.unbind(1)
        state = (torch.squeeze(state,1),0) #To match RNN builtin
        #outputs = torch.jit.annotate(List[Tensor], [])
        outputs = []
        
        n = 0
        for i in range(len(inputs)): # loop over all inputs in the sequence
            if np.mod(n,self.trunc)==0 and n>0:
                #state = (state[0].detach(),) #Truncated BPTT
                state = [i.detach() for i in state]

            if batched:
                out, state = self.cell(inputs[i][0,:].permute(1,0), 
                                       internals[i][0,:].permute(1,0),
                                       state)
                out = out.unsqueeze(1)
                out = out.permute(*[i for i in range(1,len(out.size()))],0)
            else:
                out, state = self.cell(torch.unsqueeze(inputs[i][0,:],0), 
                                       internals[i][0,:], state)
                
            state_th = state #first theta = 0 get state...
            out = [out]
            for th in range(theta): # Theta-cycle inside the sequence (1 timestep) #loop over all theta steps within the time step
                if batched:
                    out_th, state_th = self.cell(inputs[i][th+1,:].permute(1,0),
                                                 internals[i][th+1,:].permute(1,0),
                                                 state_th) #get output of cell 
                    out_th = out_th.unsqueeze(1)
                    out_th = out_th.permute(*[i for i in range(1,len(out_th.size()))],0)
                else:
                    out_th, state_th = self.cell(torch.unsqueeze(inputs[i][th+1,:],0), 
                                                 internals[i][th+1,:], state_th)
                out += [out_th]
            out = torch.cat(out,0)
            
            if hasattr(self,'continuousTheta') and self.continuousTheta:
                state = state_th # Theta state is inherited at the next timestep (t+1)
                
            outputs += [out]
            n += 1
        
        state = torch.unsqueeze(state[0],0) #To match RNN builtin
        return torch.stack(outputs,1), state


## CUTTING OUT SPARSE GATED for now...

#Unit Tests

import torch.nn.functional as F
def test_script_thrnn_layer(seq_len, input_size, hidden_size, trunc, theta):
    """
    Compares thetaRNNLayer output to PyTorch native LSTM output.
    """
    inp = torch.randn(1,seq_len,input_size)
    inp = F.pad(inp,(0,0,0,theta))
    state = torch.randn(1, 1, hidden_size)
    internal = torch.zeros(theta+1 ,seq_len+theta, hidden_size)
    rnn = thetaRNNLayer(RNNCell, trunc, input_size, hidden_size)
    out, out_state = rnn(inp, internal, state, theta=theta) #NEED TO CHANGE FROM (inp, internal, state, theta=theta) to (inp, state, internal, theta=theta)
    #out, out_state = rnn(inp, state, internal, theta=theta) 

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