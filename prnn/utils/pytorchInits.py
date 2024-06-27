#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:33:09 2022

@author: dl2820
"""
import torch
from torch import nn
import numpy as np

def init_(tensor, weights):
    with torch.no_grad():
        tensor.fill_(0)
        tensor.add_(weights)
        return tensor
    
        

from prnn.utils.CANNtools import multiCANNmatrix
def CANN_(tensor, size, Nmaps, selfconnect=False, width=1, peak=None, inh=0):
    Ncells = tensor.size(0)
    if peak is None:
        peak = np.e*np.sqrt(1/Ncells)
        
    CANNmatrix,locations = multiCANNmatrix(Ncells,size,Nmaps,selfconnect,
                                           peak=peak,
                                           width=width,
                                           inh=inh)
    # Modify tensor with CANN weight matrix
    init_(tensor, torch.tensor(CANNmatrix))
    return locations
    
    
    
    
    
    
    
    
