#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:49:31 2022

@author: dl2820
"""

import numpy as np


def expKernel(dist, width=1):
    return np.exp(-dist / width)


def periodicDist(x, size):
    dist = np.zeros((np.size(x, 0), np.size(x, 0)))  # n x n cells
    for d in range(len(size)):
        thisaxis = x[:, d]
        dx = np.abs(thisaxis.reshape(-1, 1) - thisaxis)
        dx_wrap = np.abs(dx - size[d])
        dist += np.minimum(dx, dx_wrap) ** 2  # For periodic distance
    dist = np.sqrt(dist)
    return dist


def CANNmatrix(Ncells, size, selfconnect=False, peak=1, width=1, inh=0):
    """
    Size: N-D list
    """
    # Assign random locations
    D = len(size)
    locations = np.random.rand(Ncells, D) * size

    # Calculate distance matrix
    distance = periodicDist(locations, size)
    # Apply kernel to distance matrix
    weight = (
        expKernel(distance, width) * peak - inh
    )  # The weight exponentially decays with the distance
    weight = np.maximum(0, weight)

    # Remove self-connections
    if selfconnect is False:
        np.fill_diagonal(weight, 0)

    return weight, locations


def multiCANNmatrix(Ncells, size, Nmaps, selfconnect=False, peak=1, width=1, inh=0):
    weights = np.zeros((Ncells, Ncells))
    locations = []
    for mmap in range(Nmaps):
        w, loc = CANNmatrix(
            Ncells, size, selfconnect=selfconnect, peak=peak, width=width, inh=inh
        )
        weights += w
        locations.append(loc)
    weights = weights / Nmaps
    return weights, locations
