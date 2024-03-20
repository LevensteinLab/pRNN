#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:35:12 2022

@author: dl2820
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.general import delaydist,fit_exp_linear,kl_divergence

def calculateCoverage(trajectory, envbounds, showFig=False, mask=None):
    """
    envbounds: [minx maxx miny maxy]
    """
    
    binedges = [np.arange(envbounds[0],envbounds[1]+1)-0.5,
                np.arange(envbounds[2],envbounds[3]+1)-0.5]
    bincenters = [np.arange(envbounds[0],envbounds[1]),
                np.arange(envbounds[2],envbounds[3])]
    
    occupancy,_,_ = np.histogram2d(trajectory['x'].values,trajectory['y'].values,
                            bins=binedges)
    occupancy = occupancy/np.sum(occupancy)
    
    if mask is None:
        uniformoccupancy = np.ones(np.shape(occupancy))
    else:
        uniformoccupancy = 1-mask
    uniformoccupancy = uniformoccupancy/np.sum(uniformoccupancy)
    
    nonuniformity = np.max(np.abs(occupancy-uniformoccupancy))
    #nonuniformity = kl_divergence(occupancy[:],uniformoccupancy[:])
    
    coverage = {
        'occupancy'     :   occupancy,
        'binedges'      :   binedges,
        'bincenters'    :   bincenters,
        'nonuniformity' :   nonuniformity,
        }
    
    if showFig:
        plt.figure
        plt.imshow(coverage['occupancy'])
        plt.text(1, 1, f"{coverage['nonuniformity']:0.1}", fontsize=10,color='r')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
    return coverage


def calculateContinuity(trajectory, showFig=False, numdelays=10, maxdist=15):
    
    delays = np.arange(1,numdelays+1)
    dxbins = np.arange(0,maxdist)
    delay_dist, delay_kl = delaydist(trajectory.values,numdelays=numdelays,
                                     maxdist=maxdist, firstdelay=1)
    #A, K = fit_exp_linear(np.arange(1,numdelays+1), delay_kl, C=0)
    
    #For threshold - shuffle trajectories
    numshuff = 100
    _, shuff_kl = delaydist(np.random.permutation(trajectory.values),numdelays=numshuff,
                                     maxdist=maxdist, firstdelay=1)
    
    threshold = np.mean(shuff_kl) + 3*np.std(shuff_kl) + 1e-3
    underthresh = np.append(delay_kl<threshold,True)
    underthresh = np.argmax(underthresh)
    
    continuity = {
        'delay_dist'     :   delay_dist,
        'delay_kl'      :   delay_kl,
        'delays'        : delays,
        'dxbins'        : dxbins,
        'underthresh'   : underthresh,
        'threshold'     : threshold
        }
        #'kl_decay'      :   -1/K,
        #'kl_A'          : A,
    
    if showFig:
        fig,ax1 = plt.subplots()
        ax2 = ax1.twinx()
        dd= ax1.imshow(continuity['delay_dist'],origin='lower',extent=(0.5,numdelays+0.5,-0.5,maxdist-0.5),aspect='auto')
        ax2.plot(continuity['delays'],continuity['delay_kl'],'o--r')
        ax1.text(numdelays-1,maxdist-2, f"{continuity['underthresh']}", fontsize=10,color='r')
        ax1.set_xlabel('dt')
        ax1.set_ylabel('dx')
        ax2.set_ylabel('K-L Div from P[dx]')
        
    
    return continuity