#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:30:00 2023

@author: Sabrina
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import choice
from ratinabox.utils import get_angle, get_distances_between
from ratinabox.Agent import Agent
import time

import datetime

def randActionSequence(tsteps,action_space,action_probability):
    
    action_space = np.arange(action_space.n) #convert gym to np
    action_sequence = choice(action_space, size=(tsteps,), p=action_probability)
    
    return action_sequence

class RandomBatchedActionSequencesAgent:
    def __init__(self, action_space, default_action_probability=None):
        
        self.action_space = action_space
        self.default_action_probability = default_action_probability
        if default_action_probability is None:
            self.default_action_probability = np.ones_like(self.action_space)/self.action_space.n
        self.name = 'RandomBatchedActionSequencesAgent'
        print("Initialized RandomBatchedActionSequencesAgent")
        print(self.action_space)
        
        
    def generateActionSequence(self, tsteps, action_probability=None):
        """
        Generate an action sequence. If self.action_space is a vector/list of
        per-environment spaces, return an array shaped (tsteps, n_envs) where
        column i is the sequence for environment i. Otherwise return a 1D
        array of length tsteps for a single (non-vectorized) action space.
        """
        if action_probability is None:
            action_probability = self.default_action_probability

        # Detect vectorized action_space: treat as batched if it's iterable
        # and does not expose a single Discrete-like `.n` attribute.
        is_vectorized = (hasattr(self.action_space, '__len__')
                         and not hasattr(self.action_space, 'n'))

        if is_vectorized:
            n_envs = len(self.action_space)
            actions = np.zeros((tsteps, n_envs), dtype=int)

            for i in range(n_envs):
                space_i = self.action_space[i]
                # Support per-env or shared action_probability:
                if np.ndim(action_probability) == 2 and action_probability.shape[0] == n_envs:
                    p = action_probability[i]
                else:
                    p = action_probability
                actions[:, i] = randActionSequence(tsteps, space_i, p)
            return actions
        else:
            # Single (shared) action space
            return randActionSequence(tsteps, self.action_space, action_probability)
    
    
    def getObservations(self, env, shell_env, tsteps, reset=True, includeRender=False, render_highlight=True, **kwargs):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """
        tic = time.time()
        act = self.generateActionSequence(tsteps)
        #print(act.shape)   
        render = False
        print("generateActionSequence*...." +  str(time.time()-tic) + " " + str(tsteps))    
        tic = time.time()
        conspecific = False #This is ugly and shouldn't be here... sorry please don't hate me Alex :')
        if hasattr(shell_env.env, 'conspecific'): # handle only the first env to avoid issues with vectorized envs
            conspecific = True

        # initialize container for observations (tsteps+1). Use object dtype so each entry
        # can hold either a single observation or a batch (for vectorized envs).
        obs = np.empty(tsteps+1, dtype=object)
        obs[:] = None


        if reset:
            # Reset env and store initial observation(s). Support vectorized envs
            reset_result = env.reset()
            # gym sometimes returns (obs, info)
            if isinstance(reset_result, tuple):
                reset_obs = reset_result[0]
            else:
                reset_obs = reset_result
            obs[0] = reset_obs
        state = {'agent_pos': np.resize(shell_env.get_agent_pos(),(1,2)), 
                 'agent_dir': shell_env.get_agent_dir()
                } #handle only first env to avoid issues with vectorized envs
        if includeRender:
            render = np.empty(tsteps+1, dtype=object)
            render[:] = None
            render[0] = env.call("render")
            

        if conspecific:
            state['conspecific_pos'] = np.resize(shell_env.env.conspecific.cur_pos,(1,2))
        
        state['agent_pos'] = np.zeros((tsteps+1, 2), dtype=int)
        state['agent_dir'] = np.zeros(tsteps+1, dtype=int)
        if conspecific:
            state['conspecific_pos'] =  np.zeros((tsteps+1, 2))
            
        for aa in range(tsteps):
            # pick the action(s) for this timestep (will be scalar or vector)
            action = act[aa]
            #print(f"{action=}")

            # step the env; gym returns either obs or a tuple whose first element is obs
            step_result = env.step(action)
            #print("after step...." +  str(time.time()-tic))    
            #tic = time.time()
             
            if isinstance(step_result, (tuple, list)):
                step_obs = step_result[0]
            else:
                step_obs = step_result

            obs[aa+1] = step_obs

            state['agent_pos'][aa] = shell_env.get_agent_pos()
            
            state['agent_dir'][aa] = shell_env.get_agent_dir()
            if conspecific:
                state['conspecific_pos'][aa] = shell_env.env.conspecific.cur_pos

            if includeRender:
                render[aa+1] =  [None]
        print("step done...." +  str(time.time()-tic)) 
        return obs, act, state, render    

def create_batched_agent(envname, envs, agentkey, agentname = ""):
    action_probability = np.array([0.15,0.15,0.6,0.1])
    agent = RandomBatchedActionSequencesAgent(envs.envs.action_space, action_probability)

    return agent