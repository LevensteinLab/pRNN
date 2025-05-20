#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 20:05:03 2021

@author: dl2820
"""

import numpy as np

from numpy.random import choice

def randActionSequence(tsteps,action_space,action_probability):
    
    action_space = np.arange(action_space.n) #convert gym to np
    action_sequence = choice(action_space, size=(tsteps,), p=action_probability)
    
    return action_sequence
    


class RandomActionAgent:
    def __init__(self, action_space, default_action_probability=None):
        
        self.action_space = action_space
        self.default_action_probability = default_action_probability
        if default_action_probability is None:
            self.default_action_probability = np.ones_like(self.action_space)/self.action_space.n
        self.name = 'RandomActionAgent'
        
        
    def generateActionSequence(self, tsteps, action_probability=None):
        if action_probability is None:
            action_probability = self.default_action_probability
        action_sequence = randActionSequence(tsteps,
                                             self.action_space, action_probability)
        return action_sequence
    
    
    def getObservations(self, env, tsteps, reset=True, includeRender=False, **kwargs):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """
        act = self.generateActionSequence(tsteps)

        render = False
            
        obs = [None for t in range(tsteps+1)]
        if reset:
            obs[0] = env.reset()
        else:
            o = env.env.gen_obs()
            obs[0] = env.env.observation(o)
        state = {'agent_pos': np.resize(env.get_agent_pos(),(1,2)), 
                 'agent_dir': env.get_agent_dir()
                }
        if includeRender:
            render = [None for t in range(tsteps+1)]
            render[0] = env.render(mode=None)
            
        for aa in range(tsteps):
            obs[aa+1] = env.step(act[aa])[0]
            state['agent_pos'] = np.append(state['agent_pos'],
                                           np.resize(env.get_agent_pos(),(1,2)),axis=0)
            state['agent_dir'] = np.append(state['agent_dir'],
                                           env.get_agent_dir())
            if includeRender:
                render[aa+1] = env.render(mode=None)

        return obs, act, state, render
    
 
class RandomHDAgent:
    def __init__(self, action_space, default_action_probability=None, constantAction=-1):
        
        self.action_space = action_space
        self.default_action_probability = default_action_probability
        if default_action_probability is None:
            self.default_action_probability = np.ones_like(self.action_space)/self.action_space.n
        self.constantAction = constantAction
        self.name = 'RandomHDAgent'
        
        
    def generateActionSequence(self, tsteps, action_probability=None):
        if action_probability is None:
            action_probability = self.default_action_probability
        action_sequence = randActionSequence(tsteps,
                                             self.action_space, action_probability)
        return action_sequence
    
    
    def getObservations(self, env, tsteps, reset=True, includeRender=False, **kwargs):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """
        act = self.generateActionSequence(tsteps)

        render = False
            
        obs = [None for t in range(tsteps+1)]
        if reset:
            obs[0] = env.reset()
        else:
            o = env.env.gen_obs()
            obs[0] = env.env.observation(o)
        state = {'agent_pos': np.resize(env.get_agent_pos(),(1,2)), 
                 'agent_dir': env.get_agent_dir()
                }
        if includeRender:
            render = [None for t in range(tsteps+1)]
            render[0] = env.render(mode=None)
            
        for aa in range(tsteps):
            obs[aa+1] = env.step(act[aa])[0]
            state['agent_pos'] = np.append(state['agent_pos'],
                                           np.resize(env.get_agent_pos(),(1,2)),axis=0)
            state['agent_dir'] = np.append(state['agent_dir'],
                                           env.get_agent_dir())
            if includeRender:
                render[aa+1] = env.render(mode=None)
                
        act = np.ones_like(act) * self.constantAction

        return obs, act, state, render
    

class RatInABoxAgent:
    def __init__(self, name):
        self.name = name

    def getObservations(self, shell, tsteps, reset=True, includeRender=False,
                        discretize=False, inv_x=False, inv_y=False):

        obs, act, state, render = shell.getObservations(tsteps, reset, includeRender,
                                                        discretize, inv_x, inv_y)

        return obs, act, state, render
    

def create_agent(envname, env, agentname):
    if agentname == 'RandomActionAgent':
        if 'LRoom' in envname:
            action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
        else:
            action_probability = np.array([0.15,0.15,0.6,0.1])
        agent = RandomActionAgent(env.action_space, action_probability)

    elif agentname == 'RatInABoxAgent':
        agent = RatInABoxAgent(name=type(env).__name__)

    elif agentname == 'MiniworldAgent':
        agent = RatInABoxAgent(name="RiaB-to-Miniworld")

    return agent