#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 20:05:03 2021

@author: dl2820
"""

from numpy.random import choice
import numpy as np
from ratinabox.utils import get_angle, get_distances_between

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
        
        
    def generateActionSequence(self, tsteps, action_probability=None):
        if action_probability is None:
            action_probability = self.default_action_probability
        action_sequence = randActionSequence(tsteps,
                                             self.action_space, action_probability)
        return action_sequence
    
    
    def getObservations(self, env, tsteps, reset=True, includeRender=False, render_highlight=True, **kwargs):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """
        act = self.generateActionSequence(tsteps)

        render = False
        # if reset is False:
        #     raise ValueError('Reset=False not implemented yet...')

        conspecific = False #This is ugly and shouldn't be here... sorry please don't hate me Alex :')
        if hasattr(env.env, 'conspecific'):
            conspecific = True
            
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
            render[0] = env.render(mode=None, highlight=render_highlight)
        if conspecific:
            state['conspecific_pos'] = np.resize(env.env.conspecific.cur_pos,(1,2))
            
        for aa in range(tsteps):
            obs[aa+1] = env.step(act[aa])[0]
            state['agent_pos'] = np.append(state['agent_pos'],
                                           np.resize(env.get_agent_pos(),(1,2)),axis=0)
            state['agent_dir'] = np.append(state['agent_dir'],
                                           env.get_agent_dir())
            if conspecific:
                state['conspecific_pos'] = np.append(state['conspecific_pos'],
                                           np.resize(env.env.conspecific.cur_pos,(1,2)),axis=0)
            if includeRender:
                render[aa+1] = env.render(mode=None, highlight=render_highlight)

        return obs, act, state, render
    
 
class RandomHDAgent:
    def __init__(self, action_space, default_action_probability=None, constantAction=-1):
        
        self.action_space = action_space
        self.default_action_probability = default_action_probability
        if default_action_probability is None:
            self.default_action_probability = np.ones_like(self.action_space)/self.action_space.n
        self.constantAction = constantAction
        
        
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
        # if reset is False:
        #     raise ValueError('Reset=False not implemented yet...')
            
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
    # Basically all you need in RiaB is this function but for compatibility with other code
    # there is a separate class

    def getObservations(self, shell, tsteps, reset=True, includeRender=False,
                        discretize=False, inv_x=False, inv_y=False):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """

        render = False # Placeholder for compatibility, actual render is in the Shell's 'show_state(_traj)' function
        if reset:
            shell.reset()
        else:
            shell.reset(keep_state=True)

        for aa in range(tsteps):
            shell.ag.update()
            shell.vision[0].update()
            shell.vision[1].update()

        rot_vel = np.array(shell.ag.history['rot_vel'][1:])*shell.ag.dt/np.pi
        vel = np.array(shell.ag.history['vel'][1:])*shell.ag.dt
        act = np.concatenate((rot_vel[:,None], vel), axis=1)

        walls = np.array(shell.vision[0].history["firingrate"])
        objects = np.array(shell.vision[1].history["firingrate"])
        n_neurons = walls.shape[1]
        objects = objects.reshape((tsteps+1, n_neurons, -1), order='F')
        obs = np.concatenate((walls[...,None], objects), axis=-1)

        pos = np.array(shell.ag.history['pos'])
        if discretize:
            # Transform the positions from continuous float coordinates to discrete int coordinates
            dx = shell.env.dx
            coord = shell.env.flattened_discrete_coords
            dist = get_distances_between(np.array(pos), coord)
            pos = ((coord[dist.argmin(axis=1)]-dx/2)/dx).astype(int)
        if inv_x:
            max_x = np.round(pos[:,0].max())
            pos[:,0] = max_x - pos[:,0]
        if inv_y:
            max_y = np.round(pos[:,1].max())
            pos[:,1] = max_y - pos[:,1]

        state = {'agent_pos': pos, 
                 'agent_dir': np.array([get_angle(x) for x in shell.ag.history['vel']]),
                 'mean_vel': shell.ag.speed_mean,
                }

        return obs, act, state, render
    

def create_agent(envname, env, agentname):
    if 'Lava' in envname:
        action_probability = np.array([0.15,0.15,0.6,0.1])
    else:
        action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
    if agentname == 'RandomActionAgent':
        agent = RandomActionAgent(env.action_space, action_probability)
    elif agentname == 'RatInABoxAgent':
        agent = RatInABoxAgent()
    return agent