#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 20:05:03 2021

@author: dl2820
"""

import numpy as np

from numpy.random import choice
from ratinabox.utils import get_angle, get_distances_between
from ratinabox.Agent import Agent

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


class MiniworldRandomAgent(Agent):        
    def __init__(self, riab_env, name='', params={
                                    "dt": 0.1,
                                    "speed_mean": 0.2,
                                    "thigmotaxis": 0.2,
                                    "wall_repel_distance": 0.2,
                                    }):
        
        super().__init__(riab_env, params)
        self.reset()

    def update(self, dt=None, drift_velocity=None, drift_to_random_strength_ratio=1):
        super().update(dt, drift_velocity, drift_to_random_strength_ratio)
        self.history["speed"].append(
            np.linalg.norm(np.array(self.history["pos"][-1]) - np.array(self.history["pos"][-2]))
        )

        angle_now = get_angle(np.array(self.history["pos"][-1]) - np.array(self.history["pos"][-2]))
        angle_before = self.history["angle"][-1]
        if abs(angle_now - angle_before) > np.pi:
            if angle_now > angle_before:
                angle_now -= 2 * np.pi
            elif angle_now < angle_before:
                angle_before -= 2 * np.pi
        self.history["rotation"].append(angle_now - angle_before)
        self.history["angle"].append(angle_now)
        return


    def generateActionSequence(self, pos, direction, T=1000):
        self.pos = pos
        self.velocity = self.speed_std * np.array([np.cos(direction), np.sin(direction)])
        self.history["pos"] = [self.pos]
        self.history["vel"] = [self.velocity]
        self.history["speed"] = [np.linalg.norm(self.velocity)]
        self.history["angle"] = [get_angle(self.velocity)]

        for i in range(T):
            self.update()

        traj = np.vstack((np.array(self.history["speed"]) * 10, np.array(self.history["rotation"])))

        return traj[:, -T:]
    
    def getObservations(self, env, tsteps=0, reset=True, includeRender=False,
                        act=None, discretize=False, **kwargs):   
        obs = [None for t in range(tsteps+1)]
        
        if reset:
            obs[0] = env.reset()
            self.reset()
        else:
            obs[0] = env.env.render_obs()
            
        if act is None:
            pos = env.env.agent.pos
            pos = np.array([pos[0] - env.env.padding, env.env.size - pos[2] + env.env.padding]) / env.env.size
            direction = env.env.agent.dir
            act = self.generateActionSequence(pos, direction, tsteps)
        else:
            tsteps = act.shape[1]
            if act.shape[0] != 2:
                raise ValueError("act must be a 2D array with shape (2, tsteps)")
            
        if tsteps <= 0:
            raise ValueError("tsteps must be a positive integer")

        render = False
            
        state = {'agent_pos': np.resize(env.get_agent_pos(),(1,2)),
                 'agent_dir': env.get_agent_dir()
                }
        if includeRender:
            render = [None for t in range(tsteps+1)]
            render[0] = env.env.render_top_view()
            
        for aa in range(tsteps):
            obs[aa+1] = env.step(act[:,aa])[0]
            state['agent_pos'] = np.append(state['agent_pos'],
                                           np.resize(env.get_agent_pos(),(1,2)),axis=0) # probably resize not needed
            state['agent_dir'] = np.append(state['agent_dir'],
                                           env.get_agent_dir())
            if includeRender:
                render[aa+1] = env.env.render_top_view()

        if discretize: # using RiaB coordinates for the positions to be decoded
            state['pos_continuous'] = state['agent_pos'].copy()
            pos = np.array(self.history['pos'])
            # Transform the positions from continuous float coordinates to discrete int coordinates
            dx = self.Environment.dx
            coord = self.Environment.flattened_discrete_coords
            dist = get_distances_between(np.array(pos), coord)
            pos = ((coord[dist.argmin(axis=1)]-dx/2)/dx).astype(int)
            state['agent_pos'] = pos

        return obs, act, state, render
    
    def reset(self):
        self.reset_history()
        self.initialise_position_and_velocity()
        self.history["t"] = [0]
        self.history["pos"] = [self.pos]
        self.history["vel"] = [self.velocity]
        self.history["rot_vel"] = [self.rotational_velocity]
        self.history["speed"] = [np.linalg.norm(self.velocity)]
        self.history["rotation"] = [0]
        self.history["angle"] = [get_angle(self.velocity)]
    

def create_agent(envname, env, agentname):
    if agentname == 'RandomActionAgent':
        if 'LRoom' in envname:
            action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
        else:
            action_probability = np.array([0.15,0.15,0.6,0.1])
        agent = RandomActionAgent(env.action_space, action_probability)

    elif agentname == 'RatInABoxAgent':
        agent = RatInABoxAgent(name=type(env).__name__)

    elif agentname == 'MiniworldRandomAgent':
        from prnn.examples.RatEnvironment import make_rat_env
        riab_env = make_rat_env(envname)
        agent = MiniworldRandomAgent(riab_env, name="RiaB-to-Miniworld")

    return agent