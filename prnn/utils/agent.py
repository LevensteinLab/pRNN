#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 20:05:03 2021

@author: dl2820
"""

import numpy as np
import warnings

from numpy.random import choice
import random
from ratinabox.utils import get_angle, get_distances_between
from ratinabox.Agent import Agent
from prnn.utils.mouseSpeed import (BOUT_KW, CM_PER_UU, MOUSE_FIT, STEP_SECONDS,
                                  simulate_bout_gated, simulate_riab_1d)

def randActionSequence(tsteps,action_space,action_probability):
    
    action_space = np.arange(action_space.n) #convert gym to np
    action_sequence = choice(action_space, size=(tsteps,), p=action_probability)
    
    return action_sequence

def random_insert(vector, insert_value, probability=0.2):
    # Generate a random mask for insertion points (one less than the vector length)
    insertion_mask = np.random.rand(len(vector) - 1) < probability

    # Create a new list to hold the result
    result = []
    for i in range(len(vector) - 1):
        result.append(vector[i])  # Add the current element
        if insertion_mask[i]:     # Insert 3 with the given probability
            result.append(insert_value)
    result.append(vector[-1])    # Add the last element

    return result
    
class LoopAgent:
    def __init__(self, action_space, p_stop=0.2):
        
        self.action_space = action_space
        self.p_stop = p_stop
        self.name = 'LoopAgent'
    
    def generateActionSequence(self, tsteps, env, p_stop=0.2):

        #Deterministic action sequences
        straight = 2
        left_turn = 0
        right_turn = 1
        stop = 3

        # Agent always goes left
        loop_sequence =  [left_turn] + \
                                [straight]*(env.env.block_width+env.env.corridor_width) + [left_turn] + \
                                [straight]*(env.env.block_height+env.env.corridor_width) + [left_turn] + \
                                [straight]*(env.env.block_width+env.env.corridor_width) + [left_turn] + \
                                [straight]*(env.env.block_height+env.env.corridor_width)

        # Overshoot by repeating the sequence enough times
        action_sequence = loop_sequence * ((tsteps // len(loop_sequence)) + 1)
        action_sequence = random_insert(action_sequence, stop, probability=p_stop)

        # Slice to get exactly tsteps actions
        action_sequence = action_sequence[:tsteps] 

        return action_sequence

    def getObservations(self, env, tsteps, reset=True, includeRender=False, render_highlight=True, **kwargs):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """
        act = self.generateActionSequence(tsteps, env, p_stop=self.p_stop)
        #Alternative to deterministic - probabilistic based on agent position from env
        render = False
        if reset is False:
            raise ValueError('Reset must currently be true for this agent...')
            
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
            
        for aa in range(tsteps):
            obs[aa+1] = env.step(act[aa])[0]
            state['agent_pos'] = np.append(state['agent_pos'],
                                           np.resize(env.get_agent_pos(),(1,2)),axis=0)
            state['agent_dir'] = np.append(state['agent_dir'],
                                           env.get_agent_dir())
            if includeRender:
                render[aa+1] = env.render(mode=None, highlight=render_highlight)

        return obs, act, state, render

class AlternationAgent:
    def __init__(self, action_space, p_stop=0.2, random_trial_start=True):
        
        self.action_space = action_space
        self.p_stop = p_stop
        self.random_trial_start = random_trial_start
    
    def generateActionSequence(self, tsteps, env, p_stop=0.2, random_trial_start=True):


        #Deterministic action sequences
        straight = 2
        left_turn = 0
        right_turn = 1
        stop = 3
        leftfirst = True
        if random_trial_start:
            leftfirst = random.random() < 0.5
        if leftfirst:
            alternation_sequence =  [left_turn] + \
                                    [straight]*(env.env.block_width+env.env.corridor_width) + [left_turn] + \
                                    [straight]*(env.env.block_height+env.env.corridor_width) + [left_turn] + \
                                    [straight]*(env.env.block_width+env.env.corridor_width) + [left_turn] + \
                                    [straight]*(env.env.block_height+env.env.corridor_width) + [right_turn] + \
                                    [straight]*(env.env.block_width+env.env.corridor_width) + [right_turn] + \
                                    [straight]*(env.env.block_height+env.env.corridor_width) + [right_turn] + \
                                    [straight]*(env.env.block_width+env.env.corridor_width) + [right_turn] + \
                                    [straight]*(env.env.block_height+env.env.corridor_width)
        else:
            alternation_sequence =  [right_turn] + \
                                    [straight]*(env.env.block_width+env.env.corridor_width) + [right_turn] + \
                                    [straight]*(env.env.block_height+env.env.corridor_width) + [right_turn] + \
                                    [straight]*(env.env.block_width+env.env.corridor_width) + [right_turn] + \
                                    [straight]*(env.env.block_height+env.env.corridor_width) + [left_turn] + \
                                    [straight]*(env.env.block_width+env.env.corridor_width) + [left_turn] + \
                                    [straight]*(env.env.block_height+env.env.corridor_width) + [left_turn] + \
                                    [straight]*(env.env.block_width+env.env.corridor_width) + [left_turn] + \
                                    [straight]*(env.env.block_height+env.env.corridor_width)

        # Overshoot by repeating the sequence enough times
        action_sequence = alternation_sequence * ((tsteps // len(alternation_sequence)) + 1)
        action_sequence = random_insert(action_sequence, stop, probability=p_stop)

        # Slice to get exactly tsteps actions
        action_sequence = action_sequence[:tsteps] 

        return action_sequence

    def getObservations(self, env, tsteps, reset=True, includeRender=False, render_highlight=True, **kwargs):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """
        act = self.generateActionSequence(tsteps, env, p_stop=self.p_stop, random_trial_start=self.random_trial_start)
        #Alternative to deterministic - probabilistic based on agent position from env
        render = False
        if reset is False:
            raise ValueError('Reset must currently be true for this agent...')
            
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
            
        for aa in range(tsteps):
            obs[aa+1] = env.step(act[aa])[0]
            state['agent_pos'] = np.append(state['agent_pos'],
                                           np.resize(env.get_agent_pos(),(1,2)),axis=0)
            state['agent_dir'] = np.append(state['agent_dir'],
                                           env.get_agent_dir())
            if includeRender:
                render[aa+1] = env.render(mode=None, highlight=render_highlight)

        return obs, act, state, render

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
                                    "speed_std": 0.2,
                                    "thigmotaxis": 0.2,
                                    "wall_repel_distance": 0.2,
                                    }):
        
        super().__init__(riab_env, params)
        self.name = name
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
            pos = env.env.unwrapped.agent.pos
            pos = np.array([pos[0] - env.env.padding, env.env.size - pos[2] + env.env.padding]) / env.env.size
            direction = env.env.unwrapped.agent.dir
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
    

class UnityRandomAgent:
    """Random agent for Unity environments with discrete action spaces.

    Works with any Shell that wraps a Gymnasium-style Unity env
    (discrete actions, visual observations).
    """

    def __init__(self, action_space, default_action_probability=None):
        self.action_space = action_space
        self.default_action_probability = default_action_probability
        if default_action_probability is None:
            self.default_action_probability = (
                np.ones(action_space.n) / action_space.n
            )
        self.name = 'UnityRandomAgent'

    def generateActionSequence(self, tsteps, action_probability=None):
        if action_probability is None:
            action_probability = self.default_action_probability
        return randActionSequence(tsteps, self.action_space, action_probability)

    def getObservations(self, env, tsteps, reset=True, includeRender=False,
                        **kwargs):
        """Collect an observation/action trajectory from a UnityShell.

        Parameters
        ----------
        env : UnityShell
        tsteps : int
        reset : bool

        Returns
        -------
        obs, act, state, render
        """
        act = self.generateActionSequence(tsteps)

        render = False
        obs = [None for _ in range(tsteps + 1)]

        if reset:
            obs[0] = env.reset()
        else:
            # If not resetting, grab current observation via a no-op render
            obs[0] = env.env.render()

        state = {'agent_pos': np.resize(env.get_agent_pos(), (1, 2)),
                 'agent_dir': env.get_agent_dir()}

        if includeRender:
            render = [None for _ in range(tsteps + 1)]
            render[0] = env.render()

        for aa in range(tsteps):
            step_result = env.step(act[aa])
            obs[aa + 1] = step_result[0]
            state['agent_pos'] = np.append(
                state['agent_pos'],
                np.resize(env.get_agent_pos(), (1, 2)), axis=0)
            state['agent_dir'] = np.append(
                state['agent_dir'], env.get_agent_dir())
            if includeRender:
                render[aa + 1] = env.render()

        return obs, act, state, render


class GimblAgentConstant:
    """Constant-speed agent for the Gimbl Unity corridor.

    Sends a fixed forward speed at every timestep. This is the constant-speed
    baseline and it stays -- the `0.15` and `0.7` datasets were generated with
    it. For variable (mouse-calibrated) speed see `GimblAgentVariable` below;
    the two coexist as separate conditions rather than one superseding the other.

    Note that a constant speed makes the action input carry ZERO information --
    it is the same number every step, so the pRNN cannot use it. That is the
    main thing `GimblAgentVariable` changes.

    Args:
        speed: forward speed value (uu/step) sent to Unity at each step
    """

    def __init__(self, speed=0.15):
        self.speed = speed
        self.name = 'GimblAgentConstant'

    def generateActionSequence(self, tsteps):
        return [np.array([self.speed], dtype=np.float32) for _ in range(tsteps)]

    def getObservations(self, env, tsteps, reset=True, includeRender=False, **kwargs):
        act = self.generateActionSequence(tsteps)
        obs = [None] * (tsteps + 1)
        render = False

        obs[0] = env.reset() if reset else env.render()

        state = {'agent_pos': np.resize(env.get_agent_pos(), (1, 3)),
                 'agent_dir': env.get_agent_dir()}

        if includeRender:
            render = [None] * (tsteps + 1)
            render[0] = env.render()

        for t in range(tsteps):
            step_result = env.step(act[t])
            obs[t + 1] = step_result[0]
            terminated, truncated = step_result[2], step_result[3]
            if terminated or truncated:
                obs[t + 1] = env.reset()
            state['agent_pos'] = np.append(
                state['agent_pos'],
                np.resize(env.get_agent_pos(), (1, 3)), axis=0)
            state['agent_dir'] = np.append(state['agent_dir'], env.get_agent_dir())
            if includeRender:
                render[t + 1] = env.render()

        return obs, act, state, render


GIMBL_VARIABLE_CONDITIONS = ('riab_fitted', 'riab_with_pauses')


class GimblAgentVariable:
    """Variable-speed agent for the Gimbl Unity corridor.

    Sends a forward speed that VARIES over time, calibrated against real CA3
    treadmill velocity. Sibling of `GimblAgentConstant` (which stays as the
    constant-speed baseline) -- the two are separate training conditions.

    Why this exists: with a constant speed the action input carries no
    information, and the network has no behavioural variability to contend with.
    Behavioural noise is the low-hanging fruit for the unreliability mystery,
    and it makes a testable prediction -- adding it should push pRNN EVspace
    DOWN toward CA3's.

    Two simulated conditions, both DATA-FREE (parameters are baked into
    `MOUSE_FIT`; nothing is read from disk at run time, because the real-trace
    database is what the future `preloaded` mode will consume):

      'riab_fitted'      a single tight OU. Always running, never stops.
                          The clean "smooth variable speed" control.
      'riab_with_pauses'  that same OU, gated by run/stop bout durations.
                          The mouse-like one, and the default.

    A single OU cannot do both, which is why there are two: `speed_std`
    controls both the width of the running-speed distribution and how often the
    process visits zero, so tightening it to match a mouse's running spread
    necessarily deletes the stops. See prnn/utils/mouseSpeed.py.

    Units. The trace is generated and reasoned about in cm/s; the action Unity
    receives is uu/step, converted at the boundary:

        action[uu/step] = v[cm/s] * dt / cm_per_uu

    with dt=0.1 s and cm_per_uu=2.0 for the RNN corridor, i.e. v/20. Meghan's
    behaviorMate env is 3 cm/uu -> pass cm_per_uu=3.0 for v/30. Landmarks:
    running mode 11.18 cm/s -> 0.56 uu/step; ceiling 20.15 -> 1.01.

    Speed is CLIPPED AT 0, never abs()'d -- the wheel is forward-only, and
    abs() would turn every zero-crossing into a speed peak (it collapses 36
    real run bouts into 1).

    ONE RNG PER AGENT, never reseeded. `generateActionSequence` is called once
    per trajectory by `generate_trajectories`, so a generator seeded *inside*
    that method would hand every trajectory in the dataset the identical speed
    trace -- a dataset that loads, trains, and reports no error while having
    zero across-trajectory behavioural variability, which is the entire point
    of this agent. Keeping the generator on the instance gives fresh traces per
    trajectory plus reproducibility of the whole set from one seed.

    `name` GOES INTO THE DATASET CACHE PATH. `create_dataloader` builds
    `folder/env.name-agent.name-env.act_enc` (data.py), and
    generate_gimbl_trajectories.py and trainNet_res_all.py's --zscore_latents
    branch each rebuild that string independently. So the condition must appear
    in the name or a `riab_fitted` dataset and a `riab_with_pauses` one
    silently share a folder. Prefer constructing via `create_agent` with one of
    the preset keys, which sets `name` to the key and keeps all three
    reconstructions in agreement.

    Args:
        condition: one of GIMBL_VARIABLE_CONDITIONS.
        dt: seconds per pRNN step (0.1). Also the OU integration step.
        cm_per_uu: environment scale. 2.0 = RNN corridor, 3.0 = behaviorMate.
        seed: seeds the instance generator once, at construction.
        max_speed: optional hard cap in cm/s. Off by default. The fitted OU's
            Gaussian tail reaches ~23 cm/s against the mouse's observed 20.15,
            so ~20 is the value that would close that gap.
        min_running_fraction: optional reject -- redraw a trajectory whose
            fraction of frames above the running threshold falls below this.
            Off by default because rejecting biases the distribution, but with
            frac_running=0.543 a 500-step (50 s) trajectory can legitimately be
            almost entirely one long stop, i.e. a trajectory with almost no
            optic flow.
        max_resample: attempts before giving up on min_running_fraction.
        fit: override MOUSE_FIT wholesale (for refits / sensitivity tests).
        name: override the cache-path name. `create_agent` sets this to the
            preset key.
    """

    def __init__(self, condition='riab_with_pauses', dt=STEP_SECONDS,
                 cm_per_uu=CM_PER_UU, seed=0, max_speed=None,
                 min_running_fraction=None, max_resample=20, fit=None,
                 name=None):
        if condition not in GIMBL_VARIABLE_CONDITIONS:
            raise ValueError(
                'unknown condition %r; expected one of %s'
                % (condition, ', '.join(GIMBL_VARIABLE_CONDITIONS)))
        self.condition = condition
        self.dt = dt
        self.cm_per_uu = cm_per_uu
        self.seed = seed
        self.max_speed = max_speed
        self.min_running_fraction = min_running_fraction
        self.max_resample = max_resample
        self.fit = dict(MOUSE_FIT) if fit is None else dict(fit)
        # One generator for the life of the agent, deliberately never reseeded.
        self.rng = np.random.default_rng(seed)
        self.name = name if name else 'GimblAgentVariable_' + condition

    def _drawSpeedTrace(self, tsteps):
        """One raw cm/s trace, clipped at 0 and capped."""
        ou, _ = simulate_riab_1d(tsteps, self.dt,
                                 self.fit['run_speed_mean'],
                                 self.fit['run_speed_std'],
                                 coherence_time=self.fit['run_coherence_time'],
                                 use_package=False, rng=self.rng)
        if self.condition == 'riab_fitted':
            v = ou
        else:
            v = simulate_bout_gated(tsteps, self.dt, np.clip(ou, 0, None),
                                    self.fit['run_dur'], self.fit['stop_dur'],
                                    self.fit['rest_speed_sd'], rng=self.rng,
                                    frac_running=self.fit['frac_running'])
        v = np.clip(v, 0, None)          # forward-only wheel; never abs()
        if self.max_speed is not None:
            v = np.minimum(v, self.max_speed)
        return v

    def speedTrace(self, tsteps):
        """cm/s trace for one trajectory, honouring `min_running_fraction`."""
        for _ in range(self.max_resample + 1):
            v = self._drawSpeedTrace(tsteps)
            if self.min_running_fraction is None:
                return v
            if np.mean(v > BOUT_KW['speed_threshold']) >= self.min_running_fraction:
                return v
        warnings.warn(
            '%s: no trajectory reached min_running_fraction=%.2f in %d attempts; '
            'returning the last draw. Lower the threshold or raise max_resample.'
            % (self.name, self.min_running_fraction, self.max_resample + 1))
        return v

    def generateActionSequence(self, tsteps):
        v = self.speedTrace(tsteps)
        act = v * self.dt / self.cm_per_uu          # cm/s -> uu/step
        return [np.array([a], dtype=np.float32) for a in act]

    def getObservations(self, env, tsteps, reset=True, includeRender=False, **kwargs):
        act = self.generateActionSequence(tsteps)
        obs = [None] * (tsteps + 1)
        render = False

        obs[0] = env.reset() if reset else env.render()

        state = {'agent_pos': np.resize(env.get_agent_pos(), (1, 3)),
                 'agent_dir': env.get_agent_dir()}

        if includeRender:
            render = [None] * (tsteps + 1)
            render[0] = env.render()

        for t in range(tsteps):
            step_result = env.step(act[t])
            obs[t + 1] = step_result[0]
            terminated, truncated = step_result[2], step_result[3]
            if terminated or truncated:
                obs[t + 1] = env.reset()
            state['agent_pos'] = np.append(
                state['agent_pos'],
                np.resize(env.get_agent_pos(), (1, 3)), axis=0)
            state['agent_dir'] = np.append(state['agent_dir'], env.get_agent_dir())
            if includeRender:
                render[t + 1] = env.render()

        return obs, act, state, render


# Named presets, in the spirit of Architectures.py's netOptions: the calibrated
# conditions are fixed, not free parameters. This matters because `agent.name`
# is part of the dataset cache path and is reconstructed independently in three
# places -- a preset key is one string all three derive identically, whereas
# free kwargs would have to be re-passed at generation, z-scoring AND training
# time to reproduce the same folder.
GIMBL_VARIABLE_PRESETS = {
    'GimblAgentVariableSimFitted': dict(condition='riab_fitted'),
    'GimblAgentVariableSimPauses': dict(condition='riab_with_pauses'),
}


def create_agent(envname, env, agentkey, agentname = "", **agent_kwargs):
    if agentkey == 'RandomActionAgent':
        n = env.action_space.n
        base = np.array([0.15, 0.15, 0.6, 0.1])
        action_probability = np.concatenate([base, np.zeros(n - 4)]) if n > 4 else base
        agent = RandomActionAgent(env.action_space, action_probability)

    elif agentkey == 'RatInABoxAgent':
        agent = RatInABoxAgent(name=type(env).__name__)

    elif agentkey == 'MiniworldRandomAgent':
        from prnn.environments.RatEnvironment import make_rat_env
        riab_env = make_rat_env(envname)
        agent = MiniworldRandomAgent(riab_env, name=agentname)

    elif agentkey == 'LoopAgent':
        agent = LoopAgent(env.action_space, p_stop=0.2)

    elif agentkey == 'UnityRandomAgent':
        agent = UnityRandomAgent(env.action_space)

    elif agentkey == 'GimblAgentConstant':
        agent = GimblAgentConstant(**agent_kwargs)

    elif agentkey in GIMBL_VARIABLE_PRESETS:
        # name = the preset key, so the three independent reconstructions of the
        # dataset cache path (generate_gimbl_trajectories.py, the
        # --zscore_latents branch of trainNet_res_all.py, and create_dataloader)
        # cannot disagree.
        preset = dict(GIMBL_VARIABLE_PRESETS[agentkey])
        preset.update(agent_kwargs)
        agent = GimblAgentVariable(name=agentkey, **preset)

    else:
        raise ValueError(
            'unknown agentkey %r. Valid keys: RandomActionAgent, '
            'RatInABoxAgent, MiniworldRandomAgent, LoopAgent, UnityRandomAgent, '
            'GimblAgentConstant, %s'
            % (agentkey, ', '.join(sorted(GIMBL_VARIABLE_PRESETS))))

    return agent