import torch
import numpy as np
import matplotlib
import random
import math

from matplotlib.collections import EllipseCollection

from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from ratinabox.utils import get_angle, get_distances_between
from ratinabox.contribs.ValueNeuron import ValueNeuron

from prnn.utils import env
from prnn.utils.ActionEncodings import *
from prnn.utils.general import saveFig

actionOptions = {'OneHotHD' : OneHotHD ,
                 'OneHotHDPrevAct' : OneHotHDPrevAct,
                 'SpeedHD' : SpeedHD ,
                 'SpeedNextHD' : SpeedNextHD,
                 'OneHot' : OneHot,
                 'Velocities' : Velocities,
                 'NoAct' : NoAct,
                 'HDOnly': HDOnly,
                 'Continuous': Continuous,
                 'ContSpeedRotation': ContSpeedRotation,
                 'ContSpeedHD': ContSpeedHD,
                 'ContSpeedOnehotHD': ContSpeedOnehotHD
                 }

HDmap = {0: 270,
         1: 180,
         2: 90,
         3: 0
        }

class Shell:
    def __init__(self, env, act_enc, env_key):
        self.env = env
        self.name = env_key
        self.encodeAction = actionOptions[act_enc]
        self.dataLoader = None
        self.DL_iterator = None
        self.n_obs = 1 #default number of observation modalities
        self.repeats = np.array([1], dtype=int) # repeats elevate the signal from an observation modality
        self.multiply = False # if True, repeats are multiplied by the number of repeats, otherwise they are repeated

    def addDataLoader(self, dataloader):
        self.dataLoader = dataloader
        self.DL_iterator = iter(self.dataLoader)

    def killIterator(self):
        iterator = self.DL_iterator
        self.DL_iterator = None
        return iterator

    def collectObservationSequence(self, agent, tsteps, batch_size=1,
                                   obs_format='pred', includeRender=False,
                                   discretize=False, inv_x=False, inv_y=False,
                                   seed=None, dataloader=False, reset=True, save_env=False):
        """
        Use an agent (action generator) to collect an observation/action sequence
        In tensor format for feeding to the predictive net
        Note: batches are implemented only for pre-generated data
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        if dataloader:
            if not self.DL_iterator:
                self.DL_iterator = iter(self.dataLoader)
            try:
                data = next(self.DL_iterator)
            except StopIteration:
                self.DL_iterator = iter(self.dataLoader)
                data = next(self.DL_iterator)
            obs = data[:self.n_obs]
            if (self.repeats != 1).any():
                for i in np.where(self.repeats!=1)[0]:
                    obs[i] = obs[i].repeat(1,1,1,self.repeats[i])
            if len(obs) == 1:
                obs = obs[0]
            act = data[self.n_obs]
            state = None
            render = None
        else:
            for bb in range(batch_size): #TODO: fix for batch_size>1
                obs, act, state, render = agent.getObservations(self,tsteps,
                                                        includeRender=includeRender,
                                                        discretize=discretize,
                                                        inv_x=inv_x,
                                                        inv_y=inv_y,
                                                        reset=reset)
                if save_env:
                    obs_env = obs # save the environment format
                if obs_format == 'pred': # to train right away
                    obs, act = self.env2pred(obs, act)
                    if (self.repeats != 1).any():
                        if not self.multiply:
                            for i in np.where(self.repeats!=1)[0]:
                                obs[i] = obs[i].repeat(1,1,self.repeats[i])
                        else:
                            for i in np.where(self.repeats!=1)[0]:
                                obs[i] = obs[i] * self.repeats[i]
                elif obs_format == 'npgrid': # to save as numpy array
                    obs, act = self.env2np(obs, act)
                elif obs_format is None:
                    continue
        if save_env:
            return obs, act, state, render, obs_env
        else:
            return obs, act, state, render # for backward compatibility

    def dir2deg(self, dir):
        raise NotImplementedError('Environment-specific "Shell" class should be used')
    
    def env2pred(self, **kwargs):
        """
        Convert observation input from environment format to pytorch
        arrays for input to the predictive net, tensor of shape (N,L,H)
        N: Batch size
        L: timestamps
        H: input_size
        https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        """
        raise NotImplementedError('Environment-specific "Shell" class should be used')
    
    def env2np(self, **kwargs):
        """
        Convert observation input from environment format to np.array
        """
        raise NotImplementedError('Environment-specific "Shell" class should be used')

    def pred2np(self, **kwargs):
        """
        Convert sequence of observations from pytorch format to image-filled np.array
        """
        raise NotImplementedError('Environment-specific "Shell" class should be used')
    
    def get_hd(self):
        raise NotImplementedError('Environment-specific "Shell" class should be used')
    
    def getActSize(self):
        """
        Gets the size of the action vector
        """
        raise NotImplementedError('Environment-specific "Shell" class should be used')
    
    def getActType(self):
        """
        Gets the Torch dtype of the action vector
        """
        raise NotImplementedError('Environment-specific "Shell" class should be used')
    
    def getObsSize(self):
        """
        Gets the size of the flattened observation
        """
        raise NotImplementedError('Environment-specific "Shell" class should be used')
    
    def get_viewpoint(self, **kwargs):
        """
        Gets the viewpoint of the agent at a given position and direction
        """
        raise NotImplementedError('Environment-specific "Shell" class should be used')
    
    def get_map_bins(self):
        """
        Gets the binned locations for tuning curves analysis
        """
        raise NotImplementedError('Environment-specific "Shell" class should be used')
    
    def reset(self, **kwargs):
        raise NotImplementedError('Environment-specific "Shell" class should be used')


class GymMinigridShell(Shell):
    def __init__(self, env, act_enc, env_key, **kwargs):
        super().__init__(env, act_enc, env_key)
        self.obs_shape = self.env.observation_space['image'].shape
        self.numHDs = 4
        self.height = self.env.unwrapped.grid.height
        self.width = self.env.unwrapped.grid.width
        self.continuous = False
        self.max_dist = False
        self.loc_mask = [x==None or x.can_overlap() for x in env.grid.grid]
        self.hd_trans = np.array([-1,1,0,0])
        self.start_pos = 1 # the numbering of occupiable locations starts from this
    
    @property
    def action_space(self):
        return self.env.action_space

    def dir2deg(self, dir):
        return HDmap[dir]
    
    def env2pred(self, obs, act=None, hd_from='obs', actoffset=0):
        if hd_from=='obs':
            hd = np.array([self.get_hd(obs[t]) for t in range(len(obs))])
        elif hd_from=='act':
            hd = self.act2hd(obs[actoffset], act, actoffset)
        else:
            KeyError('hd_from should be either "obs" or "act"')
        if act is not None:
            act = self.encodeAction(act=act,
                                    obs=hd,
                                    numSuppObs=self.numHDs,
                                    numActs=self.action_space.n)

        obs = np.array([self.get_visual(obs[t]) for t in range(len(obs))])
        obs = torch.tensor(obs, dtype=torch.float, requires_grad=False)
        obs = torch.unsqueeze(obs, dim=0)
        obs = obs/255 #normalize image

        if hd_from=='obs':
            return obs, act
        else:
            return obs, act, torch.tensor(hd, dtype=torch.int, requires_grad=False)
    
    def env2np(self, obs, act=None):
        hd = np.array([self.get_hd(obs[t]) for t in range(len(obs))])
        if act is not None:
            act = np.array(self.encodeAction(act=act,
                                             obs=hd,
                                             numSuppObs=self.numHDs,
                                             numActs=self.action_space.n))

        obs = np.array([self.get_visual(obs[t]) for t in range(len(obs))])[None]
        obs = obs/255
        return obs, act

    def pred2np(self, obs, whichPhase=0, timesteps=None):
        obs = obs.detach().numpy()
        if timesteps:
            obs = obs[:,timesteps,...]
        obs = np.reshape(obs[whichPhase,:,:],(-1,)+self.obs_shape)
        return obs
    
    def act2hd(self, obs, act, act_offset=0):
        #TODO: adapt for theta with At-1
        hd = [self.get_hd(obs)]
        for a in reversed(act[:act_offset]):
            hd.insert(0, hd[0] - self.hd_trans[a])
        for a in act[act_offset:]:
            hd.append(hd[-1] + self.hd_trans[a] +
                      4 * ((hd[-1]==0) & (self.hd_trans[a]<0)) -
                      4 * ((hd[-1]==3) & (self.hd_trans[a]>0)))
        return np.array(hd)
    
    def get_agent_pos(self):
        return self.env.unwrapped.agent_pos
    
    def get_agent_dir(self):
        return self.env.unwrapped.agent_dir
    
    def get_hd(self, obs):
        return obs['direction']
    
    def get_visual(self, obs):
        return np.reshape(obs['image'],(-1))

    def getActSize(self):
        action = self.encodeAction(act=np.ones(1),
                                   obs=np.ones(2),
                                   numActs=self.action_space.n,
                                   numSuppObs=self.numHDs)
        act_size = action.size(2)
        return act_size
    
    def getActType(self):
        return torch.int64

    def getObsSize(self):
        obs_size = np.prod(self.obs_shape)
        return obs_size
    
    def get_map_bins(self):
        minmax=(0.5, self.width-1.5,
                0.5, self.height-1.5)
        return self.width-2, self.height-2, minmax
    
    def get_HD_bins(self):
        minmax = (-0.5, 4.5)
        return self.numHDs, minmax
    
    def get_viewpoint(self, agent_pos, agent_dir):

        self.env.unwrapped.agent_dir = agent_dir
        self.env.unwrapped.agent_pos = agent_pos
        
        obs = self.env.gen_obs()
        obs = self.env.observation(obs)
        
        return obs['image']/255
    
    def load_state(self, state):
        self.set_agent_pos(state[:2])
        self.set_agent_dir(state[2])
    
    def save_state(self, state):
        return np.append(state['agent_pos'][-1], state['agent_dir'][-1])
    
    def set_agent_pos(self, pos):
        self.env.unwrapped.agent_pos = pos
    
    def set_agent_dir(self, hd):
        self.env.unwrapped.agent_dir = hd
    
    def show_state(self, render, t, **kwargs):
        plt.imshow(render[t])
    
    def show_state_traj(self, start, end, state, render, **kwargs):
        trajectory_ts = np.arange(start, end)
        if render is not None:
            plt.imshow(render[trajectory_ts[-1]])
        plt.plot((state['agent_pos'][trajectory_ts,0]+0.5)*512/16,
                    (state['agent_pos'][trajectory_ts,1]+0.5)*512/16,color='r')
        
    def step(self, action):
        return self.env.step(action)
    
    # TODO: Do we use mode='human' anywhere? Remove?
    def render(self, highlight=True, mode=None):
        return self.env.render(mode=mode, highlight=highlight)
    
    def reset(self, seed=False):
        if seed:
            self.env.seed(seed)
        return self.env.reset()
    

class FaramaMinigridShell(GymMinigridShell):
    def __init__(self, env, act_enc, env_key, **kwargs):
        super().__init__(env, act_enc, env_key)
        
    def render(self, highlight=True, mode='human'):
        return self.env.get_frame()
    
    def reset(self, seed=False):
        if seed:
            return self.env.reset(seed=seed)[0]
        else:
            return self.env.reset()[0]
        
    # For RL
    @property
    def observation_space(self):
        return self.env.observation_space
    

class RatInABoxShell(Shell):
    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins, **kwargs):
        super().__init__(env, act_enc, env_key)
        "For ratinabox==1.7.1, for other versions you may have to update the code."
        self.init_agent(speed, thigmotaxis)
        self.numHDs = HDbins # For One-hot encoding of HD if needed

        self.height = int(round((self.env.extent[3] - self.env.extent[2])/env.dx))
        self.width = int(round((self.env.extent[1] - self.env.extent[0])/env.dx))

        self.true_height = self.env.extent[3] - self.env.extent[2]
        self.true_width = self.env.extent[1] - self.env.extent[0]
        self.max_dist = (self.true_height**2 + self.true_width**2)**0.5

        self.continuous = True
        self.start_pos = 0

    def init_agent(self, speed, thigmotaxis):
        # Create the agent
        ag = Agent(self.env, {
                    'dt': 0.1,
                    'speed_mean': speed,
                    'thigmotaxis': thigmotaxis
                    })
        
        self.ag = ag

    def dir2deg(self, dir):
        return np.rad2deg(dir)+90

    def discretize(self, dx):
        self.env.dx = dx
        self.env.discrete_coords = self.env.discretise_environment(dx=dx)
        self.env.flattened_discrete_coords = self.env.discrete_coords.reshape(
            -1, self.env.discrete_coords.shape[-1]
        )

    def getActSize(self):
        action = self.encodeAction(act=np.ones((2,3)), meanspeed=self.ag.speed_mean, nbins=self.numHDs)
        act_size = action.size(2)
        return act_size
    
    def getActType(self):
        return torch.float32
    
    def get_map_bins(self):
        minmax=(0, self.width,
                0, self.height)
        return self.width, self.height, minmax
    
    def get_HD_bins(self):
        minmax = (0, 2*np.pi)
        return self.numHDs, minmax
    
    def load_state(self, state):
        self.set_agent_pos(state[:2])
        self.set_agent_dir(state[2], state[3])
    
    def save_state(self, state):
        return np.array([*state['agent_pos'][-1],
                         state['agent_dir'][-1],
                         state['mean_vel']])
    
    def set_agent_pos(self, pos):
        self.ag.pos = pos
    
    def set_agent_dir(self, direction, vel):
        self.ag.velocity = vel * np.array([np.cos(direction),
                                           np.sin(direction)])
        self.ag.rotational_velocity = 0


class RiaBVisionShell(RatInABoxShell):
    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins,
                 FoV_params={'spatial_resolution': 0.01,
                             'angle_range': [0, 45],
                             'distance_range': [0.0, 0.33]},
                 **kwargs):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins)

        # Create vision cells
        FoV_Walls = FieldOfViewBVCs(self.ag, params=FoV_params)
        FoV_Objects = [FieldOfViewOVCs(self.ag, params=FoV_params | {
            "object_tuning_type": x
            }) for x in range(env.n_object_types)]
        
        self.vision = [FoV_Walls] + FoV_Objects

        self.obs_colors = ['black']
        facecolor = matplotlib.cm.get_cmap(self.env.object_colormap)
        for i in range(self.env.n_object_types):
            self.obs_colors.append(facecolor(i / (self.env.n_object_types - 1 + 1e-8)))

        self.reset()

    def getObservations(self, tsteps, reset=True, includeRender=False,
                        discretize=False, inv_x=False, inv_y=False):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """

        render = False # Placeholder for compatibility, actual render is in the 'show_state(_traj)' function
        if reset:
            self.reset()
        else:
            self.reset(keep_state=True)

        for aa in range(tsteps):
            self.ag.update()
            for i in range(len(self.vision)):
                self.vision[i].update()

        rot_vel = np.array(self.ag.history['rot_vel'][1:])*self.ag.dt/np.pi
        vel = np.array(self.ag.history['vel'][1:])*self.ag.dt
        act = np.concatenate((rot_vel[:,None], vel), axis=1)
        obs = np.concatenate([np.array(self.vision[i].history["firingrate"])[...,None]\
                              for i in range(len(self.vision))], axis=-1)

        pos = np.array(self.ag.history['pos'])
        if discretize:
            # Transform the positions from continuous float coordinates to discrete int coordinates
            dx = self.env.dx
            coord = self.env.flattened_discrete_coords
            dist = get_distances_between(np.array(pos), coord)
            pos = ((coord[dist.argmin(axis=1)]-dx/2)/dx).astype(int)
        if inv_x:
            max_x = np.round(pos[:,0].max())
            pos[:,0] = max_x - pos[:,0]
        if inv_y:
            max_y = np.round(pos[:,1].max())
            pos[:,1] = max_y - pos[:,1]

        state = {'agent_pos': pos, 
                 'agent_dir': np.array([get_angle(x) for x in self.ag.history['vel']]),
                 'mean_vel': self.ag.speed_mean,
                }

        return obs, act, state, render
    
    def env2pred(self, obs, act=None):
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean

        obs = obs.clip(max=1)
        obs = obs.reshape(obs.shape[:-2]+(-1,))
        obs = torch.tensor(obs, dtype=torch.float, requires_grad=False)
        obs = torch.unsqueeze(obs, dim=0)

        return obs, act
    
    def env2np(self, obs, act=None):
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean
        act = np.array(act)

        obs = obs.clip(max=1)[None]
        obs = obs.reshape(obs.shape[:-2]+(-1,))

        return obs, act
    
    def pred2np(self, obs, whichPhase=0, timesteps=None):        
        obs = obs.detach().numpy()#.squeeze()
        if timesteps:
            obs = obs[:,timesteps,...]

        img = []
        for t in range(obs.shape[1]):
            img.append(self.to_image(obs[whichPhase,t])[None,...])
        obs = np.concatenate(img, axis=0)
        return obs
    
    def to_image(self, obs):
        fig, ax = plt.subplots()
        # ax.set_ylim(top=0.5)

        n = self.env.n_object_types
        obs = obs.reshape(-1,n+1)

        y = self.vision[0].tuning_distances * np.cos(self.vision[0].tuning_angles)
        x = self.vision[0].tuning_distances * np.sin(self.vision[0].tuning_angles) + 0.5
        ww = (self.vision[0].sigma_angles * self.vision[0].tuning_distances)
        hh = self.vision[0].sigma_distances
        aa  = self.vision[0].tuning_angles * 180 / np.pi

        for obj in range(n+1):
            facecolor_array = self.vision[obj].cell_colors.copy()
            facecolor_array[:, -1] = 0.7*np.maximum( # Make the colors increasingly transparent as the next overlays the previous
                                    0, np.minimum(1, obs[:,obj] / (self.vision[obj].max_fr*n/(n+obj)))
                                                    )

            ec = EllipseCollection(ww,hh, aa, units = 'x',
                                    offsets = np.array([x,y]).T,
                                    offset_transform = ax.transData,
                                    linewidth=0.5,
                                    edgecolor="dimgrey",
                                    zorder = 2.1,
                                    )
            ec.set_facecolors(facecolor_array)

            ax.add_collection(ec) 

        plt.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0.15)
        plt.gca().invert_xaxis()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image_from_plot

    def getObsSize(self):
        obs_size = sum([self.vision[i].n for i in range(len(self.vision))])
        return obs_size
    
    def get_viewpoint(self, agent_pos, agent_dir):
        self.reset(pos=agent_pos, hd=agent_dir)
        obs = np.concatenate([np.array(self.vision[i].history["firingrate"])[...,None]\
                              for i in range(len(self.vision))], axis=-1)

        return self.env2pred(obs)[0].squeeze()
    
    def show_state(self, t, fig, ax, **kwargs):
        start_t = self.ag.history["t"][t-1]
        end_t = self.ag.history["t"][t]
        # create a little space around the env
        self.env.extent[0]-=0.05
        self.env.extent[1]+=0.05
        self.env.extent[2]-=0.05
        self.env.extent[3]+=0.05
        self.ag.plot_trajectory(t_start=start_t, t_end=end_t, fig=fig, ax=ax,color="changing")
        for i in range(len(self.vision)):
            self.vision[i].display_vector_cells(fig, ax, t=end_t)
        # reset the env extent
        self.env.extent[0]+=0.05
        self.env.extent[1]-=0.05
        self.env.extent[2]+=0.05
        self.env.extent[3]-=0.05
    
    def show_state_traj(self, start, end, fig, ax, **kwargs):
        start_t = self.ag.history["t"][start]
        end_t = self.ag.history["t"][end]
        # create a little space around the env
        self.env.extent[0]-=0.05
        self.env.extent[1]+=0.05
        self.env.extent[2]-=0.05
        self.env.extent[3]+=0.05
        self.ag.plot_trajectory(t_start=start_t, t_end=end_t, fig=fig, ax=ax, color="changing")
        for i in range(len(self.vision)):
            self.vision[i].display_vector_cells(fig, ax, t=end_t)
        # reset the env extent
        self.env.extent[0]+=0.05
        self.env.extent[1]-=0.05
        self.env.extent[2]+=0.05
        self.env.extent[3]-=0.05
    
    def reset(self, pos=np.zeros(2), vel=None, seed=False, keep_state=False):

        self.ag.reset_history()
        self.ag.t = 0
        for i in range(len(self.vision)):
            self.vision[i].reset_history()
        
        if keep_state:
            vel = self.ag.velocity
            pos = self.ag.pos

        if vel:
            self.ag.pos = pos
        else:
            vel = [0,0]

        self.ag.save_velocity = vel
        self.ag.save_to_history()
        for i in range(len(self.vision)):
            self.vision[i].update()





class RiaBVisionShell2(RiaBVisionShell):
    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins,
                 wellSigmaDistance = 0.02, wellSigmaAngleDenominator = 2,
                 repeats = np.array([1,1]),
                 FoV_params={ #need these parameters to be plugged in
                           "spatial_resolution": 0.05,
                           "angle_range": [0, 30],
                           "distance_range": [0.0, 1.2],
                           "beta": 10,
                           "walls_occlude": False},
                 **kwargs):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params)
        self.repeats = repeats

        # Create vision cells
        self.wellSigmaDistance = wellSigmaDistance
        self.wellSigmaAngleDenominator = wellSigmaAngleDenominator

        FoV_Walls = FieldOfViewBVCs(self.ag, params=FoV_params)
        FoV_Walls.sigma_distances = np.full((60,), 0.02)
        FoV_Walls.sigma_angles /= 2
        FoV_Objects = [FieldOfViewOVCs(self.ag, params=FoV_params | {
            "object_tuning_type": x
            }) for x in range(env.n_object_types)]
        FoV_Objects[0].sigma_distances = np.full((60,), wellSigmaDistance)
        FoV_Objects[0].sigma_angles /= wellSigmaAngleDenominator
        

        self.vision = [FoV_Walls] + FoV_Objects

        self.reset()
    
    def show_state_traj(self, start, end, fig, ax, **kwargs):
        start_t = self.ag.history["t"][start]
        end_t = self.ag.history["t"][end]
        # create a little space around the env
        self.env.extent[0]-=0.5
        self.env.extent[1]+=0.5
        self.env.extent[2]-=0.5
        self.env.extent[3]+=0.5
        self.ag.plot_trajectory(t_start=start_t, t_end=end_t, fig=fig, ax=ax, color="changing")
        for i in range(len(self.vision)):
            self.vision[i].display_vector_cells(fig, ax, t=end_t)
        # reset the env extent
        self.env.extent[0]+=0.5
        self.env.extent[1]-=0.5
        self.env.extent[2]+=0.5
        self.env.extent[3]-=0.5

    def vision_to_rgb(self, obs_vis, scale=50):
        # Combine separate objects into separate RGB channels
        remix = np.zeros((*obs_vis.shape[:-1], 3))
        remix += np.tile(obs_vis[..., 0, None], 3) * scale / 255
        for i in range(1, obs_vis.shape[-1]):
            remix += np.moveaxis(
                np.tile(obs_vis[..., i], [3] + [1] * (len(obs_vis[..., i].shape))),
                0,
                -1
            ) * self.obs_colors[i][:3]
        remix = remix.clip(max=1)
        remix = remix.reshape(remix.shape[:-2] + (-1,))
        return remix

    # def init_agent(self, speed, thigmotaxis):
    #     # Create the agent
    #     ag = Agent(self.env, {
    #                 'dt': 50e-3,
    #                 'speed_mean': speed,
    #                 'thigmotaxis': thigmotaxis
    #                 })
        
    #     self.ag = ag


class RiaBRemixColorsShell(RiaBVisionShell):
    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params,
                 **kwargs):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params)

    def env2pred(self, obs, act=None):
        """
        Convert observation and action input to pytorch arrays
        for input to the predictive net, tensor of shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
        https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        """
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean

        remix = np.zeros((*obs.shape[:-1],3))
        remix += np.tile(obs[...,0,None],3)*100/255
        if 'LRoom' in self.name:
            remix[...,2] += obs[...,1]
            remix[...,0] += obs[...,2]
            remix[...,0] += obs[...,3]
            remix[...,1] += obs[...,3]
        else:
            for i in range(1,obs.shape[-1]):
                remix += np.moveaxis(np.tile(obs[...,i], [2]+[1]*(len(obs[...,i].shape))),
                                     0,
                                     -1
                                     ) * self.obs_colors[i][:3]
        obs = remix


        obs = obs.clip(max=1)
        obs = obs.reshape(obs.shape[:-2]+(-1,))
        obs = torch.tensor(obs, dtype=torch.float, requires_grad=False)
        obs = torch.unsqueeze(obs, dim=0)

        return obs, act

    def env2np(self, obs, act=None):
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean
        act = np.array(act)

        remix = np.zeros((*obs.shape[:-1],3))
        remix += np.tile(obs[...,0,None],3)*100/255
        if 'LRoom' in self.name:
            remix[...,2] += obs[...,1]
            remix[...,0] += obs[...,2]
            remix[...,0] += obs[...,3]
            remix[...,1] += obs[...,3]
        else:
            for i in range(1,obs.shape[-1]):
                remix += np.moveaxis(np.tile(obs[...,i], [2]+[1]*(len(obs[...,i].shape))),
                                     0,
                                     -1
                                     ) * self.obs_colors[i][:3]
        obs = remix


        obs = obs.clip(max=1)
        obs = obs.reshape(obs.shape[:-2]+(-1,))[None]

        return obs, act
    
    def pred2np(self, obs, whichPhase=0, timesteps=None):
        """
        Convert sequence of observations from pytorch format to image-filled np.array
        """
        
        obs = obs.detach().numpy()
        if timesteps:
            obs = obs[:,timesteps,...]

        img = []
        for t in range(obs.shape[1]):
            img.append(self.to_image(obs[whichPhase,t])[None,...])
        obs = np.concatenate(img, axis=0)
        return obs
    
    def to_image(self, obs):
        fig, ax = plt.subplots()


        obs = obs.reshape(-1,3)

        y = self.vision[0].tuning_distances * np.cos(self.vision[0].tuning_angles)
        x = self.vision[0].tuning_distances * np.sin(self.vision[0].tuning_angles) + 0.5
        ww = (self.vision[0].sigma_angles * self.vision[0].tuning_distances)
        hh = self.vision[0].sigma_distances
        aa  = self.vision[0].tuning_angles * 180 / np.pi
        ec = EllipseCollection(ww,hh, aa, units = 'x',
                                offsets = np.array([x,y]).T,
                                offset_transform = ax.transData,
                                linewidth=0.5,
                                edgecolor="dimgrey",
                                zorder = 2.1,
                                )
        ec.set_facecolors(obs)

        ax.add_collection(ec) 

        plt.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0.15)
        plt.gca().invert_xaxis()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image_from_plot

    def getObsSize(self):
        obs_size = self.vision[0].n * 3
        return obs_size


class RiaBGridShell(RatInABoxShell):
    def __init__(self, env, act_enc, env_key, speed,
                 thigmotaxis, HDbins, Grid_params, seed, **kwargs):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins)

        # Create grid cells
        np.random.seed(seed) # Otherwise there will be a discrepancy with the data from dataloader
        self.grid = GridCells(self.ag, params=Grid_params)

        self.reset()

    def getObservations(self, tsteps, reset=True, includeRender=False,
                        discretize=False, inv_x=False, inv_y=False):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """

        render = False # Placeholder for compatibility, actual render is in the 'show_state(_traj)' function
        if reset:
            self.reset()
        else:
            self.reset(keep_state=True)

        for aa in range(tsteps):
            self.ag.update()
            self.grid.update()

        rot_vel = np.array(self.ag.history['rot_vel'][1:])*self.ag.dt/np.pi
        vel = np.array(self.ag.history['vel'][1:])*self.ag.dt
        act = np.concatenate((rot_vel[:,None], vel), axis=1)

        obs = np.array(self.grid.history["firingrate"])

        pos = np.array(self.ag.history['pos'])
        if discretize:
            # Transform the positions from continuous float coordinates to discrete int coordinates
            dx = self.env.dx
            coord = self.env.flattened_discrete_coords
            dist = get_distances_between(np.array(pos), coord)
            pos = ((coord[dist.argmin(axis=1)]-dx/2)/dx).astype(int)
        if inv_x:
            max_x = np.round(pos[:,0].max())
            pos[:,0] = max_x - pos[:,0]
        if inv_y:
            max_y = np.round(pos[:,1].max())
            pos[:,1] = max_y - pos[:,1]

        state = {'agent_pos': pos, 
                 'agent_dir': np.array([get_angle(x) for x in self.ag.history['vel']]),
                 'mean_vel': self.ag.speed_mean,
                }

        return obs, act, state, render
    
    def env2pred(self, obs, act=None):
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean
        
        obs = obs.clip(max=1)
        obs = torch.tensor(obs, dtype=torch.float, requires_grad=False)
        obs = torch.unsqueeze(obs, dim=0)

        return obs, act
    
    def env2np(self, obs, act=None):
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean
        act = np.array(act)

        obs = obs.clip(max=1)[None]

        return obs, act
    
    def pred2np(self, obs, whichPhase=0, timesteps=None):        
        # No images, placeholder for compatibility
        if timesteps:
            obs = obs[:,timesteps,...]
        return np.ones([obs.shape[1],1,1,3])

    def getObsSize(self):
        obs_size = self.grid.n
        return obs_size
    
    def get_viewpoint(self, agent_pos, agent_dir):
        self.reset(pos=agent_pos, hd=agent_dir)

        obs = np.array(self.grid.history["firingrate"])

        return self.env2pred(obs)[0].squeeze()
    
    def show_state(self, t, fig, ax, **kwargs):
        start_t = self.ag.history["t"][t-1]
        end_t = self.ag.history["t"][t]
        self.ag.plot_trajectory(t_start=start_t, t_end=end_t, fig=fig, ax=ax,color="changing")
    
    def show_state_traj(self, start, end, fig, ax, **kwargs):
        start_t = self.ag.history["t"][start]
        end_t = self.ag.history["t"][end]
        self.ag.plot_trajectory(t_start=start_t, t_end=end_t, fig=fig, ax=ax, color="changing")
    
    def reset(self, pos=np.zeros(2), vel=None, seed=False, keep_state=False):

        self.ag.reset_history()
        self.ag.t = 0
        self.grid.reset_history()
        
        if keep_state:
            vel = self.ag.velocity
            pos = self.ag.pos

        if vel:
            self.ag.pos = pos
        else:
            vel = [0,0]

        self.ag.save_velocity = vel
        self.ag.save_to_history()
        self.grid.update()


class RiaBColorsGridShell(RiaBVisionShell):
    def __init__(self, env, act_enc, env_key, speed,
                 thigmotaxis, HDbins, FoV_params, Grid_params, seed, **kwargs):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params)
        self.n_obs = 2
        # Create grid cells
        np.random.seed(seed) # Otherwise there will be a discrepancy with the data from dataloader
        self.grid = GridCells(self.ag, params=Grid_params)
        
        self.reset()

    def getObservations(self, tsteps, reset=True, includeRender=False,
                        discretize=False, inv_x=False, inv_y=False):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """

        render = False # Placeholder for compatibility, actual render is in the 'show_state(_traj)' function
        if reset:
            self.reset()
        else:
            self.reset(keep_state=True)

        for aa in range(tsteps):
            self.ag.update()
            self.grid.update()
            for i in range(len(self.vision)):
                self.vision[i].update()

        rot_vel = np.array(self.ag.history['rot_vel'][1:])*self.ag.dt/np.pi
        vel = np.array(self.ag.history['vel'][1:])*self.ag.dt
        act = np.concatenate((rot_vel[:,None], vel), axis=1)
        obs_vis = np.concatenate([np.array(self.vision[i].history["firingrate"])[...,None]\
                              for i in range(len(self.vision))], axis=-1)
        obs_grid = np.array(self.grid.history["firingrate"])
        obs = (obs_vis, obs_grid)

        pos = np.array(self.ag.history['pos'])
        if discretize:
            # Transform the positions from continuous float coordinates to discrete int coordinates
            dx = self.env.dx
            coord = self.env.flattened_discrete_coords
            dist = get_distances_between(np.array(pos), coord)
            pos = ((coord[dist.argmin(axis=1)]-dx/2)/dx).astype(int)
        if inv_x:
            max_x = np.round(pos[:,0].max())
            pos[:,0] = max_x - pos[:,0]
        if inv_y:
            max_y = np.round(pos[:,1].max())
            pos[:,1] = max_y - pos[:,1]

        state = {'agent_pos': pos, 
                 'agent_dir': np.array([get_angle(x) for x in self.ag.history['vel']]),
                 'mean_vel': self.ag.speed_mean,
                }

        return obs, act, state, render

    def env2pred(self, obs, act=None):
        """
        Convert observation and action input to pytorch arrays
        for input to the predictive net, tensor of shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
        https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        """
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean

        obs_vis, obs_grid = obs

        remix = np.zeros((*obs_vis.shape[:-1],3))
        remix += np.tile(obs_vis[...,0,None],3)*100/255
        if 'LRoom' in self.name:
            remix[...,2] += obs_vis[...,1]
            remix[...,0] += obs_vis[...,2]
            remix[...,0] += obs_vis[...,3]
            remix[...,1] += obs_vis[...,3]
        else:
            for i in range(1,obs_vis.shape[-1]):
                remix += np.moveaxis(np.tile(obs_vis[...,i], [2]+[1]*(len(obs_vis[...,i].shape))),
                                     0,
                                     -1
                                     ) * self.obs_colors[i][:3]
        remix = remix.clip(max=1)
        remix = remix.reshape(remix.shape[:-2]+(-1,))
        remix = torch.tensor(remix, dtype=torch.float, requires_grad=False)
        remix = torch.unsqueeze(remix, dim=0)


        obs_grid = obs_grid.clip(max=1)
        obs_grid = torch.tensor(obs_grid, dtype=torch.float, requires_grad=False)
        obs_grid = torch.unsqueeze(obs_grid, dim=0)

        obs = (remix, obs_grid)

        return obs, act

    def env2np(self, obs, act=None):
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean
        act = np.array(act)

        obs_vis, obs_grid = obs

        remix = np.zeros((*obs_vis.shape[:-1],3))
        remix += np.tile(obs_vis[...,0,None],3)*100/255
        if 'LRoom' in self.name:
            remix[...,2] += obs_vis[...,1]
            remix[...,0] += obs_vis[...,2]
            remix[...,0] += obs_vis[...,3]
            remix[...,1] += obs_vis[...,3]
        else:
            for i in range(1,obs_vis.shape[-1]):
                remix += np.moveaxis(np.tile(obs_vis[...,i], [2]+[1]*(len(obs_vis[...,i].shape))),
                                     0,
                                     -1
                                     ) * self.obs_colors[i][:3]
        remix = remix.clip(max=1)
        remix = remix.reshape(remix.shape[:-2]+(-1,))[None]


        obs_grid = obs_grid.clip(max=1)[None]

        obs = (remix, obs_grid)

        return obs, act
    
    def pred2np(self, obs, whichPhase=0, timesteps=None):
        """
        Convert sequence of observations from pytorch format to image-filled np.array
        """
        if not obs[0]==None:
            obs = obs[0].detach().numpy()
            if timesteps:
                obs = obs[:,timesteps,...]
            img = []
            for t in range(obs.shape[1]):
                img.append(self.to_image(obs[whichPhase,t])[None,...])
            obs = np.concatenate(img, axis=0)
        else:
            for i in range(self.n_obs):
                if obs[i] is not None:
                    if timesteps:
                        obs[i] = obs[i][:,timesteps,...]
                    o = np.ones([obs[i].shape[1],1,1,3])
                    break
            obs = o
        return obs
    
    def to_image(self, obs):
        fig, ax = plt.subplots()


        obs = obs.reshape(-1,3)

        y = self.vision[0].tuning_distances * np.cos(self.vision[0].tuning_angles)
        x = self.vision[0].tuning_distances * np.sin(self.vision[0].tuning_angles) + 0.5
        ww = (self.vision[0].sigma_angles * self.vision[0].tuning_distances)
        hh = self.vision[0].sigma_distances
        aa  = self.vision[0].tuning_angles * 180 / np.pi
        ec = EllipseCollection(ww,hh, aa, units = 'x',
                                offsets = np.array([x,y]).T,
                                offset_transform = ax.transData,
                                linewidth=0.5,
                                edgecolor="dimgrey",
                                zorder = 2.1,
                                )
        ec.set_facecolors(obs)

        ax.add_collection(ec) 

        plt.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0.15)
        plt.gca().invert_xaxis()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image_from_plot

    def getObsSize(self):
        obs_size = (self.vision[0].n * 3, self.grid.n)
        return obs_size
    
    def reset(self, pos=np.zeros(2), vel=None, seed=False, keep_state=False):
        if not hasattr(self, 'grid'):
            return
        self.ag.reset_history()
        self.ag.t = 0
        self.grid.reset_history()
        for i in range(len(self.vision)):
            self.vision[i].reset_history()
        
        if keep_state:
            vel = self.ag.velocity
            pos = self.ag.pos

        if vel is not None:
            self.ag.pos = pos
        else:
            vel = [0,0]

        self.ag.save_velocity = vel
        self.ag.save_to_history()
        self.grid.update()
        for i in range(len(self.vision)):
            self.vision[i].update()


class RiaBColorsRewardShell(RiaBVisionShell2):


    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params,
                 SigmaD, SigmaA, seed,
                 repeats = np.array([1,1]), multiply=False, **kwargs):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins,
                         SigmaD, SigmaA,
                         repeats, FoV_params)
        self.n_obs = 2
        self.multiply = multiply

        coords = env.objects["objects"]         # shape (N, 2)
        types  = env.objects["object_types"]    # shape (N,)

        # Create a mask (Boolean array) for where types == 0
        mask = (types == 0)
        coords_type_0 = coords[mask]
        if len(coords_type_0) < 3:
            raise ValueError("Not enough holes in the environment to set the specified number of rewards.")

        np.random.seed(seed+101)
        reward_hole_indices = np.random.choice(len(coords_type_0), 3, replace=False)
        distances = np.linalg.norm(coords_type_0[reward_hole_indices] - self.env.home_pos, axis=1)
        reward_hole_indices = reward_hole_indices[np.argsort(distances)]
        closest_reward = coords_type_0[reward_hole_indices][0]
        #second reward is the one closest to the first reward
        distances = np.linalg.norm(coords_type_0[reward_hole_indices][1:] - closest_reward, axis=1)
        second_reward_index = np.argsort(distances)[0] + 1 #+1
        reward_hole_indices = np.concatenate([[reward_hole_indices[0]], [reward_hole_indices[second_reward_index]], reward_hole_indices[2:]], axis=0)
        reward_positions = coords_type_0[reward_hole_indices]

        self.create_rewards(reward_positions)
        
        
        self.reset()

    def create_rewards(self, reward_positions):
        self.reward_positions = reward_positions
        #Create reward neurons (another place cell hidden behind the barrier) 
        self.Reward = PlaceCells(
        self.ag,
        params={
            "n": 3,
            "place_cell_centres": np.array(reward_positions),
            # "description": "top_hat",
            "widths": 0.025,
            "max_fr": 1,
            "color": "C5",
            "wall_geometry": "euclidean",
        },
        )

    def getObservations(self, tsteps, reset=True, includeRender=False,
                        discretize=False, inv_x=False, inv_y=False):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """

        render = False # Placeholder for compatibility, actual render is in the 'show_state(_traj)' function
        if reset:
            self.reset()
        else:
            self.reset(keep_state=True)

        for aa in range(tsteps):
            self.ag.update()
            self.Reward.update() #switched from self.grid.update()
            for i in range(len(self.vision)):
                self.vision[i].update()

        rot_vel = np.array(self.ag.history['rot_vel'][1:])*self.ag.dt/np.pi
        vel = np.array(self.ag.history['vel'][1:])*self.ag.dt
        act = np.concatenate((rot_vel[:,None], vel), axis=1)
        obs_vis = np.concatenate([np.array(self.vision[i].history["firingrate"])[...,None]\
                              for i in range(len(self.vision))], axis=-1)
        obs_reward = np.asarray(self.Reward.history["firingrate"])      # (T,)
        obs = (obs_vis, obs_reward)

        pos = np.array(self.ag.history['pos'])
        if discretize:
            # Transform the positions from continuous float coordinates to discrete int coordinates
            dx = self.env.dx
            coord = self.env.flattened_discrete_coords
            dist = get_distances_between(np.array(pos), coord)
            pos = ((coord[dist.argmin(axis=1)]-dx/2)/dx).astype(int)
        if inv_x:
            max_x = np.round(pos[:,0].max())
            pos[:,0] = max_x - pos[:,0]
        if inv_y:
            max_y = np.round(pos[:,1].max())
            pos[:,1] = max_y - pos[:,1]

        state = {'agent_pos': pos, 
                 'agent_dir': np.array([get_angle(x) for x in self.ag.history['vel']]),
                 'mean_vel': self.ag.speed_mean,
                }

        return obs, act, state, render

    def env2pred(self, obs, act=None):
        """
        Convert observation and action input to pytorch arrays
        for input to the predictive net, tensor of shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
        https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        """
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean

        obs_vis, obs_Reward = obs
        

        remix = self.vision_to_rgb(obs_vis)
        remix = torch.tensor(remix, dtype=torch.float, requires_grad=False)
        remix = torch.unsqueeze(remix, dim=0)


        obs_Reward = obs_Reward.clip(max=1)
        obs_Reward = torch.tensor(obs_Reward, dtype=torch.float, requires_grad=False)
        obs_Reward = torch.unsqueeze(obs_Reward, dim=0)

        obs = [remix, obs_Reward]

        return obs, act

    def env2np(self, obs, act=None):
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean
        act = np.array(act)

        obs_vis, obs_Reward = obs

        remix = np.zeros((*obs_vis.shape[:-1],3))
        remix += np.tile(obs_vis[...,0,None],3)*100/255
        for i in range(1,obs_vis.shape[-1]):
            remix += np.moveaxis(np.tile(obs_vis[...,i], [3]+[1]*(len(obs_vis[...,i].shape))),
                                    0,
                                    -1
                                    ) * self.obs_colors[i][:3]
        remix = remix.clip(max=1)
        remix = remix.reshape(remix.shape[:-2]+(-1,))[None]


        obs_Reward = obs_Reward.clip(max=1)[None]

        obs = (remix, obs_Reward)

        return obs, act
    
    def pred2np(self, obs, whichPhase=0, timesteps=None):
        """
        Convert sequence of observations from pytorch format to image-filled np.array
        """
        obs = obs[0].detach().numpy()
        if timesteps:
            obs = obs[:,timesteps,...]

        img = []
        for t in range(obs.shape[1]):
            img.append(self.to_image(obs[whichPhase,t])[None,...])
        obs = np.concatenate(img, axis=0)
        return obs
    
    def to_image(self, obs):
        fig, ax = plt.subplots()


        obs = obs.reshape(-1,3)

        y = self.vision[0].tuning_distances * np.cos(self.vision[0].tuning_angles)
        x = self.vision[0].tuning_distances * np.sin(self.vision[0].tuning_angles) + 0.5
        ww = (self.vision[0].sigma_angles * self.vision[0].tuning_distances)
        hh = self.vision[0].sigma_distances
        aa  = self.vision[0].tuning_angles * 180 / np.pi
        ec = EllipseCollection(ww,hh, aa, units = 'x',
                                offsets = np.array([x,y]).T,
                                offset_transform = ax.transData,
                                linewidth=0.5,
                                edgecolor="dimgrey",
                                zorder = 2.1,
                                )
        ec.set_facecolors(obs)

        ax.add_collection(ec) 

        plt.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0.15)
        plt.gca().invert_xaxis()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image_from_plot

    def getObsSize(self):
        obs_size = np.array((self.vision[0].n * 3, self.Reward.n * self.repeats[1])) * self.repeats**(not self.multiply)
        return obs_size
    
    def reset(self, pos=None, vel=[0,0], seed=False, keep_state=False):
        if not hasattr(self, 'Reward'):
            return
        self.ag.reset_history()
        self.ag.t = 0
        self.Reward.reset_history()
        for i in range(len(self.vision)):
            self.vision[i].reset_history()
        
        if keep_state:
            vel = self.ag.velocity
            pos = self.ag.pos

        if pos is None and hasattr(self.env, "home_pos"):
            pos = self.env.home_pos.copy()

        if pos is not None:
            self.ag.pos = pos

        self.ag.save_velocity = vel
        self.ag.save_to_history()
        self.Reward.update()
        for i in range(len(self.vision)):
            self.vision[i].update()


class RiaBColorsGridRewardShell(RiaBVisionShell2): #switching to 2 to test dif sigma distances and angles (Hadrien)


    def __init__(self, env, act_enc, env_key, speed, thigmotaxis,
                 HDbins, FoV_params, Grid_params,
                 SigmaD, SigmaA, seed, repeats = np.array([1,1,1,1]),
                 multiply=False, **kwargs):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins,
                         SigmaD, SigmaA,
                         repeats, FoV_params)
        self.n_obs = 4
        self.multiply = multiply

        np.random.seed(seed)
        self.grid = GridCells(self.ag, params=Grid_params)

        coords = env.objects["objects"]         # shape (N, 2)
        types  = env.objects["object_types"]    # shape (N,)

        # Create a mask (Boolean array) for where types == 0
        mask = (types == 0)
        coords_type_0 = coords[mask]
        if len(coords_type_0) < 3:
            raise ValueError("Not enough holes in the environment to set the specified number of rewards.")


        np.random.seed(seed+101)
        reward_hole_indices = np.random.choice(len(coords_type_0), 3, replace=False)
        reward_positions = coords_type_0[reward_hole_indices]
    
        #Create reward neuron (another place cell hidden behind the barrier) 
        self.Reward = PlaceCells(
        self.ag,
        params={
            "n": 3,
            "place_cell_centres": np.array(reward_positions),
            #"description": "top_hat", #commenting out to use gaussian instead
            "widths": 0.025, 
            "max_fr": 1,
            "color": "C5",
            "wall_geometry": "euclidean",
        },
        )
        
        self.reset()

    def getObservations(self, tsteps, reset=True, includeRender=False,
                        discretize=False, inv_x=False, inv_y=False):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """

        render = False # Placeholder for compatibility, actual render is in the 'show_state(_traj)' function
        if reset:
            self.reset()
        else:
            self.reset(keep_state=True)

        for aa in range(tsteps):
            self.ag.update()
            self.grid.update()
            self.Reward.update() #switched from self.grid.update()
            for i in range(len(self.vision)):
                self.vision[i].update()

        rot_vel = np.array(self.ag.history['rot_vel'][1:])*self.ag.dt/np.pi
        vel = np.array(self.ag.history['vel'][1:])*self.ag.dt
        act = np.concatenate((rot_vel[:,None], vel), axis=1)
        obs_vis = np.concatenate([np.array(self.vision[i].history["firingrate"])[...,None]\
                              for i in range(len(self.vision))], axis=-1)
        obs_vis_inside = obs_vis[..., 1]

        obs_reward = np.array(self.Reward.history["firingrate"])
        obs_grid = np.array(self.grid.history["firingrate"])
        obs = (obs_vis_inside, obs_vis, obs_grid, obs_reward)

        pos = np.array(self.ag.history['pos'])
        if discretize:
            # Transform the positions from continuous float coordinates to discrete int coordinates
            dx = self.env.dx
            coord = self.env.flattened_discrete_coords
            dist = get_distances_between(np.array(pos), coord)
            pos = ((coord[dist.argmin(axis=1)]-dx/2)/dx).astype(int)
        if inv_x:
            max_x = np.round(pos[:,0].max())
            pos[:,0] = max_x - pos[:,0]
        if inv_y:
            max_y = np.round(pos[:,1].max())
            pos[:,1] = max_y - pos[:,1]

        state = {'agent_pos': pos, 
                 'agent_dir': np.array([get_angle(x) for x in self.ag.history['vel']]),
                 'mean_vel': self.ag.speed_mean,
                }

        return obs, act, state, render


    def env2pred(self, obs, act=None):
        """
        Convert observation and action input to pytorch arrays for input to predictive net.
        """
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean

        obs_vis_inside, obs_vis, obs_grid, obs_reward = obs

        obs_vis_inside = torch.tensor(obs_vis_inside, dtype=torch.float,
                                      requires_grad=False)
        obs_vis_inside = torch.unsqueeze(obs_vis_inside, dim=0)

        remix = self.vision_to_rgb(obs_vis)
        remix = torch.tensor(remix, dtype=torch.float, requires_grad=False)
        remix = torch.unsqueeze(remix, dim=0)

        obs_reward = torch.tensor(obs_reward.clip(max=1), dtype=torch.float, requires_grad=False)
        obs_reward = torch.unsqueeze(obs_reward, dim=0)

        obs_grid = torch.tensor(obs_grid.clip(max=1), dtype=torch.float, requires_grad=False)
        obs_grid = torch.unsqueeze(obs_grid, dim=0)

        obs = [obs_vis_inside, remix, obs_grid, obs_reward]
        return obs, act

    def env2np(self, obs, act=None):
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean
        act = np.array(act)

        obs_vis_inside, obs_vis, obs_grid, obs_reward = obs

        obs_vis_inside = obs_vis_inside[None]
        remix = self.vision_to_rgb(obs_vis)[None]
        obs_reward = obs_reward.clip(max=1)[None]
        obs_grid = obs_grid.clip(max=1)[None]

        obs = (obs_vis_inside, remix, obs_grid, obs_reward)
        return obs, act

    
    def pred2np(self, obs, whichPhase=0, timesteps=None):
        """
        Convert sequence of observations from pytorch format to image-filled np.array
        """
        if not obs[1] == None:
            obs = obs[1].detach().numpy()
            if timesteps:
                obs = obs[:,timesteps,...]
            img = []
            for t in range(obs.shape[1]):
                img.append(self.to_image(obs[whichPhase,t])[None,...])
            obs = np.concatenate(img, axis=0)
        #TODO: do for wells
        else:
            for i in range(self.n_obs):
                if obs[i] is not None:
                    if timesteps:
                        obs[i] = obs[i][:,timesteps,...]
                    o = np.ones([obs[i].shape[1],1,1,3])
                    break
            obs = o
        return obs
    
    def to_image(self, obs):
        fig, ax = plt.subplots()


        obs = obs.reshape(-1,3)

        y = self.vision[0].tuning_distances * np.cos(self.vision[0].tuning_angles)
        x = self.vision[0].tuning_distances * np.sin(self.vision[0].tuning_angles) + 0.5
        ww = (self.vision[0].sigma_angles * self.vision[0].tuning_distances)
        hh = self.vision[0].sigma_distances
        aa  = self.vision[0].tuning_angles * 180 / np.pi
        ec = EllipseCollection(ww,hh, aa, units = 'x',
                                offsets = np.array([x,y]).T,
                                offset_transform = ax.transData,
                                linewidth=0.5,
                                edgecolor="dimgrey",
                                zorder = 2.1,
                                )
        ec.set_facecolors(obs)

        ax.add_collection(ec) 

        plt.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0.15)
        plt.gca().invert_xaxis()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image_from_plot

    def getObsSize(self):
        obs_size = np.array((self.vision[0].n, self.vision[0].n * 3, self.grid.n, self.Reward.n)) * self.repeats**(not self.multiply)
        return obs_size
    
    def reset(self, pos=None, vel=[0,0], seed=False, keep_state=False):
        if not hasattr(self, 'Reward'):
            return
        self.ag.reset_history()
        # self.ag.t = 0
        self.Reward.reset_history()
        self.grid.reset_history()
        for i in range(len(self.vision)):
            self.vision[i].reset_history()
        
        if keep_state:
            vel = self.ag.velocity
            pos = self.ag.pos

        if pos is None and hasattr(self.env, "home_pos"):
            pos = self.env.home_pos.copy()

        if pos is not None:
            self.ag.pos = pos

        self.ag.save_velocity = vel
        self.ag.save_to_history()
        self.Reward.update()
        self.grid.update()
        for i in range(len(self.vision)):
            self.vision[i].update()




class RiaBColorsRewardDirectedShell(RiaBColorsRewardShell):
    #showstatetrajectory

    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params,
                 seed, SigmaD= 0.1, SigmaA = 2,
                 repeats = np.array([1,1]), time_at_reward=1, num_place_cells=200,
                 training_length = 60000, multiply=False):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins,
                         FoV_params, SigmaD, SigmaA,
                         seed, repeats, multiply)

        self.time_at_reward = time_at_reward

        self.active_reward_idx  = 0
        self.success_threshold  = 0.9 #switched from 0.7 to experiment  

        self.total_tsteps = 0 
        self.total_training_length = training_length

        np.random.seed(seed+1000)
        n_pc = num_place_cells #run from 200 to 1000
        self.Inputs = PlaceCells(
            self.ag,
            params={
                "n": n_pc,
                #"widths": np.random.uniform(0.04, 0.4, size=(n_pc)), #large and small widths
                "widths": np.random.uniform(0.04, 0.3, size=(n_pc)),
                "color": "C1",
            },
        )

        self.ValNeur = ValueNeuron(
            self.ag, params={
                        "input_layers": [self.Inputs], 
                        "tau": 15, #default was 10, this seems to work better
                        "eta": 0.001,  
                        "L2": 0.1,  # L2 regularisation
                        "activation_function": {"activation": "relu"}, #can try with relu, tanh, softmax etc. see ratinabox/utils.py: activate() for list
                        "color": "C2",
                        "n": 4}
        )
        w = self.ValNeur.inputs["PlaceCells"]["w"]
        #self.ValNeur.inputs["PlaceCells"]["w"] = 0.01 * np.random.randn(*w.shape) 
         #to be periodically updated, a scale for how "big" the vf is so we know where the threshold is

        # --- shrink initial weights so L2 penalty starts tiny but nonzero ---

        w *= 1e-3


        # now set up your max_value tracking as before
        self.ValNeur.max_value = np.full(self.ValNeur.n, 1e-6)

        # self.all_firing_rates = []

        self.ag.history["active_reward"] = []
        
        self.reset()

    def create_rewards(self, reward_positions):
        self.reward_positions = reward_positions
        #add reward position for home base
        reward_positions = np.concatenate([reward_positions, self.env.home_pos[None]], axis=0)

        self.Reward = PlaceCells(
            self.ag,
            params={
                "n": 4,
                "place_cell_centres": np.array(reward_positions),
                #"description": "top_hat", #commenting out to use gaussian instead
                "widths": 0.025,
                "max_fr": 1,
                "color": "C5",
                "wall_geometry": "euclidean",
            },
        )

    
    def init_agent(self, speed, thigmotaxis):
        # Make the agent
        self.ag = Agent(self.env,
                        params={
                                "dt": 0.1,
                                "speed_mean": speed,
                                "thigmotaxis": thigmotaxis,
                                "wall_repel_distance": 0.03,
                                "wall_repel_strength": 0.5})
        
        self.ag.exploit_explore_ratio = 1.0  # exploit/explore parameter we will use later

    def new_curriculum(self, num_sequences, seqlength):
        self.ValNeur.tau = 10
        for i in range(num_sequences):
            self.active_reward_idx = i % 3
            self.getRandomPos()
            print("Starting sequence", i, "with active reward channel", self.active_reward_idx)
            tsteps = seqlength
            obs, act, state, render = self.getObservations(tsteps, new_curriculum=True)
            self.ValNeur.plot_rate_map()
            fig, ax = plt.subplots(figsize=(10,10))
            self.show_state_traj(0, seqlength, fig, ax)
        return True


    def getObservations(self, tsteps, reset=True, includeRender=False,
                        discretize=False, inv_x=False, inv_y=False, new_curriculum=False,
                        set_explore_ratio = False, explore_ratio = 0.1):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """
        switches = 0

        render = False # Placeholder for compatibility, actual render is in the 'show_state(_traj)' function
        if reset:
            self.reset()
        # else:
        #     self.reset(keep_state=True)

        t_at_goal = 0

        if(set_explore_ratio):
            self.ag.exploit_explore_ratio = explore_ratio
        for k in range(tsteps):

            if(not set_explore_ratio):
                self.ag.exploit_explore_ratio = 0.1 + (self.total_tsteps / self.total_training_length) * 0.3 #continuous increase of the exploit/explore ratio
            gradV = self.get_steep_ascent(self.ag.pos)
            if gradV is not None:
                drift_vel = 3 * self.ag.speed_mean * gradV
            else:
                drift_vel = None
                # if the gradient is untrusted, go full random

          

            idx = self.active_reward_idx
            if(idx == 3):
                center_x, center_y = self.Reward.place_cell_centres[idx]
                dir_to_reward = self.Reward.place_cell_centres[idx] - self.ag.pos #cheat to sent it to center
                drift_vel = (
                    3 * self.ag.speed_mean * (dir_to_reward / np.linalg.norm(dir_to_reward))
                )
            else:
                center_x, center_y = self.Reward.place_cell_centres[idx]
                radius = 0.1 
                dx = self.ag.pos[0] - center_x #x-distance from center
                dy = self.ag.pos[1] - center_y #y-distance from center
                if (dx*dx + dy*dy) <= radius*radius: #checks if agent is within radius
                    dir_to_reward = self.Reward.place_cell_centres[idx] - self.ag.pos #cheat to sent it to center
                    drift_vel = (
                        3 * self.ag.speed_mean * (dir_to_reward / np.linalg.norm(dir_to_reward))
                    )



            self.ag.update(drift_velocity = drift_vel, drift_to_random_strength_ratio = self.ag.exploit_explore_ratio)

            for v in self.vision:
                v.update()
            self.Reward.update()
            self.Inputs.update()
            self.ValNeur.update()

            self.ValNeur.update_weights(reward = self.Reward.firingrate)

            self.ValNeur.max_value = np.maximum(self.ValNeur.max_value, self.ValNeur.firingrate)
            self.ag.history["active_reward"].append(self.active_reward_idx)


            # rotate goal if close enough to the active target
            if not new_curriculum:
                if (self.Reward.firingrate[self.active_reward_idx] > self.success_threshold):
                    t_at_goal += 1
                    if(t_at_goal > self.time_at_reward): # if the agent is at the goal for more than n timesteps, switch to the next reward
                        self.active_reward_idx = (self.active_reward_idx + 1) % self.ValNeur.n
                        print("Reward channel changed to", self.active_reward_idx)
                        t_at_goal = 0
                        switches += 1
            self.total_tsteps += 1
            if(self.total_tsteps % 10000 == 0):
                print("Exploit/Explore ratio: ", self.ag.exploit_explore_ratio)
                self.ValNeur.plot_rate_map()
                if (k > 0):
                    fig, ax = plt.subplots(figsize=(10,10))
                    self.show_state_traj(self.total_tsteps - 10000, self.total_tsteps - 1, fig, ax)



        rot_vel = np.array(self.ag.history['rot_vel'][1:])*self.ag.dt/np.pi
        vel = np.array(self.ag.history['vel'][1:])*self.ag.dt
        act = np.concatenate((rot_vel[:,None], vel), axis=1)
        obs_vis = np.concatenate([np.array(self.vision[i].history["firingrate"])[...,None]\
                              for i in range(len(self.vision))], axis=-1)

                              
        obs_reward = np.array(self.Reward.history["firingrate"])[:,:3]
        obs = (obs_vis, obs_reward)

        pos = np.array(self.ag.history['pos'])
        if discretize:
            # Transform the positions from continuous float coordinates to discrete int coordinates
            dx = self.env.dx
            coord = self.env.flattened_discrete_coords
            dist = get_distances_between(np.array(pos), coord)
            pos = ((coord[dist.argmin(axis=1)]-dx/2)/dx).astype(int)
        if inv_x:
            max_x = np.round(pos[:,0].max())
            pos[:,0] = max_x - pos[:,0]
        if inv_y:
            max_y = np.round(pos[:,1].max())
            pos[:,1] = max_y - pos[:,1]

        state = {'agent_pos': pos, 
                 'agent_dir': np.array([get_angle(x) for x in self.ag.history['vel']]),
                 'mean_vel': self.ag.speed_mean,
                }
        
        # self.all_firing_rates = np.array(self.all_firing_rates)

        print("Number of switches:", switches)

        return obs, act, state, render
    
    def getRandomPos(self):
        center = np.array([0.6, 0.6])
        radius = 0.6

        # pick a random angle
        theta = np.random.uniform(0, 2*np.pi)
        # pick a radius with sqrt-uniform to ensure uniform area density
        r = np.sqrt(np.random.uniform(0, 1)) * radius

        # convert to Cartesian
        offset = np.array([r * np.cos(theta), r * np.sin(theta)])
        self.ag.pos = center + offset

        # optional: randomize heading too
        self.ag.heading = np.random.uniform(0, 2*np.pi)

    
    def reset(self, pos=None, vel=[0,0], seed=False, keep_state=False):
        if not hasattr(self, 'ValNeur'):
            return
        super().reset(pos=pos, vel=vel, seed=seed, keep_state=keep_state)

        self.active_reward_idx = 0
        self.ValNeur.reset() 
        self.Inputs.reset_history()



    def get_steep_ascent(self, pos):
        """
        Direction of steepest ascent of the *active* value field.
        Returns None when the local value is too small to be trusted.
        """
        idx = self.active_reward_idx

        V = self.ValNeur.get_state(evaluate_at=None,pos=pos)[idx][0]
        if V <= 0.000001 * self.ValNeur.max_value[idx]:   # per‑channel scale
            return None

        # finite‑difference gradient for that channel
        dx     = np.array([1e-3, 0])
        dy     = np.array([0, 1e-3])
        Vx     = self.ValNeur.get_state(evaluate_at=None,pos=pos+dx)[idx][0]
        Vy     = self.ValNeur.get_state(evaluate_at=None,pos=pos+dy)[idx][0]
        gradV  = np.array([Vx-V, Vy-V])

        norm   = np.linalg.norm(gradV)
        if norm < math.exp(-9):
            return None
        return gradV / norm
    

    # def get_steep_ascent(self, pos):
    #     idx = self.active_reward_idx
    #     # skip the value‐threshold gate so you always follow ∇V:
    #     V = self.ValNeur.get_state(pos=pos)[idx]
    #     # if you really want a tiny gate, use e.g.: 
    #     # if V <= 1e-6: return None

    #     # finite-difference gradient
    #     dx = np.array([1e-3, 0])
    #     dy = np.array([0, 1e-3])
    #     Vx     = self.ValNeur.get_state(pos=pos+dx)[idx]
    #     Vy     = self.ValNeur.get_state(pos=pos+dy)[idx]
    #     gradV  = np.array([Vx-V, Vy-V])
    #     norm   = np.linalg.norm(gradV)
    #     if norm < 1e-9: 
    #         return None
    #     return gradV / norm


    def _current_reward_vector(self):
        """1‑hot reward vector: only the active channel is non‑zero."""
        r      = np.zeros(self.ValNeur.n)
        r[self.active_reward_idx] = self.Reward.firingrate[self.active_reward_idx]
        return r
    def get_reward_positons(self): 
        return self.reward_positions

def sample_in_circle(center, radius, rng=np.random):
    theta = rng.uniform(0.0, 2*np.pi)       # angle
    r     = radius * np.sqrt(rng.uniform()) # radius with √ trick
    return np.asarray(center) + np.array([r*np.cos(theta), r*np.sin(theta)])
