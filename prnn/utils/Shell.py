import torch
import numpy as np
import matplotlib
import random

from matplotlib.collections import EllipseCollection

from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from ratinabox.utils import get_angle, get_distances_between

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
            obs = [o[:,:tsteps+1] for o in data[:self.n_obs]]
            if len(obs) == 1:
                obs = obs[0]
            act = data[self.n_obs][:,:tsteps]
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
    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins):
        super().__init__(env, act_enc, env_key)
        "For ratinabox==1.7.1, for other versions you may have to update the code."

        # Create the agent
        Ag = Agent(self.env, {
                    'dt': 0.1,
                    'speed_mean': speed,
                    'thigmotaxis': thigmotaxis
                    })
        
        self.ag = Ag
        self.numHDs = HDbins # For One-hot encoding of HD if needed

        self.height = int((self.env.extent[3] - self.env.extent[2])/env.dx)
        self.width = int((self.env.extent[1] - self.env.extent[0])/env.dx)

        self.true_height = self.env.extent[3] - self.env.extent[2]
        self.true_width = self.env.extent[1] - self.env.extent[0]
        self.max_dist = (self.true_height**2 + self.true_width**2)**0.5

        self.continuous = True
        self.start_pos = 0

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
                             'distance_range': [0.0, 0.33]}
                             ):
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





class RiaBVisionShell2(RatInABoxShell):
    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins,
                 FoV_params={'spatial_resolution': 0.01,
                             'angle_range': [0, 45],
                             'distance_range': [0.0, 0.33]}
                             ):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins)

        # Create vision cells

        FoV_Walls = FieldOfViewBVCs(self.ag, params=FoV_params)
        FoV_Walls.sigma_distances = np.full((60,), 0.02)
        FoV_Walls.sigma_angles /= 2#np.full((60,), 1.)
        FoV_Objects = [FieldOfViewOVCs(self.ag, params=FoV_params | {
            "object_tuning_type": x
            }) for x in range(env.n_object_types)]
        FoV_Objects[0].sigma_distances = np.full((60,), 0.02)
        FoV_Objects[0].sigma_angles /= 2#np.full((60,), 1.)
        
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
    
    def reset(self, pos=np.zeros(2), vel=None, seed=False, keep_state=False):

        self.ag.reset_history()
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


class RiaBRemixColorsShell(RiaBVisionShell):
    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params):
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
                 thigmotaxis, HDbins, Grid_params):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins)

        # Create grid cells
        np.random.seed(42) # Otherwise there will be a discrepancy with the data from dataloader
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
                 thigmotaxis, HDbins, FoV_params, Grid_params):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params)
        self.n_obs = 2
        # Create grid cells
        np.random.seed(42) # Otherwise there will be a discrepancy with the data from dataloader
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
        obs_size = (self.vision[0].n * 3, self.grid.n)
        return obs_size
    
    def reset(self, pos=np.zeros(2), vel=None, seed=False, keep_state=False):
        if not hasattr(self, 'grid'):
            return
        self.ag.reset_history()
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


class RiaBColorsRewardShell(RiaBVisionShell2): #switching to 2 to test dif sigma distances and angles (Hadrien)


    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params, seed, n_repeats = 1):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params)
        self.n_obs = 2
        self.n_repeats = n_repeats

        coords = env.objects["objects"]         # shape (N, 2)
        types  = env.objects["object_types"]    # shape (N,)

        # Create a mask (Boolean array) for where types == 0
        mask = (types == 0)
        coords_type_0 = coords[mask]
        if len(coords_type_0) < 3:
            raise ValueError("Not enough holes in the environment to set the specified number of rewards.")
        np.random.seed(seed)
        reward_hole_indices = np.random.choice(len(coords_type_0), 3, replace=False)
        reward_positions = env.objects['objects'][reward_hole_indices]

        # Make the agent 
        Ag = Agent(env)
        Ag.dt = 50e-3  # set discretisation time, large is fine
        Ag.episode_data = {
            "start_time": [],
            "end_time": [],
            "start_pos": [],
            "end_pos": [],
            "success_or_failure": [],
        }  # a dictionary we will use later
    
        #Create reward neuron (another place cell hidden behind the barrier) 
        self.Reward = PlaceCells(
        Ag,
        params={
            "n": 3,
            "place_cell_centres": np.array(reward_positions),
            "description": "top_hat",
            "widths": 0.04,
            "max_fr": 1,
            "color": "C5",
            "wall_geometry": "euclidean",
        },
        )

        self.home_pos = env.sample_random_position()[0]   
        Ag.pos = self.home_pos.copy() 
        Ag.save_to_history()   
        
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
            self.Reward.update() #switched from self.grid.update()
            for i in range(len(self.vision)):
                self.vision[i].update()

        rot_vel = np.array(self.ag.history['rot_vel'][1:])*self.ag.dt/np.pi
        vel = np.array(self.ag.history['vel'][1:])*self.ag.dt
        act = np.concatenate((rot_vel[:,None], vel), axis=1)
        obs_vis = np.concatenate([np.array(self.vision[i].history["firingrate"])[...,None]\
                              for i in range(len(self.vision))], axis=-1)
        obs_reward = np.array(self.Reward.history["firingrate"]) #switched from self.grid.history["firingrate"]
        obs_reward = np.tile(reward_obs, self.n_repeats)  
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

        remix = np.zeros((*obs_vis.shape[:-1],3))
        remix += np.tile(obs_vis[...,0,None],3)*50/255
        if 'LRoom' in self.name:
            remix[...,2] += obs_vis[...,1]
            remix[...,0] += obs_vis[...,2]
            remix[...,0] += obs_vis[...,3]
            remix[...,1] += obs_vis[...,3]
        else:
            for i in range(1,obs_vis.shape[-1]):
                remix += np.moveaxis(np.tile(obs_vis[...,i], [3]+[1]*(len(obs_vis[...,i].shape))),
                                     0,
                                     -1
                                     ) * self.obs_colors[i][:3]
        remix = remix.clip(max=1)
        remix = remix.reshape(remix.shape[:-2]+(-1,))
        remix = torch.tensor(remix, dtype=torch.float, requires_grad=False)
        remix = torch.unsqueeze(remix, dim=0)


        obs_Reward = obs_Reward.clip(max=1)
        obs_Reward = torch.tensor(obs_Reward, dtype=torch.float, requires_grad=False)
        obs_Reward = torch.unsqueeze(obs_Reward, dim=0)

        obs = (remix, obs_Reward)

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
        obs_size = (self.vision[0].n * 3, self.Reward.n)
        return obs_size
    
    def reset(self, pos=np.zeros(2), vel=None, seed=False, keep_state=False):
        if not hasattr(self, 'Reward'):
            return
        self.ag.reset_history()
        self.Reward.reset_history()
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
        self.Reward.update()
        for i in range(len(self.vision)):
            self.vision[i].update()


class RiaBColorsRewardDirectedShell(RiaBVisionShell2):
    #showstatetrajectory

    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins, FoV_params)
        self.n_obs = 2

        coords = env.objects["objects"]         # shape (N, 2)
        types  = env.objects["object_types"]    # shape (N,)

        # Create a mask (Boolean array) for where types == 0
        mask = (types == 0)
        coords_type_0 = coords[mask]
        if len(coords_type_0) < 3:
            raise ValueError("Not enough holes in the environment to set the specified number of rewards.")
        #add a seed to make it reproducible TODO (add as argument as well)
        reward_hole_indices = np.random.choice(len(coords_type_0), 3, replace=False)
        reward_positions = env.objects['objects'][reward_hole_indices]

        self.active_reward_idx  = 0
        self.success_threshold  = 0.7 
        self.reward_positions = reward_positions

        # Make the agent 
        Ag = Agent(Env)
        Ag.dt = 50e-3  # set discretisation time, large is fine
        Ag.episode_data = {
            "start_time": [],
            "end_time": [],
            "start_pos": [],
            "end_pos": [],
            "success_or_failure": [],
        }  # a dictionary we will use later
        Ag.exploit_explore_ratio = 0.3  # exploit/explore parameter we will use later
        #Ag.exploit_explore_ratio = 1 # set this high so agent climbs the VF


        #Input features (200 place cells of random widths)
        n_pc = 200 
        Inputs = PlaceCells(
            Ag,
            params={
                "n": n_pc,
                "widths": np.random.uniform(0.04, 0.4, size=(n_pc)), #large and small widths
                "color": "C1",
            },
        )
        # just manually setting the last four cell's widths and locations for some plots I wish to make later, this is not critical
        Inputs.place_cell_widths[-4:] = 0.2
        Inputs.place_cell_centres[-4:] = np.array(
            [[0.15, 0.55], [0.55, 0.25], [0.65, 0.7], [0.9, 0.7]]
        )
        Inputs.plot_rate_map(chosen_neurons=[-4, -3, -2, -1], autosave=False)


        #The reward neuron (another place cell hidden behind the barrier) 
        Reward = PlaceCells(
            Ag,
            params={
                "n": 3,
                "place_cell_centres": np.array(reward_positions),
                "description": "top_hat",
                "widths": 0.2,
                "max_fr": 1,
                "color": "C5",
                "wall_geometry": "euclidean",
            },
        )
        Reward.episode_end_time = 3  # a param we will use later
        Reward.plot_rate_map(chosen_neurons="1")



        # The Value Neuron
        # Now initialise a Value neuron, full documentation of this class in ratinabox/contribs/ValueNeuron.py
        # It's a feedforward layer summing (then non-linearly activating) its input features. 
        # It takes: 
        # • a list of inputs (the place cells we just made) 
        # • a timescale for the discounting of future rewards (tau)
        # • a non-linear activation function (relu)
        ValNeur = ValueNeuron(
            Ag, params={
                        "input_layers": [Inputs], 
                        "tau": 10,
                        "eta":0.001,
                        "L2": 0.1,  # L2 regularisation
                        "activation_function": {"activation": "relu"}, #can try with relu, tanh, softmax etc. see ratinabox/utils.py: activate() for list
                        "color": "C2"
                        "n": 3}
        )
        ValNeur.inputs['PlaceCells']['w'] *= 0.01
        ValNeur.max_value = np.max(ValNeur.get_state(evaluate_at='all')) #to be periodically updated, a scale for how "big" the vf is so we know where the threshold is

        #have value update for 3 rewards seperately, then relay between the 3 
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

            #get greedy direction of steepest ascent of the value function
            gradV = get_steep_ascent(self.ValNeur, Ag.pos)
            if gradV is None: drift_velocity = None #if None, the agent will just randomly explore
            else: drift_velocity = 3 * Ag.speed_mean * gradV
            # you can ignore this (force agent to travel towards reward when v nearby) helps stability.
            if (Ag.pos[0] > 0.8) and (Ag.pos[1] < 0.4):
                dir_to_reward = Reward.place_cell_centres[0] - Ag.pos
                drift_velocity = (
                    3 * Ag.speed_mean * (dir_to_reward / np.linalg.norm(dir_to_reward))
                )

            # move the agent
            Ag.update(
                drift_velocity=drift_velocity,
                drift_to_random_strength_ratio=Ag.exploit_explore_ratio,
            )

            self.Reward.update() #switched from self.grid.update()
            self.Inputs.update()
            self.ValNeur.update() 

            self.ValNeur.update_weights(reward = self._current_reward_vector())

            if (self.Reward.firingrate[self.active_reward_idx] > self.success_threshold):
                # rotate to the next reward channel
                self.active_reward_idx = (self.active_reward_idx + 1) % self.ValNeur.n

            for i in range(len(self.vision)):
                self.vision[i].update()

        rot_vel = np.array(self.ag.history['rot_vel'][1:])*self.ag.dt/np.pi
        vel = np.array(self.ag.history['vel'][1:])*self.ag.dt
        act = np.concatenate((rot_vel[:,None], vel), axis=1)
        obs_vis = np.concatenate([np.array(self.vision[i].history["firingrate"])[...,None]\
                              for i in range(len(self.vision))], axis=-1)
        obs_grid = np.array(self.Reward.history["firingrate"]) #switched from self.grid.history["firingrate"]
        obs_inputs = np.array(self.Inputs.history["firingrate"])
        obs_valneur = np.array(self.ValNeur.history["firingrate"])
        obs = (obs_vis, obs_grid, obs_inputs, obs_valneur)

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

        remix = np.zeros((*obs_vis.shape[:-1],3))
        remix += np.tile(obs_vis[...,0,None],3)*100/255
        if 'LRoom' in self.name:
            remix[...,2] += obs_vis[...,1]
            remix[...,0] += obs_vis[...,2]
            remix[...,0] += obs_vis[...,3]
            remix[...,1] += obs_vis[...,3]
        else:
            for i in range(1,obs_vis.shape[-1]):
                remix += np.moveaxis(np.tile(obs_vis[...,i], [3]+[1]*(len(obs_vis[...,i].shape))),
                                     0,
                                     -1
                                     ) * self.obs_colors[i][:3]
        remix = remix.clip(max=1)
        remix = remix.reshape(remix.shape[:-2]+(-1,))
        remix = torch.tensor(remix, dtype=torch.float, requires_grad=False)
        remix = torch.unsqueeze(remix, dim=0)


        obs_Reward = obs_Reward.clip(max=1)
        obs_Reward = torch.tensor(obs_Reward, dtype=torch.float, requires_grad=False)
        obs_Reward = torch.unsqueeze(obs_Reward, dim=0)

        obs_inputs = obs_inputs.clip(max=1)
        obs_inputs = torch.tensor(obs_inputs, dtype=torch.float, requires_grad=False)
        obs_inputs = torch.unsqueeze(obs_inputs, dim=0)

        obs_valneur = obs_valneur.clip(max=1)
        obs_valneur = torch.tensor(obs_valneur, dtype=torch.float, requires_grad=False)
        obs_valneur = torch.unsqueeze(obs_valneur, dim=0)

        obs = (remix, obs_Reward, obs_inputs, obs_valneur)

        return obs, act

    def env2np(self, obs, act=None):
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_mean
        act = np.array(act)

        obs_vis, obs_Reward = obs

        remix = np.zeros((*obs_vis.shape[:-1],3))
        remix += np.tile(obs_vis[...,0,None],3)*100/255
        if 'LRoom' in self.name:
            remix[...,2] += obs_vis[...,1]
            remix[...,0] += obs_vis[...,2]
            remix[...,0] += obs_vis[...,3]
            remix[...,1] += obs_vis[...,3]
        else:
            for i in range(1,obs_vis.shape[-1]):
                remix += np.moveaxis(np.tile(obs_vis[...,i], [3]+[1]*(len(obs_vis[...,i].shape))),
                                     0,
                                     -1
                                     ) * self.obs_colors[i][:3]
        remix = remix.clip(max=1)
        remix = remix.reshape(remix.shape[:-2]+(-1,))[None]


        obs_Reward = obs_Reward.clip(max=1)[None]
        obs_inputs = obs_inputs.clip(max=1)[None]
        obs_valneur = obs_valneur.clip(max=1)[None]

        obs = (remix, obs_Reward, obs_inputs, obs_valneur)

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
        print(image_from_plot.shape)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image_from_plot

    def getObsSize(self):
        obs_size = (self.vision[0].n * 3, self.Reward.n)
        return obs_size
    
    def reset(self, pos=np.zeros(2), vel=None, seed=False, keep_state=False):
        if not hasattr(self, 'Reward'):
            return
        self.active_reward_idx = 0
        self.ag.reset_history()
        self.Reward.reset_history()
        for i in range(len(self.vision)):
            self.vision[i].reset_history()
        
        if keep_state:
            vel = self.ag.vel
            pos = self.ag.pos

        if vel:
            self.ag.pos = pos
        else:
            vel = [0,0]

        self.ag.save_velocity = vel
        self.ag.save_to_history()
        self.Reward.update()
        self.ValNeur.reset() 
        self.Inputs.reset()
        for i in range(len(self.vision)):
            self.vision[i].update()
    
    def get_steep_ascent(ValueNeuron, pos):
        """This function will be used for policy improvement. Calculates direction steepest ascent (gradient) of the value function and returns a drift velocity in this direction. Returns None when the local gradient is exceedingly low"""
        V = ValueNeuron.get_state(evaluate_at=None, pos=pos)[0][0] #query the firing rate at the given position
        if V <= 0.05*ValueNeuron.max_value:
            return None # if the value function is too low it is unreliable, return None
        else:  # calculate gradient locally
            V_plusdx = ValueNeuron.get_state(evaluate_at=None, pos=pos + np.array([1e-3, 0]))[0][0]
            V_plusdy = ValueNeuron.get_state(evaluate_at=None, pos=pos + np.array([0, 1e-3]))[0][0]
            gradV = np.array([V_plusdx - V, V_plusdy - V])
            norm = np.linalg.norm(gradV)
            gradV = gradV / norm
            return gradV

    def _current_reward_vector(self):
        """1‑hot reward vector: only the active channel is non‑zero."""
        r      = np.zeros(self.ValNeur.n)
        r[self.active_reward_idx] = self.Reward.firingrate[self.active_reward_idx]
        return r

