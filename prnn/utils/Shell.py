import torch
import numpy as np
import matplotlib
import random

from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from ratinabox.contribs.FieldOfViewNeurons import FieldOfViewNeurons

from prnn.utils.ActionEncodings import *
from prnn.utils.general import saveFig

actionOptions = {'OnehotHD' : OneHotHD ,
                 'OnehotHDPrevAct' : OneHotHDPrevAct,
                 'SpeedHD' : SpeedHD ,
                 'SpeedNextHD' : SpeedNextHD,
                 'Onehot' : OneHot,
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
                obs, act = next(self.DL_iterator)
            except StopIteration:
                self.DL_iterator = iter(self.dataLoader)
                obs, act = next(self.DL_iterator)
            obs = obs[:,:tsteps+1]
            act = act[:,:tsteps]
            state = None
            render = None
        else:
            for bb in range(batch_size):
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
    
    @property
    def action_space(self):
        return self.env.action_space

    def dir2deg(self, dir):
        return HDmap[dir]
    
    def env2pred(self, obs, act=None):
        hd = np.array([self.get_hd(obs[t]) for t in range(len(obs))])
        if act is not None:
            act = self.encodeAction(act=act,
                                    obs=hd,
                                    numSuppObs=self.numHDs,
                                    numActs=self.action_space.n)

        obs = np.array([self.get_visual(obs[t]) for t in range(len(obs))])
        obs = torch.tensor(obs, dtype=torch.float, requires_grad=False)
        obs = torch.unsqueeze(obs, dim=0)
        obs = obs/255 #normalize image

        return obs, act
    
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

    def pred2np(self, obs, whichPhase=0):
        obs = obs.detach().numpy()
        obs = np.reshape(obs[whichPhase,:,:],(-1,)+self.obs_shape)
        return obs
    
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
    
    def get_viewpoint(self, agent_pos, agent_dir):

        self.env.unwrapped.agent_dir = agent_dir
        self.env.unwrapped.agent_pos = agent_pos
        
        obs = self.env.gen_obs()
        obs = self.env.observation(obs)
        
        return obs['image']/255
    
    def load_state(self, state):
        self.set_agent_pos(state[:2])
        self.set_agent_dir(state[2])
    
    def save_state(self, act, state):
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

        # Create vision cells
        FoV_Walls = FieldOfViewNeurons(Ag, params={
            'spatial_resolution': 0.045,
            "FoV_angles": [0, 45],
            "FoV_distance": [0.0, 0.33],
            })
        locs = self.env.discretise_environment(dx=0.04)
        locs = locs.reshape(-1, locs.shape[-1])
        FoV_Walls.super.cell_fr_norm = np.ones(FoV_Walls.super.n)
        FoV_Walls.super.cell_fr_norm = np.max(FoV_Walls.super.get_state(evaluate_at=None, pos=locs), axis=1)
        FoV_Objects = FieldOfViewNeurons(Ag, params={
            'spatial_resolution': 0.045,
            "FoV_angles": [0, 45],
            "FoV_distance": [0.0, 0.33],
            'cell_type':'OVC',
            })
        
        self.ag = Ag
        self.vision = (FoV_Walls, FoV_Objects)
        self.numHDs = HDbins # For One-hot encoding of HD if needed

        self.height = int((self.env.extent[3] - self.env.extent[2])/env.dx)
        self.width = int((self.env.extent[1] - self.env.extent[0])/env.dx)

        self.true_height = self.env.extent[3] - self.env.extent[2]
        self.true_width = self.env.extent[1] - self.env.extent[0]
        self.max_dist = (self.true_height**2 + self.true_width**2)**0.5

        self.continuous = True

        self.reset()

    def dir2deg(self, dir):
        return np.rad2deg(dir)+90

    def discretize(self, dx):
        self.env.dx = dx
        self.env.discrete_coords = self.env.discretise_environment(dx=dx)
        self.env.flattened_discrete_coords = self.env.discrete_coords.reshape(
            -1, self.env.discrete_coords.shape[-1]
        )
        self.height = int((self.env.extent[3] - self.env.extent[2])/dx)
        self.width = int((self.env.extent[1] - self.env.extent[0])/dx)
    
    def env2pred(self, obs, act=None):
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_means

        obs = obs.clip(max=1)
        obs = obs.reshape(obs.shape[:-2]+(-1,))
        obs = torch.tensor(obs, dtype=torch.float, requires_grad=False)
        obs = torch.unsqueeze(obs, dim=0)

        return obs, act
    
    def env2np(self, obs, act=None):
        if act is not None:
            act = self.encodeAction(act=act, meanspeed=self.ag.speed_mean, nbins=self.numHDs)
            act[:,:,0] = act[:,:,0]/self.ag.speed_means
        act = np.array(act)

        obs = obs.clip(max=1)[None]
        obs = obs.reshape(obs.shape[:-2]+(-1,))

        return obs, act
    
    def pred2np(self, obs):
        self.obs_colors = ['black']
        facecolor = matplotlib.cm.get_cmap(self.env.object_colormap)
        for i in range(self.env.n_object_types):
            self.obs_colors.append(facecolor(i / (self.env.n_object_types - 1 + 1e-8)))
        
        obs = obs.detach().numpy().squeeze()

        img = []
        for t in range(obs.shape[0]):
            img.append(self.to_image(obs[t])[None,...])
        obs = np.concatenate(img, axis=0)
        return obs
    
    def to_image(self, obs):
        fig, ax = plt.subplots()
        # ax.set_ylim(top=0.5)

        n = self.env.n_object_types
        obs = obs.reshape(-1,n+1)

        for (i, coord) in enumerate(self.vision[0].manifold_coords_euclid):
            [x, y] = coord

            for obj in range(n+1):
                facecolor = list(matplotlib.colors.to_rgba(self.obs_colors[obj]))
                facecolor[-1] = max(0, min(1, obs[i,obj] / self.vision[int(bool(obj))].super.max_fr)*n/(n+obj)) # Make the colors increasingly transparent as the next overlays the previous
                circ = matplotlib.patches.Circle(
                    (-x+0.5, y),
                    radius=0.5 * self.vision[int(bool(obj))].spatial_resolution,
                    linewidth=0.5,
                    edgecolor="dimgrey",
                    facecolor=facecolor,
                    zorder=2.1,
                )
                ax.add_patch(circ)

        plt.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image_from_plot

    def getActSize(self):
        action = self.encodeAction(act=np.ones((2,3)), meanspeed=self.ag.speed_mean, nbins=self.numHDs)
        act_size = action.size(2)
        return act_size
    
    def getActType(self):
        return torch.float32

    def getObsSize(self):
        obs_size = self.vision[0].n + self.vision[1].n
        return obs_size
    
    def get_viewpoint(self, agent_pos, agent_dir):
        self.reset(pos=agent_pos, hd=agent_dir)

        walls = np.array(self.vision[0].history["firingrate"][0])
        objects = np.array(self.vision[1].history["firingrate"][0])
        n_neurons = walls.shape[0]
        objects = objects.reshape((n_neurons, -1), order='F')
        obs = np.concatenate((walls[...,None], objects), axis=-1)

        return self.env2pred(obs)[0].squeeze()
    
    def get_map_bins(self):
        minmax=(0, self.width,
                0, self.height)
        return self.width, self.height, minmax
    
    def load_state(self, state):
        self.set_agent_pos(state[:2])
        self.set_agent_dir(state[2:])
    
    def save_state(self, act, state):
        # TODO: check this
        return np.array([state['agent_pos'][-1], act[-1,1:]])
    
    def set_agent_pos(self, pos):
        self.ag.pos = pos
    
    def set_agent_dir(self, vel):
        self.ag.save_velocity = vel
    
    def show_state(self, t, fig, ax, **kwargs):
        start_t = self.ag.history["t"][t-1]
        end_t = self.ag.history["t"][t]
        self.ag.plot_trajectory(t_start=start_t, t_end=end_t, fig=fig, ax=ax,color="changing")
        self.vision[0].display_manifold(fig, ax, t=end_t)
        for obj in range(self.env.n_object_types):
            self.vision[1].display_manifold(fig, ax, t=end_t, object_type=obj)
    
    def show_state_traj(self, start, end, fig, ax, **kwargs):
        start_t = self.ag.history["t"][start]
        end_t = self.ag.history["t"][end]
        self.ag.plot_trajectory(t_start=start_t, t_end=end_t, fig=fig, ax=ax, color="changing")
        self.vision[0].display_manifold(fig, ax, t=end_t)
        for obj in range(self.env.n_object_types):
            self.vision[1].display_manifold(fig, ax, t=end_t, object_type=obj)
    
    def reset(self, pos=np.zeros(2), vel=None, seed=False, keep_state=False):

        self.ag.reset_history()
        self.vision[0].reset_history()
        self.vision[1].reset_history()
        
        if keep_state:
            vel = self.ag.vel
            pos = self.ag.pos

        if vel:
            self.ag.pos = pos
        else:
            vel = [0,0]

        self.ag.save_velocity = vel
        self.ag.save_to_history()
        self.vision[0].update()
        self.vision[1].update()


# TODO: colormapping as argument
class RiaBRemixColorsShell(RatInABoxShell):
    def __init__(self, env, act_enc, env_key, speed, thigmotaxis, HDbins):
        super().__init__(env, act_enc, env_key, speed, thigmotaxis, HDbins)

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
        remix[...,2] += obs[...,1]
        remix[...,0] += obs[...,2]
        remix[...,0] += obs[...,3]
        remix[...,1] += obs[...,3]
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
        remix[...,2] += obs[...,1]
        remix[...,0] += obs[...,2]
        remix[...,0] += obs[...,3]
        remix[...,1] += obs[...,3]
        obs = remix


        obs = obs.clip(max=1)
        obs = obs.reshape(obs.shape[:-2]+(-1,))[None]

        return obs, act
    
    def pred2np(self, obs):
        """
        Convert sequence of observations from pytorch format to image-filled np.array
        """
        
        obs = obs.detach().numpy().squeeze()

        img = []
        for t in range(obs.shape[0]):
            img.append(self.to_image(obs[t])[None,...])
        obs = np.concatenate(img, axis=0)
        return obs
    
    def to_image(self, obs):
        fig, ax = plt.subplots()


        obs = obs.reshape(-1,3)

        for (i, coord) in enumerate(self.vision[0].manifold_coords_euclid):
            [x, y] = coord
            facecolor = list(obs[i])
            facecolor.append(1.0)
            circ = matplotlib.patches.Circle(
                (-x+0.5, y),
                radius=0.5 * self.vision[0].spatial_resolution,
                linewidth=0.5,
                edgecolor="dimgrey",
                facecolor=facecolor,
                zorder=2.1,
            )
            ax.add_patch(circ)

        plt.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image_from_plot

    def getObsSize(self):
        obs_size = self.vision[0].n * 3
        return obs_size