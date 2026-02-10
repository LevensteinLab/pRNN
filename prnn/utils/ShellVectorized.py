import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from prnn.utils.ActionEncodings import *
from prnn.utils.Shell import actionOptions

HDmap = {0: 270,
         1: 180,
         2: 90,
         3: 0
        }


class ShellVectorized:
    def __init__(self, envs, act_enc, env_key):
        self.envs = envs
        self.name = env_key
        self.encodeAction = actionOptions[act_enc]
        self.dataLoader = None
        self.DL_iterator = None
        self.n_obs = 1 #default number of observation modalities
        self.num_envs = envs.num_envs

    @property
    def env(self):
        """Alias so that shell_env.env works (e.g. for conspecific check)."""
        return self.envs

    def addDataLoader(self, dataloader):
        self.dataLoader = dataloader
        self.DL_iterator = iter(self.dataLoader)

    def killIterator(self):
        iterator = self.DL_iterator
        self.DL_iterator = None
        return iterator

    def collectObservationSequence(self, agent, tsteps, obs_format='pred',
                                   includeRender=False, discretize=False,
                                   inv_x=False, inv_y=False, seed=None,
                                   dataloader=False, reset=True,
                                   save_env=False, device='cpu',
                                   compute_loss=False, render_highlight=True):
        """
        Use an agent (action generator) to collect an observation/action sequence
        in tensor format for feeding to the predictive net.
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
            act = data[self.n_obs]
            state = None
            render = None
            obs = [*data[:self.n_obs]]
            if len(obs) == 1:
                obs = obs[0]
            if self.dataLoader.dataset.raw:
                obs, act = self.env2pred(obs, act, state=state,
                                         device=device,
                                         compute_loss=compute_loss,
                                         from_raw=True)
        else:
            # Pass AsyncVectorEnv and self (as shell_env) to the agent
            obs, act, state, render = agent.getObservations(self.envs, self, tsteps,
                                                    includeRender=includeRender,
                                                    reset=reset,
                                                    render_highlight=render_highlight)

            if obs_format == 'pred':
                obs, act = self.env2pred(obs, act, state=state,
                                            device=device,
                                            compute_loss=compute_loss)
            elif obs_format == 'npgrid':
                nps = self.env2np(obs, act, state=state,
                                    device=device, save_env=save_env)
                obs, act = nps[0], nps[1]

                if save_env:
                    obs_env = nps[2]
        if save_env:
            return obs, act, state, render, obs_env
        else:
            return obs, act, state, render

    # Backward-compatible alias used by predictiveNetVectorized
    def collectObservationSequenceVectorized(self, envs, agent, tsteps,
                                             obs_format='pred',
                                             includeRender=False,
                                             discretize=False,
                                             inv_x=False, inv_y=False,
                                             seed=None, dataloader=False,
                                             device='cpu', compute_loss=False):
        """Backward compatible wrapper. The envs argument is ignored since
        self.envs already holds the AsyncVectorEnv."""
        return self.collectObservationSequence(
            agent, tsteps, obs_format=obs_format,
            includeRender=includeRender, discretize=discretize,
            inv_x=inv_x, inv_y=inv_y, seed=seed,
            dataloader=dataloader, device=device,
            compute_loss=compute_loss)

    def dir2deg(self, dir):
        raise NotImplementedError('Environment-specific "Shell" class should be used')

    def env2pred(self, **kwargs):
        raise NotImplementedError('Environment-specific "Shell" class should be used')

    def env2np(self, **kwargs):
        raise NotImplementedError('Environment-specific "Shell" class should be used')

    def pred2np(self, **kwargs):
        raise NotImplementedError('Environment-specific "Shell" class should be used')

    def get_hd(self):
        raise NotImplementedError('Environment-specific "Shell" class should be used')

    def getActSize(self):
        raise NotImplementedError('Environment-specific "Shell" class should be used')

    def getActType(self):
        raise NotImplementedError('Environment-specific "Shell" class should be used')

    def getObsSize(self):
        raise NotImplementedError('Environment-specific "Shell" class should be used')

    def get_map_bins(self):
        raise NotImplementedError('Environment-specific "Shell" class should be used')

    def reset(self, **kwargs):
        raise NotImplementedError('Environment-specific "Shell" class should be used')


class GymMinigridShellVectorized(ShellVectorized):
    def __init__(self, envs, act_enc, env_key, **kwargs):
        super().__init__(envs, act_enc, env_key)
        self.obs_shape = self.envs.single_observation_space['image'].shape
        self.numHDs = 4
        # Get grid dimensions from sub-environments via AsyncVectorEnv.call
        heights = self.envs.call('get_wrapper_attr', 'height')
        widths = self.envs.call('get_wrapper_attr', 'width')
        self.height = heights[0]
        self.width = widths[0]
        self.continuous = False
        self.max_dist = False
        self.hd_trans = np.array([-1,1,0,0])
        self.start_pos = 1

    @property
    def action_space(self):
        return self.envs.single_action_space

    def dir2deg(self, dir):
        return HDmap[dir]

    def env2pred(self, obs, act=None, state=None, hd_from='obs', actoffset=0,
                 device='cpu', compute_loss=False, from_raw=False, **kwargs):
        """
        Convert batched observations and actions to tensor format.

        Args:
            obs: Object array of length (tsteps+1,). Each element is a dict
                 with 'image' (num_envs, H, W, C) and 'direction' (num_envs,).
            act: Action array of shape (tsteps, num_envs).
            state: State dictionary with agent_pos and agent_dir.

        Returns:
            obs_tensor: Tensor of shape (num_envs, tsteps+1, obs_size)
            act: Tensor of shape (num_envs, tsteps, act_enc_size)
        """
        n_steps = len(obs)

        # Extract images: (num_envs, tsteps+1, H, W, C)
        obs_images = np.stack([obs[t]['image'] for t in range(n_steps)], axis=1)

        # Extract head directions: (num_envs, tsteps+1)
        hd = None
        if hd_from == 'obs':
            hd = np.stack([obs[t]['direction'] for t in range(n_steps)], axis=1)

        # Flatten spatial dims: (num_envs, tsteps+1, H*W*C)
        obs_flat = obs_images.reshape(self.num_envs, n_steps, -1)

        # Normalize to [0, 1]
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, requires_grad=False)
        obs_tensor = obs_tensor / 255.0

        if act is not None:
            # act shape: (tsteps, num_envs) -> (num_envs, tsteps)
            if act.ndim == 2:
                act_transposed = act.T
            else:
                act_transposed = act.reshape(-1, self.num_envs).T

            act_encoded_list = []
            for env_idx in range(self.num_envs):
                hd_env = hd[env_idx] if hd is not None else np.zeros(n_steps)

                act_enc = self.encodeAction(
                    act=act_transposed[env_idx],
                    obs=hd_env,
                    numSuppObs=self.numHDs,
                    numActs=self.action_space.n
                )
                act_encoded_list.append(act_enc)

            # Stack: list of (1, tsteps, act_size) -> (num_envs, tsteps, act_size)
            act = torch.cat(act_encoded_list, dim=0)

        return obs_tensor, act

    def env2np(self, obs, act=None, state=None, device='cpu', save_env=False, **kwargs):
        """Convert batched observations and actions to numpy format."""
        n_steps = len(obs)

        obs_images = np.stack([obs[t]['image'] for t in range(n_steps)], axis=1)
        hd = np.stack([obs[t]['direction'] for t in range(n_steps)], axis=1)

        obs_flat = obs_images.reshape(self.num_envs, n_steps, -1)
        obs_np = obs_flat / 255.0

        if act is not None:
            if act.ndim == 2:
                act_transposed = act.T
            else:
                act_transposed = act.reshape(-1, self.num_envs).T

            act_encoded_list = []
            for env_idx in range(self.num_envs):
                act_enc = self.encodeAction(
                    act=act_transposed[env_idx],
                    obs=hd[env_idx],
                    numSuppObs=self.numHDs,
                    numActs=self.action_space.n
                ).numpy()
                act_encoded_list.append(act_enc)
            act = np.concatenate(act_encoded_list, axis=0)

        return obs_np, act

    def pred2np(self, obs, whichPhase=0, timesteps=None):
        obs = obs.detach().numpy()
        if timesteps:
            obs = obs[:,timesteps,...]
        obs = np.reshape(obs[whichPhase,:,:],(-1,)+self.obs_shape)
        return obs

    def get_agent_pos(self):
        """Get agent position from the first sub-environment."""
        positions = self.envs.call('get_wrapper_attr', 'agent_pos')
        return positions[0]

    def get_agent_dir(self):
        """Get agent direction from the first sub-environment."""
        directions = self.envs.call('get_wrapper_attr', 'agent_dir')
        return directions[0]

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

    def show_state(self, render, t, **kwargs):
        frame = render[t]
        # AsyncVectorEnv.call("render") returns a list of frames (one per env)
        if isinstance(frame, (list, tuple)):
            frame = frame[0]
        
        if frame is not None:
            plt.imshow(frame)

    def show_state_traj(self, start, end, state, render, **kwargs):
        trajectory_ts = np.arange(start, end+1)
        if render is not None:
            frame = render[trajectory_ts[-1]]
            # AsyncVectorEnv.call("render") returns a list of frames (one per env)
            if isinstance(frame, (list, tuple)):
                frame = frame[0]
                
            if frame is not None:
                plt.imshow(frame)
        plt.plot((state['agent_pos'][trajectory_ts,0]+0.5)*512/16,
                    (state['agent_pos'][trajectory_ts,1]+0.5)*512/16,color='r')

    def step(self, action):
        return self.envs.step(action)

    def render(self, highlight=True, mode=None):
        return self.envs.call('render')

    def reset(self, seed=False):
        if seed:
            return self.envs.reset(seed=seed)
        return self.envs.reset()

    def close(self):
        self.envs.close()


class FaramaMinigridShellVectorized(GymMinigridShellVectorized):
    def __init__(self, envs, act_enc, env_key, **kwargs):
        super().__init__(envs, act_enc, env_key)

    def render(self, highlight=True, mode='human'):
        return self.envs.call('get_frame')

    def reset(self, seed=False):
        if seed:
            return self.envs.reset(seed=seed)[0]
        else:
            return self.envs.reset()[0]

    @property
    def observation_space(self):
        return self.envs.single_observation_space

    def pre_save(self):
        """Remove the AsyncVectorEnv before pickling (it contains
        unpicklable closures and subprocess handles)."""
        out = self.envs
        self.envs = None
        return out

    def post_save(self, envs):
        """Restore the AsyncVectorEnv after pickling."""
        self.envs = envs
