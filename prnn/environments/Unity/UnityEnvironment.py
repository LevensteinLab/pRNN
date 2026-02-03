"""
Gymnasium-compatible wrapper for any Unity ML-Agents environment.

Automatically discovers the first behavior, builds matching Gymnasium
action / observation spaces, and exposes the standard Gymnasium API.

Usage:
    from prnn.environments.Unity.UnityEnvironment import UnityEnv

    env = UnityEnv(app_path="/path/to/your/build.app")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

# ---------------------------------------------------------------------------
# Monkey-patch mlagents_envs to handle HWC <-> CHW mismatch for compressed
# visual observations.  The Unity build may send compressed (PNG) images that
# decompress to HWC, while mlagents_envs 0.30.0 expects CHW.  Rather than
# editing the installed package, we patch the single function here so the fix
# travels with the project.
# ---------------------------------------------------------------------------
import mlagents_envs.rpc_utils as _rpc_utils
from mlagents_envs.timers import timed as _timed

@_timed
def _patched_observation_to_np_array(obs, expected_shape=None):
    if expected_shape is not None:
        if list(obs.shape) != list(expected_shape):
            raise _rpc_utils.UnityObservationException(
                f"Observation did not have the expected shape - "
                f"got {obs.shape} but expected {expected_shape}"
            )
    if obs.compression_type == _rpc_utils.COMPRESSION_TYPE_NONE:
        img = np.array(obs.float_data.data, dtype=np.float32)
        img = np.reshape(img, obs.shape)
        return img
    else:
        expected_channels = obs.shape[2] if len(obs.shape) == 3 else obs.shape[-1]
        img = _rpc_utils.process_pixels(
            obs.compressed_data, expected_channels,
            list(obs.compressed_channel_mapping),
        )
        if list(obs.shape) != list(img.shape):
            # Handle HWC -> CHW transpose when shapes match after transposing
            if (len(img.shape) == 3
                    and list(obs.shape) == [img.shape[2], img.shape[0], img.shape[1]]):
                img = np.transpose(img, (2, 0, 1))
            else:
                raise _rpc_utils.UnityObservationException(
                    f"Decompressed observation did not have the expected shape - "
                    f"decompressed had {img.shape} but expected {obs.shape}"
                )
        return img

_rpc_utils._observation_to_np_array = _patched_observation_to_np_array

class UnityEnv(gymnasium.Env):
    """Single-agent Gymnasium wrapper around a Unity ML-Agents environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        app_path: str,
        worker_id: int = 0,
        no_graphics: bool = False,
        time_scale: float = 20.0,
        max_steps: int = 0,
        seed: int = 0,
        render_mode: str | None = "rgb_array",
    ):
        super().__init__()
        self.render_mode = render_mode
        self._max_steps = max_steps
        self._step_count = 0
        self._last_visual_obs = None

        # Engine configuration side channel
        self._engine_channel = EngineConfigurationChannel()
        self._engine_channel.set_configuration_parameters(time_scale=time_scale)

        # Launch Unity environment
        self._unity_env = UnityEnvironment(
            file_name=app_path,
            worker_id=worker_id,
            no_graphics=no_graphics,
            seed=seed,
            side_channels=[self._engine_channel],
        )

        # First reset to discover behavior specs
        self._unity_env.reset()
        self._behavior_name = list(self._unity_env.behavior_specs)[0]
        spec = self._unity_env.behavior_specs[self._behavior_name]

        # Build action space
        action_spec = spec.action_spec
        if action_spec.is_discrete():
            branches = action_spec.discrete_branches
            if len(branches) == 1:
                self.action_space = spaces.Discrete(branches[0])
            else:
                self.action_space = spaces.MultiDiscrete(list(branches))
        elif action_spec.is_continuous():
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(action_spec.continuous_size,), dtype=np.float32
            )
        else:
            # Hybrid: use Dict space
            d = {}
            d["continuous"] = spaces.Box(
                low=-1.0, high=1.0, shape=(action_spec.continuous_size,), dtype=np.float32
            )
            branches = action_spec.discrete_branches
            if len(branches) == 1:
                d["discrete"] = spaces.Discrete(branches[0])
            else:
                d["discrete"] = spaces.MultiDiscrete(list(branches))
            self.action_space = spaces.Dict(d)

        # Classify observations
        obs_specs = spec.observation_specs
        visual_specs = [s for s in obs_specs if len(s.shape) == 3]
        vector_specs = [s for s in obs_specs if len(s.shape) == 1]

        self._visual_indices = [i for i, s in enumerate(obs_specs) if len(s.shape) == 3]
        self._vector_indices = [i for i, s in enumerate(obs_specs) if len(s.shape) == 1]

        # Build observation space
        # Unity sends visual obs in CHW format; we expose HWC to gymnasium
        if visual_specs and vector_specs:
            vec_size = sum(s.shape[0] for s in vector_specs)
            c, h, w = visual_specs[0].shape
            self.observation_space = spaces.Dict({
                "visual": spaces.Box(
                    low=0, high=255, shape=(h, w, c), dtype=np.uint8
                ),
                "vector": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(vec_size,), dtype=np.float32
                ),
            })
            self._obs_mode = "dict"
        elif visual_specs:
            c, h, w = visual_specs[0].shape
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(h, w, c), dtype=np.uint8
            )
            self._obs_mode = "visual"
        else:
            vec_size = sum(s.shape[0] for s in vector_specs)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(vec_size,), dtype=np.float32
            )
            self._obs_mode = "vector"

        # Grab initial agent id
        decision_steps, _ = self._unity_env.get_steps(self._behavior_name)
        self._agent_id = decision_steps.agent_id[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_obs(self, all_obs, agent_idx: int):
        """Extract observation for a single agent from the batched obs list.

        Visual observations arrive in CHW float32 [0,1] format from Unity
        and are converted to HWC uint8 [0,255] for gymnasium.
        """
        if self._obs_mode == "visual":
            raw = all_obs[self._visual_indices[0]][agent_idx]  # (C, H, W)
            img = (np.clip(raw, 0.0, 1.0) * 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            self._last_visual_obs = img
            return img
        elif self._obs_mode == "vector":
            parts = [all_obs[i][agent_idx] for i in self._vector_indices]
            return np.concatenate(parts).astype(np.float32)
        else:
            raw = all_obs[self._visual_indices[0]][agent_idx]  # (C, H, W)
            img = (np.clip(raw, 0.0, 1.0) * 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            self._last_visual_obs = img
            parts = [all_obs[i][agent_idx] for i in self._vector_indices]
            return {
                "visual": img,
                "vector": np.concatenate(parts).astype(np.float32),
            }

    def _build_action_tuple(self, action, n_agents: int = 1):
        """Convert a gymnasium action into an mlagents ActionTuple."""
        spec = self._unity_env.behavior_specs[self._behavior_name].action_spec

        if isinstance(action, dict):
            # Hybrid
            cont = np.array(action["continuous"], dtype=np.float32).reshape(1, -1)
            disc = np.array(action["discrete"], dtype=np.int32).reshape(1, -1)
            if n_agents > 1:
                cont = np.tile(cont, (n_agents, 1))
                disc = np.tile(disc, (n_agents, 1))
            return ActionTuple(continuous=cont, discrete=disc)

        if spec.is_discrete():
            a = np.array(action, dtype=np.int32)
            if a.ndim == 0:
                a = a.reshape(1, 1)
            elif a.ndim == 1:
                a = a.reshape(1, -1)
            if n_agents > 1:
                a = np.tile(a, (n_agents, 1))
            return ActionTuple(discrete=a)

        if spec.is_continuous():
            a = np.array(action, dtype=np.float32)
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if n_agents > 1:
                a = np.tile(a, (n_agents, 1))
            return ActionTuple(continuous=a)

    def _random_action_tuple(self, n_agents: int):
        """Generate a random ActionTuple for n_agents."""
        spec = self._unity_env.behavior_specs[self._behavior_name].action_spec
        return spec.random_action(n_agents)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._step_count = 0

        self._unity_env.reset()
        decision_steps, _ = self._unity_env.get_steps(self._behavior_name)

        if len(decision_steps) == 0:
            # Edge case: do an extra step to get agents
            self._unity_env.step()
            decision_steps, _ = self._unity_env.get_steps(self._behavior_name)

        self._agent_id = decision_steps.agent_id[0]
        idx = np.where(decision_steps.agent_id == self._agent_id)[0][0]
        obs = self._extract_obs(decision_steps.obs, idx)
        return obs, {}

    def step(self, action):
        decision_steps, _ = self._unity_env.get_steps(self._behavior_name)
        n_agents = len(decision_steps)

        if n_agents > 0:
            # Build actions: our action for our agent, random for others
            if n_agents == 1:
                action_tuple = self._build_action_tuple(action, 1)
            else:
                action_tuple = self._random_action_tuple(n_agents)
                # Find our agent's index in decision_steps and overwrite
                our_idx = np.where(decision_steps.agent_id == self._agent_id)[0]
                if len(our_idx) > 0:
                    our_idx = our_idx[0]
                    single = self._build_action_tuple(action, 1)
                    if action_tuple.discrete is not None and single.discrete is not None:
                        action_tuple.discrete[our_idx] = single.discrete[0]
                    if action_tuple.continuous is not None and single.continuous is not None:
                        action_tuple.continuous[our_idx] = single.continuous[0]

            self._unity_env.set_actions(self._behavior_name, action_tuple)

        self._unity_env.step()
        self._step_count += 1

        decision_steps, terminal_steps = self._unity_env.get_steps(self._behavior_name)

        terminated = False
        truncated = False
        reward = 0.0

        # Check if our agent terminated
        if self._agent_id in terminal_steps:
            ts = terminal_steps[self._agent_id]
            reward = float(ts.reward)
            obs = self._extract_obs(
                [o[np.newaxis] for o in ts.obs], 0
            )
            terminated = not ts.interrupted
            truncated = bool(ts.interrupted)
        elif self._agent_id in decision_steps:
            ds = decision_steps[self._agent_id]
            reward = float(ds.reward)
            obs = self._extract_obs(
                [o[np.newaxis] for o in ds.obs], 0
            )
        else:
            # Agent not found — might have respawned with a new id
            if len(terminal_steps) > 0:
                aid = terminal_steps.agent_id[0]
                ts = terminal_steps[aid]
                reward = float(ts.reward)
                obs = self._extract_obs(
                    [o[np.newaxis] for o in ts.obs], 0
                )
                terminated = True
            elif len(decision_steps) > 0:
                self._agent_id = decision_steps.agent_id[0]
                idx = 0
                reward = float(decision_steps.reward[idx])
                obs = self._extract_obs(decision_steps.obs, idx)
            else:
                # No agents at all — return zeros
                obs = self.observation_space.sample()
                terminated = True

        # Optional max-step truncation
        if self._max_steps > 0 and self._step_count >= self._max_steps and not terminated:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self._last_visual_obs is not None:
            return self._last_visual_obs
        return None

    def close(self):
        self._unity_env.close()
