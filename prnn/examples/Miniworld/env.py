import math

from abc import ABC
from typing import Optional

import numpy as np
from gymnasium import spaces
from miniworld.entity import Agent, MeshEnt, Entity
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

class Goal(Entity):
    def __init__(self, radius=1):
        super().__init__()
        self.radius = radius
    def render(self):
        pass

class Rat(Agent):
    def __init__(self):
        super().__init__()
        self.cam_height = 0.75
        self.radius = 0.2
        self.height = 0.9
        self.cam_fwd_disp = 0

    def randomize(self, *args):
        pass


class LRoom(MiniWorldEnv):

    def __init__(self, continuous=True, size=10, **kwargs):
        self.size = size
        self.continuous = continuous
        self.target=False
        super().__init__(self, **kwargs)

        if continuous:
            self.action_space = spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), shape=(2,))

        else:
            # Allow only the movement actions
            self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        x_crest = self.size*0.625
        y_crest = self.size*0.5
        room1 = self.add_rect_room(
            min_x=0,
            max_x=x_crest+1,
            min_z=0,
            max_z=y_crest+1,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )
        room2 = self.add_rect_room(
            min_x=x_crest+1,
            max_x=self.size+1,
            min_z=0,
            max_z=y_crest+1,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )
        room3 = self.add_rect_room(
            min_x=0,
            max_x=x_crest+1,
            min_z=y_crest+1,
            max_z=self.size+1,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )
        self.connect_rooms(room1, room2, min_z=room1.min_z, max_z=room1.max_z)
        self.connect_rooms(room1, room3, min_x=room1.min_x, max_x=room1.max_x)

        # colorlist = list(COLOR_NAMES)

        self.place_entity(
            MeshEnt(mesh_name="building", height=20),
            pos=np.array([40, 0, 35]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="barrel", height=25),
            pos=np.array([-40, 0, 20]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="cone", height=25),
            pos=np.array([-30, 0, -20]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="duckie", height=25),
            pos=np.array([0, 0, 35]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="tree", height=25),
            pos=np.array([0, 0, -35]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="potion", height=25),
            pos=np.array([40, 0, -35]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="office_chair", height=25),
            pos=np.array([40, 0, 12]),
            dir=-math.pi,
        )

        if self.target:
            self.goal = self.place_entity(Goal(), pos=np.array([5, 0, 5]))

        self.place_agent()
        if (self.agent.pos[0] <= 0.7) or (self.agent.pos[2] <= 0.7):
            self.agent.pos = np.array([0.7, 0, 0.7])

    def turn_agent_cont(self, turn_angle):
        """
        Turn the agent left or right
        """

        orig_dir = self.agent.dir

        self.agent.dir += turn_angle

        return True

    def move_agent_cont(self, speed):
        """
        Move the agent forward
        """

        next_pos = self.agent.pos + self.agent.dir_vec * speed

        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False

        self.agent.pos = next_pos

        return True

    def reset(self, *, seed=None, options=None):
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """
        super().reset(seed=seed)

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Rat()

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        self.wall_segs = []

        # Generate the world
        self._gen_world()

        # Check if domain randomization is enabled or not
        rand = self.np_random if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(rand, self, ["sky_color", "light_pos", "light_color", "light_ambient"])

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max("forward_step")

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min(r.min_x for r in self.rooms)
        self.max_x = max(r.max_x for r in self.rooms)
        self.min_z = min(r.min_z for r in self.rooms)
        self.max_z = max(r.max_z for r in self.rooms)

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        obs = self.render_obs()

        # Return first observation
        return obs, {}

    def step(self, action):
        """
        Perform one action and update the simulation
        """

        self.step_count += 1

        if self.continuous:
            self.turn_agent_cont(action[1])
            self.move_agent_cont(action[0])
        else:
            rand = self.np_random if self.domain_rand else None
            fwd_step = self.params.sample(rand, "forward_step")
            fwd_drift = self.params.sample(rand, "forward_drift")
            turn_step = self.params.sample(rand, "turn_step")

            if action == self.actions.move_forward:
                self.move_agent(fwd_step, fwd_drift)

            elif action == self.actions.move_back:
                self.move_agent(-fwd_step, fwd_drift)

            elif action == self.actions.turn_left:
                self.turn_agent(turn_step)

            elif action == self.actions.turn_right:
                self.turn_agent(-turn_step)

        # # If the maximum time step count is reached
        # if self.step_count >= self.max_episode_steps:
        #     termination = True
        #     truncation = False
        #     reward = 0
        #     # return obs, reward, termination, truncation, {}
        # # If the goal is reached
        # elif self.target and self.near(self.goal):
        #     self.step_count = 0
        #     self.place_agent()
        #     reward = self._reward()+1
        #     termination = False
        #     truncation = False

        # elif self.target:
        #     reward = np.exp(-np.linalg.norm(self.goal.pos - self.agent.pos)/10)
        #     termination = False
        #     truncation = False


        # else:
        #     reward = 0
        #     termination = False
        #     truncation = False

        reward = 0
        termination = False
        truncation = False

        # Generate the current camera image
        obs = self.render_obs()

        return obs, reward, termination, truncation, {}