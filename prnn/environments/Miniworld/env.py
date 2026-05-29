import math

import numpy as np

from abc import ABC
from typing import Optional, Tuple
from gymnasium import spaces, utils
from gymnasium.core import ObsType

from pyglet.gl import (
    GL_TRIANGLES,
    glBegin,
    glColor3f,
    glEnd,
    glVertex3f,
)

from miniworld.entity import Agent, MeshEnt, Entity, Box
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

class Goal(Entity):
    def __init__(self, radius=1):
        super().__init__()
        self.radius = radius
    def render(self):
        pass

class Rat(Agent):
    def __init__(self, radius=0.4):
        super().__init__()
        self.cam_height = 0.75
        self.radius = radius
        self.height = 0.9
        self.cam_fwd_disp = 0

    def render(self):
        """
        Draw the agent
        """

        p = self.pos + np.array([0, 1, 0]) * self.height
        # Not dependent on self.radius
        dv = self.dir_vec * 0.5
        rv = self.right_vec * 0.5

        p0 = p + dv
        p1 = p + 0.75 * (rv - dv)
        p2 = p + 0.75 * (-rv - dv)

        glColor3f(1, 0, 0)
        glBegin(GL_TRIANGLES)
        glVertex3f(*p0)
        glVertex3f(*p2)
        glVertex3f(*p1)
        glEnd()

    def randomize(self, *args):
        pass


class LRoom(MiniWorldEnv):

    def __init__(self, continuous=True, size=10,
                 walls=("brick_wall","brick_wall","brick_wall"),
                 floors=("asphalt","asphalt","asphalt"),
                 sheep=False, **kwargs):
        self.size = np.array([size, size])
        self.continuous = continuous
        self.target=False
        self.walls = walls
        self.floors = floors
        self.sheep = sheep
        super().__init__(self, **kwargs)

        if continuous:
            self.action_space = spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), shape=(2,))

        else:
            # Allow only the movement actions
            self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        x_crest = self.size[0]*0.625
        y_crest = self.size[1]*0.5
        room1 = self.add_rect_room(
            min_x=0,
            max_x=x_crest,
            min_z=0,
            max_z=y_crest,
            wall_tex=self.walls[0],
            floor_tex=self.floors[0],
            no_ceiling=True,
        )
        room2 = self.add_rect_room(
            min_x=x_crest,
            max_x=self.size[0],
            min_z=0,
            max_z=y_crest,
            wall_tex=self.walls[1],
            floor_tex=self.floors[1],
            no_ceiling=True,
        )
        room3 = self.add_rect_room(
            min_x=0,
            max_x=x_crest,
            min_z=y_crest,
            max_z=self.size[1],
            wall_tex=self.walls[2],
            floor_tex=self.floors[2],
            no_ceiling=True,
        )
        self.connect_rooms(room1, room2, min_z=room1.min_z, max_z=room1.max_z)
        self.connect_rooms(room1, room3, min_x=room1.min_x, max_x=room1.max_x)

        # colorlist = list(COLOR_NAMES)
        if self.sheep:
            self.place_entity(
                MeshEnt(mesh_name="sheep", height=25),
                pos=np.array([40, 0, 35]),
                dir=-math.pi,
            )
        else:
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
        self.agent = Rat(radius=0.1)

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
            moved = self.move_agent_cont(action[0])
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

        reward = 0
        termination = False
        truncation = False

        # Generate the current camera image
        obs = self.render_obs()

        return obs, reward, termination, truncation, {"moved": moved if self.continuous else None}
    

class Mazest(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Maze environment in which the agent has to reach a center of target lava room and avoid other lava room.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing an RGB image of what the agents see.

    ## Rewards

    +(1 - 0.2 * (step_count / max_episode_steps)) when target box reached, -0.5 when false target box reached, and zero otherwise.

    ## Arguments

    ```python
    env = gymnasium.make("MiniWorld-Mazest-v0")
    ```
    """

    def __init__(
        self, num_rows=5, num_cols=5, room_size=3, max_episode_steps=None,
        continuous=True, **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25
        self.size = np.array([num_cols * (room_size + self.gap_size) - self.gap_size,
                              num_rows * (room_size + self.gap_size) - self.gap_size])
        self.continuous = continuous

        self.env_seed = 3042
        self.regenerate=True
        
        # Initialize layout storage
        self.room_layouts = None
        self.room_connections = None
        self.lava_room_positions = None

        MiniWorldEnv.__init__(
            self,
            max_episode_steps=max_episode_steps or num_rows * num_cols * 24,
            **kwargs,
        )
        utils.EzPickle.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _generate_layout(self):
        """
        Generate the layout of the maze: decide on textures for each room
        and which rooms should be connected. This is called only when
        regenerate is True.
        """
        # Define available textures
        textures = [
            "brick_wall", "marble", 
            "cinder_blocks", "drywall","wood", "grass",
            "marble", "slime", "rock", "water"
        ]
        
        # Store room textures: room_layouts[j][i] = (wall_tex, floor_tex, ceil_tex)
        self.room_layouts = []
        
        # For each row
        for j in range(self.num_rows):
            row = []
            # For each column
            for i in range(self.num_cols):
                # Choose random wall and floor textures for this room
                wall_tex = textures[self.np_random.integers(0, len(textures))]
                floor_tex = textures[self.np_random.integers(0, len(textures))]
                ceil_tex = textures[self.np_random.integers(0, len(textures))]
                row.append((wall_tex, floor_tex, ceil_tex))
            self.room_layouts.append(row)
        
        # Generate maze connections using recursive backtracking
        visited = set()
        self.room_connections = []  # List of (i1, j1, i2, j2, direction)
        
        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """
            visited.add((i, j))
            
            # Reorder the neighbors to visit in a random order
            orders = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            assert 4 <= len(orders)
            neighbors = []
            
            while len(neighbors) < 4:
                elem = orders[self.np_random.choice(len(orders))]
                orders.remove(elem)
                neighbors.append(elem)
            
            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj
                
                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue
                
                if (ni, nj) in visited:
                    continue
                
                # Store connection
                if di == 0:
                    direction = 'horizontal'
                else:  # dj == 0
                    direction = 'vertical'
                
                self.room_connections.append((i, j, ni, nj, direction))
                visit(ni, nj)
        
        # Generate the maze starting from the top-left corner
        visit(0, 0)
        
        # Determine which rooms should have lava textures
        # We need to identify rooms with identical wall configurations
        # We'll store positions and update textures after world generation
        self.lava_room_positions = None  # Will be set in _gen_world

    def _gen_world(self):
        """
        Build the actual world using the stored layout.
        This is called on every reset.
        """
        # Generate layout if needed
        if self.regenerate or self.room_layouts is None:
            self._generate_layout()
            self.regenerate = False
        
        rows = []
        
        # Build rooms using stored layout
        for j in range(self.num_rows):
            row = []
            for i in range(self.num_cols):
                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size
                
                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size
                
                # Get textures from stored layout
                wall_tex, floor_tex, ceil_tex = self.room_layouts[j][i]
                
                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex=wall_tex,
                    floor_tex=floor_tex,
                    ceil_tex=ceil_tex,
                )
                row.append(room)
            rows.append(row)
        
        # Connect rooms based on stored connections
        for i1, j1, i2, j2, direction in self.room_connections:
            room1 = rows[j1][i1]
            room2 = rows[j2][i2]
            
            if direction == 'horizontal':
                self.connect_rooms(
                    room1, room2, min_x=room1.min_x, max_x=room1.max_x
                )
            elif direction == 'vertical':
                self.connect_rooms(
                    room1, room2, min_z=room1.min_z, max_z=room1.max_z
                )

        # Identify rooms with identical wall configurations
        # Create a signature for each room based on which walls are solid (no portals)
        room_configs = {}  # Maps configuration signature to list of rooms
        
        rows.reverse() # otherwise top-left room would often be chosen
        for row in rows:
            for room in row:
                # Create a tuple indicating which walls are solid (True) or have openings (False)
                # Room.num_walls is typically 4 for rectangular rooms
                wall_signature = tuple(len(room.portals[i]) == 0 for i in range(room.num_walls))
                
                if wall_signature not in room_configs:
                    room_configs[wall_signature] = []
                room_configs[wall_signature].append(room)
        
        # Find two rooms with identical wall configuration (preferably 3 walls)
        lava_rooms = None
        candidates = []
        # First try to find rooms with exactly 3 solid walls
        for config, rooms in room_configs.items():
            if sum(config) == 3 and len(rooms) >= 2:
                lava_rooms = rooms[:2]
                if self.regenerate:
                    print(f"Found two rooms with 3 walls configuration: {config}")
                break
        
        # If no rooms with 3 walls found, use any matching pair
        if lava_rooms is None:
            for config, rooms in room_configs.items():
                if len(rooms) >= 2:
                    lava_rooms = rooms[:2]
                    if self.regenerate:
                        print(f"Found two rooms with configuration: {config} ({sum(config)} walls)")
                    break
        
        # Change textures to lava for the identified rooms
        if lava_rooms:
            self.boxes = []
            for room in lava_rooms:
                room.wall_tex_name = "lava"
                room.floor_tex_name = "lava"
                room.ceil_tex_name = "lava"
                if self.regenerate:
                    print(f"Set lava textures for room at ({room.mid_x:.1f}, {room.mid_z:.1f})")

            for room in lava_rooms:
                self.boxes.append(self.place_entity(Box(color="red", size=0),
                                                    pos=(room.mid_x, 0, room.mid_z),
                                                    dir=0))

        self.place_agent()

    def step(self, action):

        self.step_count += 1

        if self.continuous:
            self.turn_agent_cont(action[1])
            moved = self.move_agent_cont(action[0])
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

        reward = 0
        termination = False
        truncation = False

        # Generate the current camera image
        obs = self.render_obs()

        if self.near(self.boxes[0]):
            reward += self._reward()
            termination = True
        elif self.near(self.boxes[1]):
            reward = - self._reward()/2
            termination = True

        return obs, reward, termination, truncation, {"moved": moved if self.continuous else None}

    def turn_agent_cont(self, turn_angle):
        """
        Turn the agent left or right
        """

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
        if options and options.get('regenerate'):
            assert seed is not None, "Seed must be provided when regenerate is True"
            self.regenerate = True
            self.env_seed = seed
        if self.regenerate:
            super().reset(seed=self.env_seed)
        else:
            super().reset(seed=seed)

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Rat(radius=0.1)

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
        
        self.sky_color = np.zeros(3)

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        obs = self.render_obs()

        # Compute colormap values for the maze layout (for potential use in observations or rewards)
        if self.regenerate:
            self._compute_colormap_traversal()

        # Set regenerate to False so that layout is not changed on next resets
        self.regenerate = False

        # Return first observation
        return obs, {}

    def near(self, ent0, ent1=None):
        """
        Test if the two entities are near each other.
        Used for "go to" or "put next" type tasks
        """

        if ent1 is None:
            ent1 = self.agent

        dist = np.linalg.norm(ent0.pos - ent1.pos)
        return dist < ent0.radius + ent1.radius + 0.5

    def colormap(self, x, y):
        """
        Convert position(s) to scalar value(s) representing linearized maze traversal.
        
        The scalar follows a deterministic DFS room ordering:
        - First room maps to 0.0
        - Last room maps to 1.0
        - At branching points, branch subtrees occupy contiguous intervals in DFS order
        
        Args:
            x: X coordinate(s) - scalar or array-like
            y: Y coordinate(s) (z in miniworld) - scalar or array-like
        
        Returns:
            float or ndarray: Scalar value(s) between 0.0 and 1.0
        """
        # Compute colormap values if not already done
        if not hasattr(self, '_colormap_values'):
            self._compute_colormap_traversal()
        
        # Check if inputs are scalars
        is_scalar = np.isscalar(x) and np.isscalar(y)
        
        # Convert to numpy arrays for vectorization
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        
        # Ensure same shape
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
        
        # Vectorized operation
        result = np.zeros(x.shape, dtype=float)
        for idx in np.ndindex(x.shape):
            room_coords = self._find_room_at_position(x[idx], y[idx])
            if room_coords is not None:
                result[idx] = self._colormap_values.get(room_coords, 0.0)
        
        # Return scalar if input was scalar
        if is_scalar:
            return float(result.item())
        return result

    def _find_room_at_position(self, x, y):
        """
        Find the (i, j) grid coordinates of the room containing position (x, y).
        
        Args:
            x: X coordinate
            y: Y coordinate (z in miniworld)
        
        Returns:
            tuple: (i, j) room coordinates, or None if outside maze bounds
        """
        for j in range(self.num_rows):
            for i in range(self.num_cols):
                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size + self.gap_size
                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size + self.gap_size
                
                if min_x <= x <= max_x and min_z <= y <= max_z:
                    return (i, j)
        return None

    def _compute_colormap_traversal(self):
        """
        Compute scalar colormapping values for each room using DFS linearization.

        The root room is assigned 0.0 and the final room visited by DFS is assigned 1.0.
        At branch points, child subtrees are visited in deterministic order, so each branch
        occupies a contiguous interval immediately after the parent interval.
        """
        if self.room_connections is None or len(self.room_connections) == 0:
            self._colormap_values = {}
            return
        
        # Build adjacency list from connections
        adjacency = {}
        for i1, j1, i2, j2, direction in self.room_connections:
            if (i1, j1) not in adjacency:
                adjacency[(i1, j1)] = []
            if (i2, j2) not in adjacency:
                adjacency[(i2, j2)] = []
            adjacency[(i1, j1)].append((i2, j2))
            adjacency[(i2, j2)].append((i1, j1))

        visited = set()
        preorder = []

        def subtree_size(start, banned, base_visited):
            """Count number of nodes reachable from `start` excluding `banned` and any in `base_visited`.

            This is used to order branches so that shorter subtrees are visited first.
            """
            seen = set(base_visited)
            if banned is not None:
                seen.add(banned)
            stack = [start]
            cnt = 0
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                cnt += 1
                for nb in adjacency.get(node, []):
                    if nb not in seen:
                        stack.append(nb)
            return cnt

        def dfs(room, parent=None):
            """Depth-first traversal producing a linear room ordering.

            Neighbors are visited in ascending order of their subtree size so
            that shorter branches come before longer ones.
            """
            visited.add(room)
            preorder.append(room)

            nbrs = adjacency.get(room, [])
            # Determine ordering: sort by subtree size (ascending), tie-break by coord
            ordered = sorted(
                nbrs,
                key=lambda n: (subtree_size(n, room, visited), n),
            )
            for nbr in ordered:
                if nbr == parent or nbr in visited:
                    continue
                dfs(nbr, room)

        # Start DFS traversal from room (0, 0) when available.
        if (0, 0) in adjacency:
            dfs((0, 0))
        elif len(adjacency) > 0:
            dfs(sorted(adjacency.keys())[0])

        # Handle disconnected components defensively.
        for room in sorted(adjacency.keys()):
            if room not in visited:
                dfs(room)

        self._colormap_values = {}
        num_rooms = len(preorder)
        if num_rooms == 0:
            return
        if num_rooms == 1:
            self._colormap_values[preorder[0]] = 0.0
            return

        # Normalize DFS order to [0, 1] so first room is 0 and last room is 1.
        denom = float(num_rooms - 1)
        for idx, room in enumerate(preorder):
            self._colormap_values[room] = idx / denom