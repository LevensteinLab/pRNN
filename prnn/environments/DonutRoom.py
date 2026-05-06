from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np

# Dictionary for the color swapping logic
COLORSHIFT = {
    'red'   : 'red',
    'green' : 'grey',
    'blue'  : 'red',
    'purple': 'red',
    'yellow': 'red',
    'grey'  : 'brown'
}

class Donut_Env(MiniGridEnv):
    def __init__(
        self,
        size=16,
        Lwidth=10,
        Lheight=8,
        agent_start_pos=(3, 3),
        agent_start_dir=0,
        tri_color="blue",
        plus_color="red",
        x_color="yellow",
        order="TPXD",
        empty_color=(0, 0, 0),
        color_shift=False,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.Lwidth = Lwidth
        self.Lheight = Lheight
        self.tri_color = tri_color
        self.plus_color = plus_color
        self.x_color = x_color
        self.order = order
        self.color_shift = color_shift

        # Default max_steps matching logic
        max_steps = kwargs.pop("max_steps", 10 * size * size)

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            empty_color=empty_color,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # 1. Outer walls
        self.grid.horz_wall(0, 0)
        self.grid.vert_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(width - 1, 0)

        # 2. Donut wall (horizontal bar)
        for i in range(int(height / 2) - 4, int(height / 2) + 4):
            self.grid.horz_wall(int(self.Lwidth / 2), i, length=8)

        # 3. Place the four ordered shapes
        loc = [
            (width / 3 - 4, height / 3 - 4),
            (2 * width / 3 - 1, height / 3 - 1),
            (width / 3 - 3, 2 * height / 3 - 2),
            (2 * width / 3 - 2, 2 * height / 3 - 2),
        ]

        shapes = {
            "T": {"name": "triangle", "color": self.tri_color},
            "P": {"name": "dash", "color": self.plus_color},
            "X": {"name": "x", "color": self.x_color},
            "D": {"name": "dash", "color": self.tri_color},
        }

        for idx, char in enumerate(self.order):
            self.place_shape(shapes[char]["name"], loc[idx], shapes[char]["color"])

        # 4. Additional decorations
        self.place_shape("plus", (width / 3 - 1, height / 3 - 5), self.x_color)
        self.place_shape("plus", (width / 3,     height / 3 - 5), self.x_color)
        self.place_shape("plus", (width / 3 + 1, height / 3 - 5), self.x_color)
        self.place_shape("plus", (width / 3 + 2, height / 3 - 5), self.x_color)

        self.place_shape("plus", (width / 3 - 3, height / 3 + 6), self.plus_color)
        self.place_shape("plus", (width / 3 - 2, height / 3 + 6), self.plus_color)
        self.place_shape("plus", (width / 3 - 1, height / 3 + 6), self.plus_color)
        self.place_shape("plus", (width / 3,     height / 3 + 6), self.plus_color)

        # 5. Handle Color Shift (Apply to all objects in grid)
        if self.color_shift:
            for gridobj in self.grid.grid:
                if gridobj is not None:
                    gridobj.color = COLORSHIFT[gridobj.color]

        # 6. Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "reach the goal"

    def place_shape(self, shape, pos, color):
        shapegrid = {
            "plus": np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,1,0,0]]),
            "triangle": np.array([[1,1,1,1,1,0],[1,1,1,1,1,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]),
            "x": np.array([[0,0,0,0,1,1],[1,1,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]),
            "dash": np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,1],[0,0,0,0,1,1]]),
        }
        shapecoords = np.transpose(np.nonzero(shapegrid[shape])) + np.array(pos, dtype="int32")
        for coord in shapecoords:
            self.put_obj(Floor(color), int(coord[0]), int(coord[1]))

# --- Standard Classes ---
class DonutEnv_16(Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None, **kwargs)

class DonutEnv_18(Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18, agent_start_pos=None, **kwargs)

class DonutEnv_20(Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20, agent_start_pos=None, **kwargs)

# --- Color Shifted Classes ---
class DonutEnv_16_cshift(Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None, 
                         empty_color=(194, 178, 128), color_shift=True, **kwargs)

class DonutEnv_18_cshift(Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18, agent_start_pos=None, 
                         empty_color=(128, 128, 128), color_shift=True, **kwargs)

class DonutEnv_20_cshift(Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20, agent_start_pos=None, 
                         empty_color=(128, 128, 128), color_shift=True, **kwargs)
        
register(
    id='MiniGrid-Donut-16x16-v0',
    entry_point='gym_minigrid.envs:DonutEnv_16'
)

register(
    id='MiniGrid-Donut-18x18-v0',
    entry_point='gym_minigrid.envs:DonutEnv_18'
)

register(
    id='MiniGrid-Donut-20x20-v0',
    entry_point='gym_minigrid.envs:DonutEnv_20'
)

register(
    id='MiniGrid-Donut-CShift-16x16-v0',
    entry_point='gym_minigrid.envs:DonutEnv_16_cshift'
)

register(
    id='MiniGrid-Donut-CShift-18x18-v0',
    entry_point='gym_minigrid.envs:DonutEnv_18_cshift'
)

register(
    id='MiniGrid-Donut-CShift-20x20-v0',
    entry_point='gym_minigrid.envs:DonutEnv_20_cshift'
)