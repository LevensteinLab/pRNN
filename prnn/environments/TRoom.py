from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np

# Dictionary to handle the color swapping for objects
COLORSHIFT = {
    'red'   : 'green',
    'green' : 'grey',
    'blue'  : 'green',
    'purple': 'green',
    'yellow': 'green',
    'grey'  : 'blue'
}

class TRoom_Env(MiniGridEnv):
    def __init__(
        self,
        size=16,
        agent_start_pos=(1, 1),
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
        self.tri_color = tri_color
        self.plus_color = plus_color
        self.x_color = x_color
        self.order = order
        self.color_shift = color_shift

        # Default max_steps matching your logic
        max_steps = kwargs.pop("max_steps", 10 * size * size)

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            empty_color=empty_color,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # 1. Generate surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.vert_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(width - 1, 0)

        # 2. Place Shapes
        loc = [
            (width / 3 - 4, height / 3 - 4),
            (2 * width / 3 - 2, height / 3 - 4),
            (width / 3 - 3, 2 * height / 3 - 2),
            (2 * width / 3 - 2, 2 * height / 3 - 2),
        ]

        shapes = {
            "T": {"name": "triangle", "color": self.tri_color},
            "P": {"name": "plus", "color": self.plus_color},
            "X": {"name": "x", "color": self.x_color},
            "D": {"name": "dash", "color": self.tri_color},
        }

        for idx, char in enumerate(self.order):
            self.place_shape(shapes[char]["name"], loc[idx], shapes[char]["color"])

        # 3. T-room wall logic
        TRoom_delimeter = 5
        if width == 18:
            TRoom_delimeter = 6
        elif width == 20:
            TRoom_delimeter = 7

        for i in range(TRoom_delimeter):
            self.grid.vert_wall(i, int(width / 2), length=int(width / 2))
            self.grid.vert_wall(int(height - 2) - i, int(width / 2), length=int(width / 2))

        for j in range(int(TRoom_delimeter / 2) + 2):
            self.grid.horz_wall(0, j, length=width - 1)

        # 4. Handle Color Shift (Walls and Shape-Floors)
        if self.color_shift:
            for gridobj in self.grid.grid:
                if gridobj is not None:
                    gridobj.color = COLORSHIFT[gridobj.color]

        # 5. Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def place_shape(self, shape, pos, color):
        shapegrid = {
            "plus": np.array([[0,0,1,1,0,0],[0,0,1,1,0,0],[1,1,1,1,1,1],[1,1,1,1,1,1],[0,0,1,1,0,0],[0,0,1,1,0,0]]),
            "triangle": np.array([[1,0,0,0,0,0],[1,1,0,0,0,0],[1,1,1,0,0,0],[1,1,1,1,0,0],[1,1,1,1,1,0],[1,1,1,1,1,1]]),
            "x": np.array([[1,1,0,0,1,1],[1,1,1,1,1,1],[0,1,1,1,1,0],[0,1,1,1,1,0],[1,1,1,1,1,1],[1,1,0,0,1,1]]),
            "dash": np.array([[1,1,0,0,0,0],[1,1,1,0,0,0],[0,1,1,1,0,0],[0,0,1,1,1,0],[0,0,0,1,1,1],[0,0,0,0,1,1]]),
        }
        shapecoords = np.transpose(np.nonzero(shapegrid[shape])) + np.array(pos, dtype="int32")
        for coord in shapecoords:
            self.put_obj(Floor(color), int(coord[0]), int(coord[1]))

# --- Standard Classes ---
class TRoomEnv_16(TRoom_Env):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None, **kwargs)

class TRoomEnv_18(TRoom_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18, agent_start_pos=None, **kwargs)

class TRoomEnv_20(TRoom_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20, agent_start_pos=None, **kwargs)

# --- Color Shifted Classes ---
class TRoomEnv_16_cshift(TRoom_Env):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None, 
                         empty_color=(75, 125, 125), color_shift=True, **kwargs)

class TRoomEnv_18_cshift(TRoom_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18, agent_start_pos=None, 
                         empty_color=(75, 125, 125), color_shift=True, **kwargs)

class TRoomEnv_20_cshift(TRoom_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20, agent_start_pos=None, 
                         empty_color=(75, 125, 125), color_shift=True, **kwargs)
        

register(
    id='MiniGrid-TRoom-16x16-v0',
    entry_point='gym_minigrid.envs:TRoomEnv_16'
)

register(
    id='MiniGrid-TRoom-18x18-v0',
    entry_point='gym_minigrid.envs:TRoomEnv_18'
)

register(
    id='MiniGrid-TRoom-20x20-v0',
    entry_point='gym_minigrid.envs:TRoomEnv_20'
)

register(
    id='MiniGrid-TRoom-CShift-16x16-v0',
    entry_point='gym_minigrid.envs:TRoomEnv_16_cshift'
)

register(
    id='MiniGrid-TRoom-CShift-18x18-v0',
    entry_point='gym_minigrid.envs:TRoomEnv_18_cshift'
)

register(
    id='MiniGrid-TRoom-CShift-20x20-v0',
    entry_point='gym_minigrid.envs:TRoomEnv_20_cshift'
)