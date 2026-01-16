from minigrid.minigrid_env import *
from minigrid.core.world_object import *
from minigrid.core.mission import MissionSpace
from gymnasium.envs.registration import register

import random
import numpy as np


class L_Env(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=16,
        Lwidth=10, Lheight=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        agent_view_size=7,
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir 
        
        self.Lwidth = Lwidth
        self.Lheight = Lheight

        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=10*size*size,
            # Set this to True for maximum speed
            see_through_walls=True,
            agent_view_size=agent_view_size,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height, regenerate=True):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        #Consider: walls at -1, rather than 0
        self.grid.horz_wall(0,0)
        self.grid.vert_wall(0,0)
        self.grid.horz_wall(0,height-1)
        self.grid.vert_wall(width-1,0)
        for i in range(self.Lheight+1,height-1):
            self.grid.horz_wall(self.Lwidth+1, i, length=width-self.Lwidth-1)
        
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
            
        #Place the shapes
        triloc  =   (width//3-4,height//3-4)
        plusloc =   (2*width//3-2,height//3-4)
        xloc    =   (width//3-3,2*height//3-2)
        self.place_shape('triangle',triloc,'blue')
        self.place_shape('plus',plusloc,'red')
        self.place_shape('x',xloc,'yellow')

        self.mission = "get to the green goal square"
    
    
    def place_shape(self,shape,pos,color):
        """
        Place a 6x6 shape with lower left corner at (x,y)
        """
        shapegrid={
            'plus':np.array(
                [[0,0,1,1,0,0],
                 [0,0,1,1,0,0],
                 [1,1,1,1,1,1],
                 [1,1,1,1,1,1],
                 [0,0,1,1,0,0],
                 [0,0,1,1,0,0]]),
            'triangle':np.array(
                [[1,0,0,0,0,0],
                 [1,1,0,0,0,0],
                 [1,1,1,0,0,0],
                 [1,1,1,1,0,0],
                 [1,1,1,1,1,0],
                 [1,1,1,1,1,1]]),
            'x':np.array(
                [[1,1,0,0,1,1],
                 [1,1,1,1,1,1],
                 [0,1,1,1,1,0],
                 [0,1,1,1,1,0],
                 [1,1,1,1,1,1],
                 [1,1,0,0,1,1]])
            }
            
        shapecoords = np.transpose(np.nonzero(shapegrid[shape]))+np.array(pos,dtype='int32')

        for coord in shapecoords:
            self.put_obj(Floor(color), coord[0], coord[1])
        
        

class LEnv_16(L_Env):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None,**kwargs)
        
class LEnv_20(L_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20,Lwidth=12,Lheight=10,**kwargs)
        
class LEnv_18(L_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18,Lwidth=10,Lheight=8,
                         agent_start_pos=None,**kwargs)
        
class LEnv_18_v5(L_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18,Lwidth=10,Lheight=8,
                         agent_start_pos=None, agent_view_size=5, **kwargs)
        

register(
    id='LRoom-16x16-v0',
    entry_point='prnn.environments.Lroom:LEnv_16'
)

register(
    id='LRoom-20x20-v0',
    entry_point='prnn.environments.Lroom:LEnv_20'
)

register(
    id='LRoom-18x18-v0',
    entry_point='prnn.environments.Lroom:LEnv_18'
)

register(
    id='LRoom-18x18-v5-v0',
    entry_point='prnn.environments.Lroom:LEnv_18_v5')