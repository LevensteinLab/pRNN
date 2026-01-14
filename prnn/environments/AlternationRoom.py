from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random
import numpy as np


class Alternation_Env(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """
    
    def __init__(
        self,
        size=23,
        agent_start_pos=(11,2), #Previously (11,20)
        agent_start_dir=3,
        agent_view_size = 7,
        goal_pos = None,

    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        
        super().__init__(
            grid_size=size,
            max_steps=10*size*size,
            # Set this to True for maximum speed
            see_through_walls=True,
            agent_view_size=agent_view_size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        #Consider: walls at -1, rather than 0
        self.grid.horz_wall(0,0)
        self.grid.vert_wall(0,0)
        self.grid.horz_wall(0,height-1)
        self.grid.vert_wall(width-1,0)

        self.corridor_width = 3
        self.block_height = height-2-2*self.corridor_width
        self.block_width = int((width-2-3*self.corridor_width)/2)
        for i in range(self.corridor_width+1,height-self.corridor_width-1):
            self.grid.horz_wall(self.corridor_width+1, i, length=self.block_width)
            self.grid.horz_wall(2*self.corridor_width+self.block_width+1, i, length=self.block_width)
        
        #Place the shapes
        triloc  =   (1,1)
        plusloc =   (width-self.corridor_width-1,height/2)
        xloc    =   (1+self.corridor_width+self.block_width,height-3*self.corridor_width)
        self.place_shape('triangle',triloc,'blue')
        self.place_shape('plus',plusloc,'red')
        self.place_shape('x',xloc,'yellow')

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.agent_pos = self.place_agent()
            self.agent_dir = self.agent_dir

        self.mission = "alternate"


    def place_shape(self,shape,pos,color):
        """
        Place a 6x6 shape with lower left corner at (x,y)
        """
        shapegrid={
            'plus':np.array(
                [[0,1,0],
                 [1,1,1],
                 [0,1,0]]),
            'triangle':np.array(
                [[1,0,0,],
                 [1,1,0,],
                 [1,1,1]]),
            'x':np.array(
                [[1,0,1],
                 [0,1,0],
                 [1,0,1]])
            }
            
        shapecoords = np.transpose(np.nonzero(shapegrid[shape]))+np.array(pos,dtype='int32')

        for coord in shapecoords:
            self.put_obj(Floor(color), coord[0], coord[1])


register(
    id='MiniGrid-Alternation-24x24-v0',
    entry_point='gym_minigrid.envs:Alternation_Env'
)
