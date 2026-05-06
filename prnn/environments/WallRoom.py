from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random
import numpy as np
from numpy.random import RandomState
from operator import add


class Wall_Env(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=16,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        agent_view_size = 7,
        goal_pos = (9,3),
        random_perturb=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        
        self.goal_pos = goal_pos
        self.goal_color = [ 76, 255,  76 ]

        self.random_perturb = random_perturb
        self.perturb_mat = None
        if random_perturb>0:
            prng = RandomState(1234567891) #Perturbation is the same for each instance of the env - for reproducibility/dataset generation
            self.perturb_mat = prng.randint(-random_perturb,high=random_perturb,size=(size,size,3))
        
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

        self.grid.horz_wall(4,7,length=10)
        
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        triloc  =   (width/3-4,height/3-4)
        plusloc =   (2*width/3-0,height/3-4)
        xloc    =   (width/3-2,2*height/3-2)

        self.place_shape('triangle',triloc,'blue')
        self.place_shape('plus',plusloc,'red')
        self.place_shape('x',xloc,'yellow')

        #Add the random perturbation
        if hasattr(self, 'random_perturb') and self.random_perturb > 0:
            for i in range(width):
                for j in range(height):
                    self.grid.setP(i,j,tuple(self.perturb_mat[i,j,:]))
            
        self.mission = "get to the green goal square"
        
        
        # Place the goal
        if hasattr(self, 'goal_pos') and self.goal_pos is not None:
            #coordsW = np.random.randint(low=1, high=width-1, size=2)
            coordsW = self.goal_pos
            self.put_obj(Goal(), coordsW[0], coordsW[1])
            #self.put_obj(Goal(), coordsW[0]+1, coordsW[1])
            #self.put_obj(Goal(), coordsW[0], coordsW[1]+1)
            #self.put_obj(Goal(), coordsW[0]+1, coordsW[1]+1)
            #Consider goal 2 x 2 squares


    def place_shape(self,shape,pos,color):
        """
        Place a 6x6 shape with lower left corner at (x,y)
        """
        shapegrid={
            'plus':np.array(
                [[0,1,1,0],
                 [1,1,1,1],
                 [1,1,1,1],
                 [0,1,1,0,]]),
            'triangle':np.array(
                [[1,0,0,0],
                 [1,1,0,0],
                 [1,1,1,0],
                 [1,1,1,1]]),
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

        

class WEnv_18(Wall_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18, agent_start_pos=None,**kwargs)
        

register(
    id='MiniGrid-WallRoom-18x18-v0',
    entry_point='gym_minigrid.envs:WEnv_18'
)

