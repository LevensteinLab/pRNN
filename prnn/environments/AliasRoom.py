from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random
import numpy as np
from numpy.random import RandomState

COLORSHIFT = {
    'red'   : 'green',
    'green' : 'grey',
    'blue'  : 'green',
    'purple': 'yellow',
    'yellow': 'red',
    'grey'  : 'brown'
}


class Alias_Env(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=16,
        Lwidth=10, Lheight=8,
        agent_start_pos=(5,5),
        agent_start_dir=0,
        agent_view_size = 7,
        goal_pos = None,
        empty_color = (0,0,0),
        color_shift = False,
        random_perturb=0
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        
        self.goal_pos = goal_pos
        self.goal_color = [ 76, 255,  76 ]

        self.color_shift = color_shift
        
        self.Lwidth = Lwidth
        self.Lheight = Lheight

        self.random_perturb = random_perturb
        self.perturb_mat = None
        if random_perturb>0:
            prng = RandomState(1234567890) #Perturbation is the same for each instance of the env - for reproducibility/dataset generation
            self.perturb_mat = prng.randint(-random_perturb,high=random_perturb,size=(size,size,3))
        
        super().__init__(
            grid_size=size,
            max_steps=10*size*size,
            # Set this to True for maximum speed
            see_through_walls=True,
            agent_view_size=agent_view_size,
            empty_color=empty_color
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
        
        ULsize = [3,15]
        for i in range(1,ULsize[0]+1):
            self.grid.horz_wall(ULsize[1]+1, i, length=width-ULsize[1]-1)
            
        ULsize = [6,3]
        for i in range(1,ULsize[0]+1):
            self.grid.horz_wall(1, i, length=ULsize[1])
            
        ULsize2 = [5,8]
        for i in range(ULsize[0]+1,ULsize[0]+ULsize2[0]+1):
            self.grid.horz_wall(1, i, length=ULsize2[1])
            
        for i in range(self.Lheight+1,height-1):
            self.grid.horz_wall(self.Lwidth+1, i, length=width-self.Lwidth-1)
        
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        
            
        #Place the shapes
        #triloc  =   (4,13)
        plusloc =   (10,2)
        plusloc2  =   (4,13)
        #xloc    =   (12,9)
        #self.place_shape('triangle',triloc,'blue')
        self.place_shape('plus',plusloc,'red')
        self.place_shape('plus',plusloc2,'red')
        #self.place_shape('x',xloc,'yellow')

        if self.color_shift:
            for gridobj in self.grid.grid:
                if gridobj is not None:
                    gridobj.color = COLORSHIFT[gridobj.color]

        #Add the random perturbation
        if self.random_perturb > 0:
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
                [
                 [0,1,1,0],
                 [1,1,1,1],
                 [1,1,1,1],
                 [0,1,1,0]]),
            'triangle':np.array(
                [[1,0,0,0],
                 [1,1,0,0],
                 [1,1,1,0],
                 [1,1,1,1]]),
            'x':np.array(
                [[1,0,1,1],
                 [1,1,1,0],
                 [0,1,1,1],
                 [1,1,0,1]])
            }
            
        shapecoords = np.transpose(np.nonzero(shapegrid[shape]))+np.array(pos,dtype='int32')

        for coord in shapecoords:
            self.put_obj(Floor(color), coord[0], coord[1])
        
        

class AEnv_16(Alias_Env):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None,**kwargs)
        
class AEnv_20(Alias_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20,Lwidth=10,Lheight=14,**kwargs)

class AEnv_20_cshift(Alias_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20,Lwidth=10,Lheight=14,
                         empty_color = (75,125,125),
                         color_shift = True,
                         **kwargs)
        
class AEnv_20_rperturbL(Alias_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20,Lwidth=10,Lheight=14,random_perturb=15,**kwargs)

class AEnv_20_rperturbH(Alias_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20,Lwidth=10,Lheight=14,random_perturb=75,**kwargs)

class AEnv_20_rperturbVH(Alias_Env):
   def __init__(self, **kwargs):
       super().__init__(size=20,Lwidth=10,Lheight=14,random_perturb=80,**kwargs)
        
class AEnv_18(Alias_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18,Lwidth=10,Lheight=8,
                         agent_start_pos=None,**kwargs)
        


register(
    id='MiniGrid-AliasRoom-16x16-v0',
    entry_point='gym_minigrid.envs:AEnv_16'
)

register(
    id='MiniGrid-AliasRoom-18x18-v0',
    entry_point='gym_minigrid.envs:AEnv_18'
)

register(
    id='MiniGrid-AliasRoom-20x20-v0',
    entry_point='gym_minigrid.envs:AEnv_20'
)



register(
    id='MiniGrid-AliasRoom_cshift-20x20-v0',
    entry_point='gym_minigrid.envs:AEnv_20_cshift'
)

register(
    id='MiniGrid-AliasRoom_rperturbH-20x20-v0',
    entry_point='gym_minigrid.envs:AEnv_20_rperturbH'
)

register(
    id='MiniGrid-AliasRoom_rperturbL-20x20-v0',
    entry_point='gym_minigrid.envs:AEnv_20_rperturbL'
)

register(
    id='MiniGrid-AliasRoom_rperturbVH-20x20-v0',
    entry_point='gym_minigrid.envs:AEnv_20_rperturbVH'
)


