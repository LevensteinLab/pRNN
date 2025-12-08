from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random
import numpy as np
from numpy.random import RandomState
from operator import add


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
        agent_view_size = 7,
        goal_pos = None,
        random_perturb=0,
        other_agent = False
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        
        self.goal_pos = goal_pos
        self.goal_color = [ 76, 255,  76 ]
        
        self.Lwidth = Lwidth
        self.Lheight = Lheight

        self.random_perturb = random_perturb
        self.perturb_mat = None
        if random_perturb>0:
            prng = RandomState(1234567891) #Perturbation is the same for each instance of the env - for reproducibility/dataset generation
            self.perturb_mat = prng.randint(-random_perturb,high=random_perturb,size=(size,size,3))
        
        self.other_agent = other_agent

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
        for i in range(self.Lheight+1,height-1):
            self.grid.horz_wall(self.Lwidth+1, i, length=width-self.Lwidth-1)
        
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        triloc  =   (width/3-4,height/3-4)
        plusloc =   (2*width/3-2,height/3-4)
        xloc    =   (width/3-3,2*height/3-2)
        #Add the other agent (as a ball...) or the shapes
        if hasattr(self, 'other_agent') and self.other_agent:
            #self.place_shape('plus',triloc,'blue')
            conspecific_start = [3,3]
            self.conspecific = Ball(color='red')
            #self.place_obj(self.conspecific, max_tries=100) 
            self.place_obj(self.conspecific, top=conspecific_start, size=(width,1), max_tries=100)

            self.conspecific.move_right = True
        else:   
            #Place the shapes
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


    def step(self,action):

        if hasattr(self, 'other_agent') and self.other_agent:
            #Update conspecific agent position 
            #If current direction is 'right', move right. If 'left', move left. Initialize direction when placing agent. (Toggle option)
            old_pos = self.conspecific.cur_pos
            #BACK/FORTH MOTION
            top = tuple(map(add, old_pos, (-1, 0)))
            if self.conspecific.move_right:
                top = tuple(map(add, old_pos, (1, 0)))

            try:
                self.place_obj(self.conspecific, top=top, size=(1,1), max_tries=10)
                self.grid.set(*old_pos, None)
            except:
                pass
            #     self.conspecific.cur_pos = (old_pos[0]+1,old_pos[1])

            if np.array_equal(self.conspecific.cur_pos, old_pos):
                self.conspecific.move_right = not self.conspecific.move_right

            #RANDOM MOTION
            # top = tuple(map(add, old_pos, (-1, -1)))
            # try:
            #     self.place_obj(self.conspecific, top=top, size=(3,3), reject_fn=reject_diagonal, max_tries=100)
            #     self.grid.set(*old_pos, None)
            # except:
            #     pass

        #Step the agent
        obs, reward, done, info = super().step(action)


        if hasattr(self, 'other_agent') and self.other_agent:
            if np.array_equal(self.conspecific.cur_pos, [-1,-1]):
                self.conspecific.cur_pos = self.agent_pos
        
        return obs, reward, done, info


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

def reject_diagonal(self, pos):
    current_pos = self.conspecific.cur_pos
    diagonals = current_pos + np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    if any((pos == diagonal).all() for diagonal in diagonals):
        return True
    return False
        

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
        super().__init__(size=18,Lwidth=10,Lheight=8, agent_view_size = 5,
                         agent_start_pos=None,**kwargs)

class LEnv_18_v5_rperturbL(L_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18,Lwidth=10,Lheight=8, agent_view_size = 5,
                         random_perturb=15,
                         agent_start_pos=None,**kwargs)
        
class LEnv_18_v5_rperturbH(L_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18,Lwidth=10,Lheight=8, agent_view_size = 5,
                         random_perturb=75,
                         agent_start_pos=None,**kwargs)
        
class LEnv_18_goal(L_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18, Lwidth=10,Lheight=8,
                         agent_start_pos=None, goal_pos = [9,2],
                         **kwargs)

class LEnv_18_conspecific(L_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18,Lwidth=10,Lheight=8,
                         agent_start_pos=None, other_agent = True, **kwargs)

class LEnv_10_conspecific(L_Env):
    def __init__(self, **kwargs):
        super().__init__(size=10,Lwidth=6,Lheight=4,
                         agent_start_pos=None, other_agent = True, **kwargs)
        

register(
    id='MiniGrid-LRoom-16x16-v0',
    entry_point='gym_minigrid.envs:LEnv_16'
)



register(
    id='MiniGrid-LRoom-20x20-v0',
    entry_point='gym_minigrid.envs:LEnv_20'
)

register(
    id='MiniGrid-LRoom-18x18-v0',
    entry_point='gym_minigrid.envs:LEnv_18'
)

register(
    id='MiniGrid-LRoom_v5-18x18-v0',
    entry_point='gym_minigrid.envs:LEnv_18_v5'
)

register(
    id='MiniGrid-LRoom_v5_rperturbL-18x18-v0',
    entry_point='gym_minigrid.envs:LEnv_18_v5_rperturbL'
)

register(
    id='MiniGrid-LRoom_v5_rperturbH-18x18-v0',
    entry_point='gym_minigrid.envs:LEnv_18_v5_rperturbH'
)

register(
    id='MiniGrid-LRoom_Goal-18x18-v0',
    entry_point='gym_minigrid.envs:LEnv_18_goal'
)


register(
    id='MiniGrid-LRoom_Conspecific-18x18-v0',
    entry_point='gym_minigrid.envs:LEnv_18_conspecific'
)

register(
    id='MiniGrid-LRoom_Conspecific-10x10-v0',
    entry_point='gym_minigrid.envs:LEnv_10_conspecific'
)

