import numpy as np
import ratinabox
from ratinabox.Environment import Environment

shapegrid={
    'plus':np.array(
        [[0,0,1,1,0,0],
         [0,0,1,1,0,0],
         [1,1,1,1,1,1],
         [1,1,1,1,1,1],
         [0,0,1,1,0,0],
         [0,0,1,1,0,0]]),
    'triangle':np.array(
        [[0,0,0,0,0,1],
         [0,0,0,0,1,1],
         [0,0,0,1,1,1],
         [0,0,1,1,1,1],
         [0,1,1,1,1,1],
         [1,1,1,1,1,1]]),
    'x':np.array(
        [[1,1,0,0,1,1],
         [1,1,1,1,1,1],
         [0,1,1,1,1,0],
         [0,1,1,1,1,0],
         [1,1,1,1,1,1],
         [1,1,0,0,1,1]])
    }

arr = np.arange(start=1/32, stop=1, step=1/16)
x = shapegrid['x'].nonzero()
plus = shapegrid['plus'].nonzero()
tri = shapegrid['triangle'].nonzero()

def make_rat_env(key):
    # TODO: think about how to make this more general
    if key == 'RiaB-LRoom':
        # Create the L-shaped environment
        Env = Environment(
                          params={"boundary": [[0,0],
                                               [0.625,0],
                                               [0.625,0.5],
                                               [1,0.5],
                                               [1,1],
                                               [0,1]],
                                  "dx": 1/16,
                                  "scale": 1}
                         )
        
        Env.object_colormap = 'brg'

        # Add floor marks
        for n in range(tri[0].size):
            if n==0:
                Env.add_object([arr[tri[0][n]+1],arr[tri[1][n]+9]], type='new')
            else:
                Env.add_object([arr[tri[0][n]+1],arr[tri[1][n]+9]], type='same')

        for n in range(plus[0].size):
            if n==0:
                Env.add_object([arr[plus[0][n]+9],arr[plus[1][n]+9]], type='new')
            else:
                Env.add_object([arr[plus[0][n]+9],arr[plus[1][n]+9]], type='same')

        for n in range(x[0].size):
            if n==0:
                Env.add_object([arr[x[0][n]+2],arr[x[1][n]+1]], type='new')
            else:
                Env.add_object([arr[x[0][n]+2],arr[x[1][n]+1]], type='same')
    
    return Env


