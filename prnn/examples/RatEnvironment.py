import numpy as np
import ratinabox
import random
import math
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

#functions for the cheeseboard environment: 

#creates the cirlce boundary for the cheeseboard
def create_circular_boundary(radius=1, num_points=100, center=(0, 0)):
    """Creates a circular boundary as a list of points."""
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
    boundary = [[center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)] for t in theta]
    return boundary


#generates the food holes inside the environment
def generate_wells(env, num_holes=177, board_radius=0.6, hole_radius=0.025, offset=0.1):
    """Generates a specific number of holes evenly spaced inside a circular boundary using a rectangular grid pattern."""
    # Calculate the approximate number of rows and columns
    area_per_hole = (np.pi * board_radius**2) / num_holes
    grid_spacing = np.sqrt(area_per_hole)
    num_cols = int(np.sqrt(num_holes))
    num_rows = int(np.ceil(num_holes / num_cols))

    # Adjust grid spacing to fit the circular boundary
    x_spacing = (2 * board_radius) / num_cols
    y_spacing = (2 * board_radius) / num_rows

    holes = []
    num_holes_created = 0

    # Create holes in a rectangular grid
    for i in range(num_cols):
        for j in range(num_rows):
            if num_holes_created >= num_holes:
                break

            # Compute the candidate grid position
            x = -board_radius + (i + 0.5) * x_spacing
            y = -board_radius + (j + 0.5) * y_spacing

            # Ensure the hole is inside the circular boundary
            if np.sqrt(x**2 + y**2) + hole_radius + offset < board_radius:
                # Add objects to environment
                if(i == 0 and j == 0):
                    env.add_object([x+board_radius, y+board_radius], type='new')
                else:
                    env.add_object([x+board_radius, y+board_radius], type='same')
                num_holes_created += 1

#picks reward holes randomly 
def set_random_holes_reward(env, num_rewards=3):
    holes = env.objects['objects']  # Assuming env.holes is a list of hole objects
    if len(holes) < num_rewards:
        raise ValueError("Not enough holes in the environment to set the specified number of rewards.")
    
    #add a seed to make it reproducible TODO (add as argument as well)
    reward_hole_indices = np.random.choice(len(holes), num_rewards, replace=False)

    return reward_hole_indices
    
#adds objects around the environment in a circular pattern
def add_uniform_objects(env, dist = 0.6, gap = 0.05, num_objects = 32):

    d = dist + gap
    newangles_degrees = np.linspace(0, 360, num_objects, endpoint=False).tolist()
    newangles_radians = [math.radians(angle) for angle in newangles_degrees]
    newpositions = [(d * math.cos(theta), d * math.sin(theta)) for theta in newangles_radians]
    newpositions = [(pos[0]+0.6, pos[1]+0.6) for pos in newpositions] #center of the cheeseboard 

    random.seed(42) #set the seed for reproducibility
    random.shuffle(newpositions)


    env.add_object([newpositions[0][0], newpositions[0][1]], type='new')
    for i in range(1,10):
        env.add_object([newpositions[i][0], newpositions[i][1]], type='same')

    env.add_object([newpositions[10][0], newpositions[10][1]], type='new')
    for i in range(11,21):
        env.add_object([newpositions[i][0], newpositions[i][1]], type='same')

    env.add_object([newpositions[21][0], newpositions[21][1]], type='new')
    for i in range(22,32):
        env.add_object([newpositions[i][0], newpositions[i][1]], type='same')


def add_nonuniform_objects(env, dist = 0.6, gap = 0.05, num_objects = 64, random = False):

    d = dist + gap
    newangles_degrees = np.linspace(0, 360, num_objects, endpoint=False).tolist()
    newangles_radians = [math.radians(angle) for angle in newangles_degrees]
    newpositions = [(d * math.cos(theta) + 0.6, d * math.sin(theta) + 0.6) for theta in newangles_radians]

    if(random):
        random.shuffle(newpositions)

    env.add_object([newpositions[0][0], newpositions[0][1]], type='new')
    env.add_object([(newpositions[1][0])+.1, newpositions[1][1]], type='same')
    for i in [2,5, 12, 34, 45]:
        env.add_object([newpositions[i][0], newpositions[i][1]], type='same')
    
    env.add_object([(newpositions[10][0]) + .05, (newpositions[10][1]) +.05], type='new')
    for i in (14,16, 33, 36):
         env.add_object([newpositions[i][0], newpositions[i][1]], type='same')
    for i in [20, 25, 37]:
        env.add_object([(newpositions[i][0])-.05, newpositions[i][1]], type='same')

    env.add_object([newpositions[21][0], newpositions[21][1]], type='new')
    for i in range(26,32):
         env.add_object([newpositions[i][0], newpositions[i][1]], type='same')
    for i in range(60,64):
         env.add_object([newpositions[i][0], newpositions[i][1]], type='same')

    env.add_object([newpositions[42][0], newpositions[42][1]], type='new')
    for i in range(52,55):
         env.add_object([newpositions[i][0], newpositions[i][1]], type='same')

    

def make_rat_env(key):
    """
    Create a rat environment based on the given key.
    Important: the boundaries should be all positive,
    so there's no negative coordinates.
    """
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

    if key == 'cheeseboard':
        # Define a circular environment
        circle_boundary = create_circular_boundary(radius=.6, num_points=200, center=(0.6, 0.6))

        # Initialize the Environment with a circular boundary
        Env = Environment(params={"boundary": circle_boundary,
                                  "dx": 1/16,
                                  "scale": 1})
        Env.object_colormap = 'tab10'

        generate_wells(Env, board_radius=0.6, num_holes = 150, offset=0.1)
        #indices = set_random_holes_reward(Env, num_rewards=3)

        #reward_positions = Env.objects['objects'][indices]

        add_nonuniform_objects(Env)

        # Add floor marks
        for n in range(tri[0].size):
            if n==0:
                Env.add_object([arr[tri[0][n]+1],(arr[tri[1][n]+9]) + 0.7], type='new')
            else:
                Env.add_object([arr[tri[0][n]+1],(arr[tri[1][n]+9]) + 0.7], type='same')

        for n in range(plus[0].size):
            if n==0:
                Env.add_object([(arr[plus[0][n]+9])+.7,arr[plus[1][n]+9]], type='new')
            else:
                Env.add_object([(arr[plus[0][n]+9])+.7,arr[plus[1][n]+9]], type='same')

        for n in range(x[0].size):
            if n==0:
                Env.add_object([arr[x[0][n]+2],(arr[x[1][n]+1])-0.6], type='new')
            else:
                Env.add_object([arr[x[0][n]+2],(arr[x[1][n]+1])-0.6], type='same')

    

    return Env




