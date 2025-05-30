import numpy as np
import random
import argparse

# Hyperparameter grid
#https://www.kaggle.com/code/willkoehrsen/intro-to-model-tuning-grid-and-random-search/notebook
param_grid = {

    #Relative learning rate to initial weight distribution
    'learning_rate': list(np.logspace(-3.0, 
                                      -1.5,
                                      1000)), 
    'bptttrunc': list(np.int64(np.round(np.logspace(np.log10(100),
                                                    np.log10(2000), 
                                                    10000)))), 
    'weightdecay': list(np.logspace(np.log10(1e-4),   #ratio with learning_rate. converted later
                                      np.log10(1e-1),
                                      1000)),
    'nneurons': list(np.int64(np.round(np.logspace(np.log10(200),
                                                    np.log10(1000), 
                                                    10000)))), 
    'ntimescale': list(np.linspace(1,
                                   4,  #previously 3
                                   10000)), 
    'dropp': list(np.linspace(0,
                              0.2,
                              10000)), 
    'noisestd' : list(np.logspace(-3,-0.5,
                                      1000)), 
    'sigmaD': list(np.logspace(np.log10(0.005),
                               np.log10(0.2),
                                      1000)),
    'sigmaA': list(np.linspace(1,
                                   5,
                                   10000)), 
}

# param_grid = {

#     #Relative learning rate to initial weight distribution
#     'learning_rate': list(np.logspace(-3.5, 
#                                       -1.5,
#                                       1000)),
#     'seqdur': list(np.int64(np.round(np.logspace(np.log10(30),
#                                                  np.log10(2000),
#                                                  10000)))), 
#     'bptttrunc': list(np.int64(np.round(np.logspace(np.log10(3),
#                                                     np.log10(2000), 
#                                                     10000)))),  
#     'weightdecay': list(np.logspace(np.log10(1e-4),   #ratio with learning_rate. converted later
#                                       np.log10(1e-1),
#                                       1000)),
#     'nneurons': list(np.int64(np.round(np.logspace(np.log10(100),
#                                                     np.log10(1000), 
#                                                     10000)))), 
#     'ntimescale': list(np.linspace(1,
#                                    4,  #previously 3
#                                    10000)), 
#     'dropp': list(np.linspace(0,
#                               0.2,
#                               10000)), 
#     'noisestd' : list(np.logspace(-3,0,
#                                       1000)),
#     'sparsity' : list(np.linspace(0.1,0.8,
#                                       1000)),
#     'bias_lr' : list(np.logspace(-2.5,0,
#                                       1000)),
#     'agentspeed': list(np.linspace(0.1, 
#                                    0.7,
#                                    121)),
# }

# param_grid = {
#     'agentspeed': list(np.linspace(0.1, 
#                                    0.7,
#                                    121)),
#     'thigmotaxis': list(np.linspace(0.05, 
#                                     0.5,
#                                     91)),
#     'HDbins': [6,8,10,12,14,16],
# }



parser = argparse.ArgumentParser()
parser.add_argument("--s", type=int,
                    help="Random Seed")
parser.add_argument("--i", type=int,
                    help="Iterate")
args = parser.parse_args()
seed = args.s
iterate = args.i

random.seed(seed)


# Randomly sample from dictionary
for ii in range(iterate+1):
    random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

#random_params['learning_rate']=random_params['learning_rate']*np.sqrt(1/random_params['nneurons'])
#random_params['weightdecay']=random_params['weightdecay']*random_params['learning_rate']

#random_params
print(list(random_params.values()))

