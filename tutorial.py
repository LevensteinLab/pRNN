### This is just a copy of tutorial.ipynb, but in a python file so i can submit it as a job
ARCHITECTURE_NAME = 'thRNN_5win' 
#---

#import the pRNN class
from prnn.utils.predictiveNet import PredictiveNet

from prnn.utils.env import make_env
from prnn.utils.agent import RandomActionAgent

import matplotlib.pyplot as plt
import numpy as np

#Make a gridworld environment
env_package = 'gym-minigrid' 
env_key = 'MiniGrid-LRoom-18x18-v0'
act_enc = 'SpeedHD' #actions will be encoded as speed and one hot-encoded head direction

env = make_env(env_key=env_key, package=env_package, act_enc=act_enc)

#Let's take a look at the environment
env.reset()

#Make a pRNN
num_neurons = 500
pRNNtype = ARCHITECTURE_NAME #This will train a 5-step masked pRNN. 
                        #For a rollout network use 'thcycRNN_5win_full'

predictiveNet = PredictiveNet(env, hidden_size=num_neurons, pRNNtype=pRNNtype)

#specify an action policy (agent)
action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
agent = RandomActionAgent(env.action_space, action_probability)

#run a sample trajectory (note: predictions will be garbage, agent is untrained)
predictiveNet.plotSampleTrajectory(env,agent)
plt.savefig(f"plots/{ARCHITECTURE_NAME}_naive.png")

#Run one training epoch of 500 trials, each 500 steps long
sequence_duration = 50 # (500)
num_trials = 50 #0 (500)

predictiveNet.trainingEpoch(env, agent,
                            sequence_duration=sequence_duration,
                            num_trials=num_trials)
                        
#run a sample trajectory. did the predictions get better?
predictiveNet.plotSampleTrajectory(env,agent)
plt.savefig(f"plots/{ARCHITECTURE_NAME}_trained.png")


#Let's take a look at the spatial position decoding and tuning curves 
place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(env,agent,
                                                trainDecoder=True, saveTrainingData=True,calculatesRSA = True)

predictiveNet.calculateDecodingPerformance(env,agent,decoder)
predictiveNet.plotTuningCurvePanel()
plt.savefig(f"plots/{ARCHITECTURE_NAME}_tuning_curve.png")