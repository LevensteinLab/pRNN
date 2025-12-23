import pandas as pd
import numpy as np
from prnn.utils.predictiveNet import PredictiveNet
from prnn.utils.agent import RandomActionAgent
import itertools
import torch
import random
from prnn.utils.env import make_env
from prnn.utils.general import saveFig
from prnn.utils.figures import TrainingFigure
import matplotlib.pyplot as plt
from prnn.analysis.SpatialTuningAnalysis import SpatialTuningAnalysis
from prnn.analysis.OfflineTrajectoryAnalysis import OfflineTrajectoryAnalysis
from prnn.analysis.representationalGeometryAnalysis import representationalGeometryAnalysis

savefolder = 'BasicAnalysisFigs'

#Example Net
netname = 'thRNN_5win'
netfolder = '/maskedk_panel/'
exseed = 102
predictiveNet = PredictiveNet.loadNet(netfolder+netname+'-SpeedHD-s'+str(exseed))

env = predictiveNet.EnvLibrary[0]
agentname = 'RandomActionAgent'
action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
agent = RandomActionAgent(env.action_space,action_probability)
place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(env,agent,
                                             trainDecoder=True, trainHDDecoder = True)

predictiveNet.calculateDecodingPerformance(env,agent,decoder,
                                            savename=netname, savefolder=savefolder,
                                          trajectoryWindow=5,
                                          timesteps=1000)

STA = SpatialTuningAnalysis(predictiveNet,inputControl=True, untrainedControl=True)

STA.TCExamplesFigure(netname,savefolder)

sleepnoise = 0.03
isomap_neighbors = 15
RGA = representationalGeometryAnalysis(predictiveNet, noisestd=sleepnoise,
                                       withIsomap=True, n_neighbors = isomap_neighbors)

RGA.WakeSleepFigure(netname,savefolder)

predictiveNet.pRNN.rnn.cell


b_adapt = 1
tau_adapt=100
OTA_adapt = OfflineTrajectoryAnalysis(predictiveNet, noisestd=sleepnoise,
                                   withIsomap=False, decoder=decoder, 
                                      withAdapt=True, b_adapt = b_adapt, tau_adapt=tau_adapt,
                                      calculateViewSimilarity=True,
                                       compareWake=True)

OTA_adapt.SpontTrajectoryFigure('adaptation',savefolder, trajRange=(150,250))
