import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prnn.utils.predictiveNet import PredictiveNet
from prnn.utils.agent import RandomActionAgent
from prnn.analysis.SpatialTuningAnalysis import SpatialTuningAnalysis
from prnn.analysis.representationalGeometryAnalysis import representationalGeometryAnalysis
from prnn.analysis.OfflineTrajectoryAnalysis import OfflineTrajectoryAnalysis

savefolder = 'BasicAnalysisFigs'

#Example Net
netname = 'Masked'
netfolder = '/maskedk_panel/'
exseed = 8
#predictiveNet = PredictiveNet.loadNet(netfolder+netname+'-SpeedHD-s'+str(exseed))
predictiveNet = PredictiveNet.loadNet(netfolder+netname+'--s'+str(exseed))

env = predictiveNet.EnvLibrary[0]
agentname = 'RandomActionAgent'
action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
agent = RandomActionAgent(env.action_space,action_probability)
place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(env,agent,
                                             trainDecoder=True)

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

b_adapt = 1
tau_adapt=100
OTA_adapt = OfflineTrajectoryAnalysis(predictiveNet, noisestd=sleepnoise,
                                   withIsomap=False, decoder=decoder, 
                                      withAdapt=True, b_adapt = b_adapt, tau_adapt=tau_adapt,
                                       compareWake=True)

OTA_adapt.SpontTrajectoryFigure('adaptation',savefolder, trajRange=(150,250))


OTA_query = OfflineTrajectoryAnalysis(predictiveNet, noisemag = 0, noisestd=sleepnoise,
                               withIsomap=False, decoder=decoder,
                                     actionAgent=True,
                               compareWake=True)

OTA_query.SpontTrajectoryFigure('actionquery',savefolder, trajRange=(110,150))