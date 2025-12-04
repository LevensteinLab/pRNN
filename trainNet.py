#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 22:07:04 2022

@author: dl2820
"""


#%%
from prnn.utils.predictiveNet import PredictiveNet
from prnn.utils.agent import create_agent
from prnn.utils.data import create_dataloader
from prnn.utils.env import make_env
from prnn.utils.figures import TrainingFigure
from prnn.utils.figures import SpontTrajectoryFigure
from prnn.analysis.OfflineTrajectoryAnalysis import OfflineTrajectoryAnalysis
import argparse
from tqdm import tqdm

#TODO: get rid of these dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

# Parse arguments

parser = argparse.ArgumentParser()

## General parameters

parser.add_argument("--env",
                    default='MiniGrid-LRoom-18x18-v0',
                    # default='RiaB-LRoom',
                    help="name of the environment to train on (Default: MiniGrid-LRoom-18x18-v0, for RiaB: RiaB-LRoom)")

parser.add_argument("--agent",
                    default='RandomActionAgent',
                    # default='RatInABoxAgent',
                    help="name of the agent for environment exploration (Default: RandomActionAgent, other option: RatInABoxAgent)")

parser.add_argument("--envPackage",
                    default='gym-minigrid',
                    # default='ratinabox_remix',
                    help="which package the environment comes from? (Default: gym-minigrid; other options: farama-minigrid, ratinabox, ratinabox_remix)")

parser.add_argument("--pRNNtype", default='Masked',
                    help="Which pRNN type?")

parser.add_argument("--savefolder",
                    default='',
                    # default='riab_test_newrepo',
                    help="Where to save the net? (foldername/)")

parser.add_argument("--loadfolder", default='',
                    help="Where to load the net? (foldername/)")

parser.add_argument("--numepochs",
                    default=50,
                    type=int,
                    help="how many training epochs? (Default: 50)")

parser.add_argument("--seqdur", default=500, type=int,
                    help="how long is each behavioral sequence? (Default: 500")

parser.add_argument("--numtrials", default=1024, type=int,
                    help="How many trials in an epoch? Best if divisible by batch size (Default: 1024")

parser.add_argument("--hiddensize", default=500, type=int,
                    help="how many hidden units? (Default: 500")

parser.add_argument("-c", "--contin", action="store_true",
                    help="Continue previous training?")

parser.add_argument("--load_env", default=-1, type=int,
                    help="Load Environment for continued Training. Specify unique env id")

parser.add_argument("-s", "--seed", default=8, type=int,
                    help="Random Seed? (Default: 8)")

parser.add_argument("--lr", default=3e-3, type=float,    #former default:2e-4 (not relative)
                    help="Learning Rate? (Relative to init sqrt(1/k) for each layer) (Default: 3e-3)")

parser.add_argument("--weight_decay", default=3e-3, type=float, #former default:6e-7 (not relative)
                    help="Weight Decay? (Relative to learning rate) (Default: 3e-3)")

parser.add_argument("--ntimescale", default=2, type=float,
                    help="Neural timescale (Default: 2 timesteps)")

parser.add_argument("--dropout", default=0.15, type=float,
                    help="Dropout probability (Default: 0.15)")

parser.add_argument("--noisemean", default=0, type=float,
                    help="Mean offset for internal noise (Default: 0)")

parser.add_argument("--noisestd", default=0.03, type=float,
                    help="Std of internal noise (Default: 0.03)")

parser.add_argument('--trainBias', action='store_true', default=True)

parser.add_argument("--namext", default='',
                    help="Extension to the savename?")

parser.add_argument("--actenc",
                    default='OnehotHD',
                    # default='ContSpeedOnehotHD',
                    help="Action encoding, options: OnehotHD (default),SpeedHD, Onehot, Velocities, \
                        Continuous, ContSpeedRotation, ContSpeedHD, ContSpeedOnehotHD")

parser.add_argument('--saveTrainData', action='store_true', default=True)
parser.add_argument('--no-saveTrainData', dest='saveTrainData', action='store_false')

parser.add_argument('--withDataLoader', action='store_true', default=True)
parser.add_argument('--noDataLoader', dest='withDataLoader', action='store_false')

parser.add_argument("--datadir",
                    default='Data',
                    help="Top-level folder to save the data for DataLoader (the sub-folders will be \
                        created automatically for each individual env name)")

parser.add_argument("--dataNtraj", default=10240, type=int,
                    help="Number of trajectories in the DataLoader (Default: 10240)")

parser.add_argument("--batchsize", default=16, type=int,
                    help="Number of trajectories in the DataLoader output batch (Default: 16)")

parser.add_argument("--numworkers", default=1, type=int,
                    help="Number of dataloader workers (Default: 1)")

# Additional architecture kwargs

parser.add_argument("--useLN", default=True, type =bool, 
                    help="Use LayerNorm?")

parser.add_argument("--useFF", default=False, type=bool,
                    help="Make network Feed Forward only?")

parser.add_argument("--maskActions", default=False, type=bool,
                    help="Mask actions from model input as well?")

parser.add_argument("--actOffset", default=0, type=int,
                    help="Number of timesteps to offset actions by (backwards)")

parser.add_argument("--inMaskLength", default=0, type=int,
                    help="Number of future timesteps to mask")

parser.add_argument("--useALN", default=False, type=bool,
                    help="Use AdaptiveLayerNorm?")

parser.add_argument("--k", default=5, type=int,
                    help="Number of predictions in rollout")

parser.add_argument("--rolloutAction", default="first", type=str,
                    help="Action structure")

parser.add_argument("--continuousTheta", default=False, type=bool,
                    help="Carry over hidden state from the kth rollout to the t+1'th timestep?")


args = parser.parse_args()



savename = args.pRNNtype + '-' + args.namext + '-s' + str(args.seed)
figfolder = 'nets/'+args.savefolder+'/trainfigs/'+savename
analysisfolder = 'nets/'+args.savefolder+'/analysis/'+savename




#%%
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.contin: #continue previous training, so load net from folder
    predictiveNet = PredictiveNet.loadNet(args.loadfolder+savename)
    if args.env == '':
        env = predictiveNet.loadEnvironment(args.load_env)
        predictiveNet.addEnvironment(env)
    else:
        env = make_env(args.env, args.envPackage, args.actenc, args.agentspeed,
                       args.thigmotaxis, args.HDbins)
        predictiveNet.addEnvironment(env)
    agent = create_agent(args.env, env, args.agent)
else: #create new PredictiveNet and begin training
    env = make_env(args.env, args.envPackage, args.actenc, args.agentspeed,
                   args.thigmotaxis, args.HDbins)
    agent = create_agent(args.env, env, args.agent)
    predictiveNet = PredictiveNet(env,
                                  hidden_size = args.hiddensize,
                                  pRNNtype = args.pRNNtype,
                                  learningRate = args.lr,
                                  weight_decay = args.weight_decay,
                                  trainNoiseMeanStd = (args.noisemean,args.noisestd),
                                  trainBias = args.trainBias,
                                  bias_lr = args.bias_lr,
                                  identityInit = args.identityInit,
                                  dataloader = args.withDataLoader,
                                  f = args.sparsity,
                                  dropp = args.dropout,
                                  neuralTimescale = args.ntimescale,
                                  bptttrunc = args.bptttrunc)
    predictiveNet.seed = args.seed
    predictiveNet.trainArgs = args
    predictiveNet.plotSampleTrajectory(env,agent,
                                       savename=savename+'exTrajectory_untrained',
                                       savefolder=figfolder)
    predictiveNet.savefolder = args.savefolder
    predictiveNet.savename = savename


    if args.withDataLoader:
        # Separate Data Loader should be created for every environment
        create_dataloader(env, agent, args.dataNtraj, args.seqdur,
                          args.datadir, generate=True,
                          tmp_folder=os.path.expandvars('${SLURM_TMPDIR}'),
                          batch_size=args.batchsize, 
                          num_workers=args.numworkers)
        predictiveNet.useDataLoader = args.withDataLoader


#%% Training Epoch
#Consider these as "trainingparameters" class/dictionary
numepochs = args.numepochs
sequence_duration = args.seqdur
num_trials = args.numtrials
if args.withDataLoader:
    batchsize = args.batchsize
else:
    batchsize = 1

predictiveNet.trainingCompleted = False
if predictiveNet.numTrainingTrials == -1:
    #Calculate initial spatial metrics etc
    print('Training Baseline')
    predictiveNet.useDataLoader = False
    predictiveNet.trainingEpoch(env, agent,
                            sequence_duration=sequence_duration,
                            num_trials=1)
    predictiveNet.useDataLoader = args.withDataLoader
    print('Calculating Spatial Representation...')
    place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(env,agent,
                                                  trainDecoder=True,saveTrainingData=True,
                                                  bitsec= False,
                                                  calculatesRSA = True, sleepstd=0.03)
    predictiveNet.plotTuningCurvePanel(savename=savename,savefolder=figfolder)
    print('Calculating Decoding Performance...')
    predictiveNet.calculateDecodingPerformance(env,agent,decoder,
                                                savename=savename, savefolder=figfolder,
                                                saveTrainingData=True)
    #predictiveNet.plotDelayDist(env, agent, decoder)

if hasattr(predictiveNet, 'numTrainingEpochs') is False:
    predictiveNet.numTrainingEpochs = int(predictiveNet.numTrainingTrials/num_trials)

progress = tqdm(total=numepochs, desc="Training Epochs") #tdqm status bar

while predictiveNet.numTrainingEpochs<numepochs: #run through all epochs
    print(f'Training Epoch {predictiveNet.numTrainingEpochs}')
    predictiveNet.trainingEpoch(env, agent,
                            sequence_duration=sequence_duration,
                            num_trials=num_trials,
                            batch_size=batchsize)
    print('Calculating Spatial Representation...')
    place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(env,agent,
                                                 trainDecoder=True, trainHDDecoder = True,
                                                 saveTrainingData=True, bitsec= False,
                                                 calculatesRSA = True, sleepstd=0.03)
    print('Calculating Decoding Performance...')
    predictiveNet.calculateDecodingPerformance(env,agent,decoder,
                                                savename=savename, savefolder=figfolder,
                                                saveTrainingData=True)
    predictiveNet.plotLearningCurve(savename=savename,savefolder=figfolder,
                                    incDecode=True)
    predictiveNet.plotTuningCurvePanel(savename=savename,savefolder=figfolder)
    plt.show()
    plt.close('all')
    predictiveNet.saveNet(args.savefolder+savename)

    progress.update(1)

progress.close()

predictiveNet.trainingCompleted = True
TrainingFigure(predictiveNet,savename=savename,savefolder=figfolder)

#If the user doesn't want to save all that training data, delete it except the last one
if args.saveTrainData is False:
    predictiveNet.TrainingSaver = predictiveNet.TrainingSaver.drop(predictiveNet.TrainingSaver.index[:-1])
    predictiveNet.saveNet(args.savefolder+savename)