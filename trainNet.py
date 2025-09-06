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

#TODO: get rid of these dependencies
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

parser.add_argument("--pRNNtype", default='thRNN_2win',
                    help="which pRNN (Default: thRNN_2win)")

parser.add_argument("--savefolder",
                    default='',
                    # default='riab_test_newrepo',
                    help="Where to save the net? (foldername/)")

parser.add_argument("--loadfolder", default='',
                    help="Where to load the net? (foldername/)")

parser.add_argument("--numepochs",
                    default=80,
                    # default=1,
                    type=int,
                    help="how many training epochs? (Default: 80)")

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

parser.add_argument("--bptttrunc", default=1e8, type=int,
                    help="BPTT Truncation window? (Default: 1e8)")

parser.add_argument("--ntimescale", default=2, type=float,
                    help="Neural timescale (Default: 2 timesteps)")

parser.add_argument("--dropout", default=0.15, type=float,
                    help="Dropout probability (Default: 0.15)")

parser.add_argument("--noisemean", default=0, type=float,
                    help="Mean offset for internal noise (Default: 0)")

parser.add_argument("--noisestd", default=0.03, type=float,
                    help="Std of internal noise (Default: 0.03)")

parser.add_argument("-f", "--sparsity", default=0.5, type=float,
                    help="Activation sparsity (via layer norm, irrelevant for non-LN networks) (Default: 0.5)")

parser.add_argument('--trainBias', action='store_true', default=False)

parser.add_argument("--bias_lr", default=1, type=float,    #former default:2e-4 (not relative)
                     help="Bias Learning Rate? (Relative to learning rate) (Default: 1)")

parser.add_argument('--identityInit', action='store_true', default=False)

parser.add_argument("--agentspeed", default=0.2, type=float,
                    help="Average speed of the agent in a continuous environment (Default: 0.2)")

parser.add_argument("--thigmotaxis", default=0.2, type=float,
                    help="Agent bias towards exploring locations near the walls (RiaB) (Default: 0.2)")

parser.add_argument("--HDbins", default=12, type=int,
                    help="Number of bins for HD signal (RiaB) (Default: 12)")

parser.add_argument("--namext", default='',
                    help="Extension to the savename?")

parser.add_argument("--actenc",
                    default='OneHotHD',
                    # default='ContSpeedOneHotHD',
                    help="Action encoding, options: OneHotHD (default),SpeedHD, OneHot, Velocities, \
                        Continuous, ContSpeedRotation, ContSpeedHD, ContSpeedOneHotHD")

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

parser.add_argument("--batchsize", default=1, type=int,
                    help="Number of trajectories in the DataLoader output batch (Default: 1)")

parser.add_argument("--numworkers", default=1, type=int,
                    help="Number of dataloader workers (Default: 1)")


args = parser.parse_args()



savename = args.pRNNtype + '-' + args.namext + '-s' + str(args.seed)
figfolder = 'nets/'+args.savefolder+'/trainfigs/'+savename
analysisfolder = 'nets/'+args.savefolder+'/analysis/'+savename




#%%
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.contin:
    predictiveNet = PredictiveNet.loadNet(args.loadfolder+savename)
    if args.env == '':
        env = predictiveNet.loadEnvironment(args.load_env)
        predictiveNet.addEnvironment(env)
    else:
        env = make_env(args.env, args.envPackage, args.actenc, args.agentspeed,
                       args.thigmotaxis, args.HDbins)
        predictiveNet.addEnvironment(env)
    agent = create_agent(args.env, env, args.agent)
else:
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
                          args.datadir, args.batchsize, args.numworkers)
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
    predictiveNet.plotDelayDist(env, agent, decoder)

#TODO: Put in time counter here and ETA...
#TODO: take this out later. for backwards compatibility
if hasattr(predictiveNet, 'numTrainingEpochs') is False:
    predictiveNet.numTrainingEpochs = int(predictiveNet.numTrainingTrials/num_trials)
    
while predictiveNet.numTrainingEpochs<numepochs:
    print(f'Training Epoch {predictiveNet.numTrainingEpochs}')
    predictiveNet.trainingEpoch(env, agent,
                            sequence_duration=sequence_duration,
                            num_trials=num_trials,)
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
    #predictiveNet.plotSampleTrajectory(env,agent,savename=savename,savefolder=figfolder)
    predictiveNet.plotTuningCurvePanel(savename=savename,savefolder=figfolder)
    #SpontTrajectoryFigure(predictiveNet,decoder,noisestd=0.2,noisemag=0,
    # #                      savename=savename, savefolder=figfolder)
    # OTA = OfflineTrajectoryAnalysis(predictiveNet, actionAgent=agent, noisestd=0.03,
    #                                 withTransitionMaps=not env.continuous, wakeAgent=agent,
    #                                 decoder=decoder, calculateViewSimilarity=True)
    # OTA.SpontTrajectoryFigure(savename+'_query', figfolder)
    # predictiveNet.addTrainingData('replay_alpha', OTA.diffusionFit['alpha'])
    # predictiveNet.addTrainingData('replay_int', OTA.diffusionFit['intercept'])
    # predictiveNet.addTrainingData('replay_view', OTA.ViewSimilarity['meanstd_sleep'][0][0])
    
    # OTA = OfflineTrajectoryAnalysis(predictiveNet, noisestd=0.03,
    #                            decoder=decoder, calculateViewSimilarity=True,
    #                            wakeAgent=agent, withAdapt=True,
    #                            b_adapt = 0.3, tau_adapt=8)
    # OTA.SpontTrajectoryFigure(savename+'_adapt',figfolder)
    # predictiveNet.addTrainingData('replay_alpha_adapt',OTA.diffusionFit['alpha'])
    # predictiveNet.addTrainingData('replay_int_adapt',OTA.diffusionFit['intercept'])
    # predictiveNet.addTrainingData('replay_view_adapt',OTA.ViewSimilarity['meanstd_sleep'][0][0])


    # plt.show()
    plt.close('all')
    predictiveNet.saveNet(args.savefolder+savename)

predictiveNet.trainingCompleted = True
TrainingFigure(predictiveNet,savename=savename,savefolder=figfolder)

#If the user doesn't want to save all that training data, delete it except the last one
if args.saveTrainData is False:
    predictiveNet.TrainingSaver = predictiveNet.TrainingSaver.drop(predictiveNet.TrainingSaver.index[:-1])
    predictiveNet.saveNet(args.savefolder+savename)