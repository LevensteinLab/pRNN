#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 22:07:04 2022

@author: dl2820
"""

# %%
from prnn.utils.predictiveNet import PredictiveNet, CELL_TYPES
from prnn.utils.agent import create_agent
from prnn.utils.data import create_dataloader
from prnn.utils.env import make_env
from prnn.utils.figures import TrainingFigure
from prnn.utils.figures import SpontTrajectoryFigure
from prnn.analysis.OfflineTrajectoryAnalysis import OfflineTrajectoryAnalysis
import argparse
from tqdm import tqdm

# TODO: get rid of these dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from types import SimpleNamespace


# Parse arguments

parser = argparse.ArgumentParser()

## General parameters

parser.add_argument(
    "--env",
    default="LRoom-18x18-v0",
    # default='RiaB-LRoom',
    help="name of the environment to train on (Default: LRoom-18x18-v0 for gym-minigrid: MiniGrid-LRoom-18x18-v0, for RiaB: RiaB-LRoom)",
)

parser.add_argument(
    "--agent",
    default="RandomActionAgent",
    # default='RatInABoxAgent',
    help="name of the agent for environment exploration (Default: RandomActionAgent, other option: RatInABoxAgent)",
)

parser.add_argument(
    "--envPackage",
    default="farama-minigrid",
    # default='ratinabox_remix',
    help="which package the environment comes from? (Default: farama-minigrid; other options: gym-minigrid, ratinabox, ratinabox_remix)",
)

parser.add_argument("--pRNNtype", default="Masked", help="Which pRNN type")

parser.add_argument(
    "--cell",
    type=str,
    default=None,
    choices=list(CELL_TYPES.keys()),
    help="Override default cell type",
)

parser.add_argument(
    "--savefolder",
    default="",
    # default='riab_test_newrepo',
    help="Where to save the net? (foldername/)",
)

parser.add_argument("--loadfolder", default="", help="Where to load the net? (foldername/)")

parser.add_argument(
    "--numepochs", default=50, type=int, help="how many training epochs? (Default: 50)"
)

parser.add_argument(
    "--seqdur", default=500, type=int, help="how long is each behavioral sequence? (Default: 500"
)

parser.add_argument(
    "--numtrials",
    default=1024,
    type=int,
    help="How many trials in an epoch? Best if divisible by batch size (Default: 1024",
)

parser.add_argument(
    "--hidden_size", default=500, type=int, help="how many hidden units? (Default: 500"
)

parser.add_argument(
    "-c", "--contin", default=False, action="store_true", help="Continue previous training?"
)

parser.add_argument(
    "--load_env",
    default=-1,
    type=int,
    help="Load Environment for continued Training. Specify unique env id",
)

parser.add_argument("-s", "--seed", default=8, type=int, help="Random Seed? (Default: 8)")

parser.add_argument("--bptttrunc", default=None, type=float)

parser.add_argument(
    "--lr",
    default=2e-3,
    type=float,  # former default:2e-4 (not relative)
    help="Learning Rate? (Relative to init sqrt(1/k) for each layer) (Default: 2e-3)",
)

parser.add_argument(
    "--weight_decay",
    default=3e-3,
    type=float,  # former default:6e-7 (not relative)
    help="Weight Decay? (Relative to learning rate) (Default: 3e-3)",
)

parser.add_argument(
    "--neuralTimescale", default=2, type=float, help="Neural timescale (Default: 2 timesteps)"
)

parser.add_argument(
    "--dropout", default=0.15, type=float, help="Dropout probability (Default: 0.15)"
)

parser.add_argument(
    "--noisemean", default=0, type=float, help="Mean offset for internal noise (Default: 0)"
)

parser.add_argument(
    "--noisestd", default=0.03, type=float, help="Std of internal noise (Default: 0.03)"
)

parser.add_argument("--trainBias", action="store_true", default=True)

parser.add_argument("--namext", default="", help="Extension to the savename?")

parser.add_argument(
    "--actenc",
    default="SpeedHD",
    # default='ContSpeedOnehotHD',
    help="Action encoding, options: OneHotHD (default),SpeedHD, OneHot, Velocities, \
                        Continuous, ContSpeedRotation, ContSpeedHD, ContSpeedOnehotHD",
)

parser.add_argument("--saveTrainData", action="store_true", default=True)
parser.add_argument("--no-saveTrainData", dest="saveTrainData", action="store_false")

parser.add_argument("--withDataLoader", action="store_true", default=True)
parser.add_argument("--noDataLoader", dest="withDataLoader", action="store_false")

parser.add_argument(
    "--datadir",
    default="Data",
    help="Top-level folder to save the data for DataLoader (the sub-folders will be \
                        created automatically for each individual env name)",
)

parser.add_argument(
    "--dataNtraj",
    default=10240,
    type=int,
    help="Number of trajectories in the DataLoader (Default: 10240)",
)

parser.add_argument(
    "--batchsize",
    default=16,
    type=int,
    help="Number of trajectories in the DataLoader output batch (Default: 16)",
)

parser.add_argument(
    "--numworkers", default=1, type=int, help="Number of dataloader workers (Default: 1)"
)

# Additional architecture kwargs

parser.add_argument(
    "--use_FF",
    action="store_const",
    const=True,
    help="Use feedforward network (disable recurrence)",
)  # default is false if flag is not used and does not compromise partial use_FF flags

parser.add_argument(
    "--mask_actions",
    action="store_const",
    const=True,
    help="Mask actions from model input as well?",
)  # similar to use_FF, if flag is not used then defaults to false and wont compromise partial class flags

parser.add_argument(
    "--actOffset",
    default=None,  # defined at the predictiveNet level to avoid overriding partials
    type=int,
    help="Number of timesteps to offset actions by (backwards)",
)

parser.add_argument(
    "--k",
    default=None,  # defined at the predictiveNet level to avoid overriding partials
    type=int,
    help="Number of predictions; i.e. number of future timesteps to mask or number of rollouts",
)

parser.add_argument(
    "--rollout_action",
    default=None,  # defaults to None to avoid overriding partials
    type=str,
    help="Action structure, options first, hold, full",
)

parser.add_argument("--continuousTheta", action="store_true", dest="continuousTheta")
parser.add_argument("--no-continuousTheta", action="store_false", dest="continuousTheta")
parser.set_defaults(continuousTheta=None)

# weight init parameters
parser.add_argument(
    "--init",
    default=None,
    type=str,
    help="Weight initialization schema to use, options: xavier, log_normal, gamma",
)
# log_normal init parameters
parser.add_argument(
    "--sparsity",
    default=None,
    type=float,
    help="Activation sparsity (via layer norm, irrelevant for non-LN networks) (Default: 0.5)",
)

parser.add_argument(
    "--mean_std_ratio",
    default=None,
    type=float,
    help="Mean:STD ratio of the log normal distrubtion of the intialized weights, only used if init==log_normak",
)
# gamma init
parser.add_argument(
    "--alpha",
    default=None,
    type=float,
    help="Shape (alpha or k) of the gamma disitrbution of the initialized weights, only used if init==gamma",
)
parser.add_argument(
    "--theta",
    default=None,
    type=float,
    help="Scale (theta or beta) of the gamma disitrbution of the initialized weights, only used if init==gamma",
)
# EG params

parser.add_argument(
    "--eg_weight_decay",
    default=5e-6,
    type=float,
    help="Weight Decay for Exponentiated Gradient Descent (Default: 1e-6)",
)

parser.add_argument(
    "--reg_lambda_h",
    default=0.0,
    type=float,
    help="Activity regularization strength: loss += reg_lambda_h * mean(|h|^reg_p) (Default: 0.0)",
)

parser.add_argument(
    "--eg_lr",
    default=None,
    type=float,
    help="Learning Rate for Exponentiated Gradient Descent (Default: None (do not use EG))",
)

parser.add_argument(
    "--bias_lr",
    default=0.1,
    type=float,
    help="Learning Rate for Biases",
)

# DivNorm Params

parser.add_argument(
    "--target_mean",
    default=0.7,
    type=float,
    help="Target mean for activations for divisive normalization RNN",
)

parser.add_argument(
    "--train_divnorm",
    action="store_true",
    default=False,
    help="Whether to train the divisive normalization parameters: k_div and sigma",
)

parser.add_argument(
    "--k_div", default=1, type=float, help="Parameter in divisive normalization"
)  # TODO: add more descriptive language here

parser.add_argument(
    "--sigma", default=1, type=float, help="Parameter in divisive normalization"
)  # TODO: add more descriptive language here

parser.add_argument(
    "--test",
    action="store_true",
    default=False,
    help="for running test code",
)

args = parser.parse_args()


savename = args.pRNNtype + "-" + args.namext + "-s" + str(args.seed)
figfolder = "nets/" + args.savefolder + "/trainfigs/" + savename
analysisfolder = "nets/" + args.savefolder + "/analysis/" + savename


# === Always include (universal parameters) ===
architecture_kwargs = {
    "dropp": args.dropout,
    "neuralTimescale": args.neuralTimescale,
    "bptttrunc": args.bptttrunc,
}
architecture_kwargs = {k: v for k, v in architecture_kwargs.items() if v is not None}


# === Conditionally include (architecture-specific with None check) ===

# Boolean flags
if args.use_FF is not None:
    architecture_kwargs["use_FF"] = args.use_FF

if args.mask_actions is not None:
    architecture_kwargs["mask_actions"] = args.mask_actions

if args.continuousTheta is not None:
    architecture_kwargs["continuousTheta"] = args.continuousTheta

# Numeric parameters
if args.k is not None:
    architecture_kwargs["k"] = args.k

if args.actOffset is not None:
    architecture_kwargs["actOffset"] = args.actOffset

if args.rollout_action is not None:
    architecture_kwargs["rollout_action"] = args.rollout_action

# Weight initialization
if args.init is not None:
    architecture_kwargs["init"] = args.init

if args.sparsity is not None:
    architecture_kwargs["sparsity"] = args.sparsity

if args.mean_std_ratio is not None:
    architecture_kwargs["mean_std_ratio"] = args.mean_std_ratio

if args.alpha is not None:
    architecture_kwargs["alpha"] = args.alpha

if args.theta is not None:
    architecture_kwargs["theta"] = args.theta

# === Always include (DivNorm parameters) - they aren't harmful ===
architecture_kwargs["target_mean"] = args.target_mean
architecture_kwargs["train_divnorm"] = args.train_divnorm
architecture_kwargs["k_div"] = args.k_div
architecture_kwargs["sigma"] = args.sigma

# === Cell override ===
if args.cell is not None:
    if args.cell not in CELL_TYPES.keys():
        raise ValueError(
            f"Cell type '{args.cell}' not recognized. "
            f"Available cell types: {list(CELL_TYPES.keys())}"
        )
    architecture_kwargs["cell"] = CELL_TYPES[
        args.cell
    ]  # this way default setting for cell will used for each pRNNtype unless overridden here

# %%
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.contin:  # continue previous training, so load net from folder
    predictiveNet = PredictiveNet.loadNet(args.loadfolder + savename)
    if args.env == "":
        env = predictiveNet.loadEnvironment(args.load_env)
        predictiveNet.addEnvironment(env)
    else:
        env = make_env(args.env, args.envPackage, args.actenc)
        predictiveNet.addEnvironment(env)
    agent = create_agent(args.env, env, args.agent)
else:  # create new PredictiveNet and begin training
    env = make_env(args.env, args.envPackage, args.actenc)
    agent = create_agent(args.env, env, args.agent)
    predictiveNet = PredictiveNet(
        env,
        hidden_size=args.hidden_size,
        pRNNtype=args.pRNNtype,
        learningRate=args.lr,
        weight_decay=args.weight_decay,
        trainNoiseMeanStd=(args.noisemean, args.noisestd),
        trainBias=args.trainBias,
        eg_lr=args.eg_lr,
        eg_weight_decay=args.eg_weight_decay,
        bias_lr=args.bias_lr,
        dataloader=args.withDataLoader,
        trainArgs=SimpleNamespace(**args.__dict__),
        **architecture_kwargs,
    )  # allows values in trainArgs to be accessible

    # predictiveNet.seed = args.seed
    # predictiveNet.trainArgs = args
    predictiveNet.plotSampleTrajectory(
        env, agent, savename=savename + "exTrajectory_untrained", savefolder=figfolder
    )
    # predictiveNet.savefolder = args.savefolder
    # predictiveNet.savename = savename

    if args.withDataLoader:
        # Separate Data Loader should be created for every environment
        create_dataloader(
            env,
            agent,
            args.dataNtraj,
            args.seqdur,
            args.datadir,
            generate=True,
            #   tmp_folder=os.path.expandvars('${SLURM_TMPDIR}'), # This should stay commented out unless running on MILA cluster
            batch_size=args.batchsize,
            num_workers=args.numworkers,
        )
        predictiveNet.useDataLoader = args.withDataLoader


# %% Training Epoch
# Consider these as "trainingparameters" class/dictionary
if args.test:
    predictiveNet.saveNet(args.savefolder + savename)
else:
    numepochs = args.numepochs
    sequence_duration = args.seqdur
    num_trials = args.numtrials
    if args.withDataLoader:
        batchsize = args.batchsize
    else:
        batchsize = 1

    predictiveNet.trainingCompleted = False
    if predictiveNet.numTrainingTrials == -1:
        # Calculate initial spatial metrics etc
        print("Training Baseline")
        predictiveNet.useDataLoader = False
        predictiveNet.trainingEpoch(env, agent, sequence_duration=sequence_duration, num_trials=1)
        predictiveNet.useDataLoader = args.withDataLoader
        print("Calculating INITIAL Spatial Representation...")
        place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(
            env,
            agent,
            trainDecoder=True,
            saveTrainingData=True,
            bitsec=False,
            calculatesRSA=True,
            sleepstd=0.03,
        )
        predictiveNet.plotTuningCurvePanel(savename=savename, savefolder=figfolder)
        print("Calculating INITIAL Decoding Performance...")
        predictiveNet.calculateDecodingPerformance(
            env, agent, decoder, savename=savename, savefolder=figfolder, saveTrainingData=True
        )
        # predictiveNet.plotDelayDist(env, agent, decoder)

    if hasattr(predictiveNet, "numTrainingEpochs") is False:
        predictiveNet.numTrainingEpochs = int(predictiveNet.numTrainingTrials / num_trials)

    progress = tqdm(total=numepochs, desc="Training Epochs")  # tdqm status bar

    while predictiveNet.numTrainingEpochs < numepochs:  # run through all epochs
        print(f"Training Epoch {predictiveNet.numTrainingEpochs}")
        predictiveNet.trainingEpoch(
            env, agent, sequence_duration=sequence_duration, num_trials=num_trials
        )
        print("Calculating Spatial Representation...")
        place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(
            env,
            agent,
            trainDecoder=True,
            trainHDDecoder=False,
            saveTrainingData=True,
            bitsec=False,
            calculatesRSA=True,
            sleepstd=0.03,
        )
        print("Calculating Decoding Performance...")
        predictiveNet.calculateDecodingPerformance(
            env, agent, decoder, savename=savename, savefolder=figfolder, saveTrainingData=True
        )
        predictiveNet.plotLearningCurve(savename=savename, savefolder=figfolder, incDecode=True)
        predictiveNet.plotTuningCurvePanel(savename=savename, savefolder=figfolder)
        plt.show()
        plt.close("all")
        predictiveNet.saveNet(args.savefolder + savename)

        progress.update(1)

    progress.close()

    predictiveNet.trainingCompleted = True
    TrainingFigure(predictiveNet, savename=savename, savefolder=figfolder)

# If the user doesn't want to save all that training data, delete it except the last one
if args.saveTrainData is False:
    predictiveNet.TrainingSaver = predictiveNet.TrainingSaver.drop(
        predictiveNet.TrainingSaver.index[:-1]
    )
    predictiveNet.saveNet(args.savefolder + savename)
