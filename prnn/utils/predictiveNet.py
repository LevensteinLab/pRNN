#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 19:07:36 2021

@author: dl2820
"""
from torch import nn
import torch

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import pickle
import json
import time
import random
from pathlib import Path    


import pynapple as nap

from copy import deepcopy

from prnn.utils.general import saveFig
from prnn.utils.general import delaydist

from prnn.utils.LinearDecoder import linearDecoder

from prnn.utils.lossFuns import LPLLoss, predMSE, predRMSE

from prnn.analysis.representationalGeometryAnalysis import representationalGeometryAnalysis as RGA
from prnn.analysis.SpatialTuningAnalysis import SpatialTuningAnalysis as STA



#import timeit


from prnn.utils.Architectures import *

netOptions = {'vRNN' : vRNN,
              'RNN2L' : RNN2L,
              'vRNN_LayerNorm' : vRNN_LayerNorm,
              'thRNN_LayerNorm': thRNN_LayerNorm,
              'vRNN_LayerNormAdapt' : vRNN_LayerNormAdapt,
              'vRNN_CANN' : vRNN_CANN,
              'vRNN_CANN_FFonly' : vRNN_CANN_FFonly,
              'vRNN_adptCANN_FFonly' : vRNN_adptCANN_FFonly,
              'vRNN_0win'  :  vRNN_0win,
              'vRNN_1win'  :  vRNN_1win,
              'vRNN_2win'  :  vRNN_2win,
              'vRNN_3win'  :  vRNN_3win,
              'vRNN_4win'  :  vRNN_4win,
              'vRNN_5win'  :  vRNN_5win,
              'vRNN_1win_mask'  :  vRNN_1win_mask,
              'vRNN_2win_mask'  :  vRNN_2win_mask,
              'vRNN_3win_mask'  :  vRNN_3win_mask,
              'vRNN_4win_mask'  :  vRNN_4win_mask,
              'vRNN_5win_mask'  :  vRNN_5win_mask,
              'thRNN_0win'  :  thRNN_0win,
              'thRNN_1win'  :  thRNN_1win,
              'thRNN_2win' : thRNN_2win,
              'thRNN_3win' : thRNN_3win,
              'thRNN_4win' : thRNN_4win,
              'thRNN_5win' : thRNN_5win,
              'thRNN_6win' : thRNN_6win,
              'thRNN_7win' : thRNN_7win,
              'thRNN_8win' : thRNN_8win,
              'thRNN_9win' : thRNN_9win,
              'thRNN_10win' : thRNN_10win,
              'thRNN_1win_mask'  :  thRNN_1win_mask,
              'thRNN_2win_mask'  :  thRNN_2win_mask,
              'thRNN_3win_mask'  :  thRNN_3win_mask,
              'thRNN_4win_mask'  :  thRNN_4win_mask,
              'thRNN_5win_mask'  :  thRNN_5win_mask,
              'AutoencoderFF'  :  AutoencoderFF,
              'AutoencoderRec'  :  AutoencoderRec,
              'AutoencoderPred'  :  AutoencoderPred,
              'AutoencoderFFPred'  :  AutoencoderFFPred,
              'AutoencoderFF_LN'  :  AutoencoderFF_LN,
              'AutoencoderRec_LN'  :  AutoencoderRec_LN,
              'AutoencoderPred_LN'  :  AutoencoderPred_LN,
              'AutoencoderFFPred_LN'  :  AutoencoderFFPred_LN,
              'AutoencoderMaskedO'  :  AutoencoderMaskedO,
              'AutoencoderMaskedOA'  :  AutoencoderMaskedOA,
              'AutoencoderMaskedO_noout'  :  AutoencoderMaskedO_noout,
              'AutoencoderMaskedOA_noout'  :  AutoencoderMaskedOA_noout,
              'thcycRNN_3win' :  thcycRNN_3win,
              'thcycRNN_5win' :  thcycRNN_5win,
              'thcycRNN_5win_first' : thcycRNN_5win_first,
              'thcycRNN_5win_full' : thcycRNN_5win_full,
              'thcycRNN_5win_hold' : thcycRNN_5win_hold,
              'thcycRNN_5win_firstc' : thcycRNN_5win_firstc,
              'thcycRNN_5win_fullc' : thcycRNN_5win_fullc,
              'thcycRNN_5win_holdc' : thcycRNN_5win_holdc,
              'thcycRNN_5win_first_adapt' : thcycRNN_5win_first_adapt,
              'thcycRNN_5win_full_adapt' : thcycRNN_5win_full_adapt,
              'thcycRNN_5win_hold_adapt' : thcycRNN_5win_hold_adapt,
              'thcycRNN_5win_firstc_adapt' : thcycRNN_5win_firstc_adapt,
              'thcycRNN_5win_fullc_adapt' : thcycRNN_5win_fullc_adapt,
              'thcycRNN_5win_holdc_adapt' : thcycRNN_5win_holdc_adapt,
              'thRNN_0win_noLN'  :  thRNN_0win_noLN,
              'thRNN_1win_noLN'  :  thRNN_1win_noLN,
              'thRNN_2win_noLN' : thRNN_2win_noLN,
              'thRNN_3win_noLN' : thRNN_3win_noLN,
              'thRNN_4win_noLN' : thRNN_4win_noLN,
              'thRNN_5win_noLN' : thRNN_5win_noLN,
              'thRNN_6win_noLN' : thRNN_6win_noLN,
              }


lossOptions = {'predMSE'    :   predMSE,
               'LPL'        :   LPLLoss}

class PredictiveNet:
    """
    A predictive RNN architecture that takes observations and actions and
    returns observations at the next timestep

    Open questions:
        -Note: definition - observations are inputs taht are predicted, actions
                those that are not. How to deal with.... HD (action? But keep
                                                             in mind t vs t+1)
    """
    def __init__(self, env, pRNNtype='AutoencoderPred', hidden_size=500,
                 learningRate=2e-3, bias_lr=0.1,
                 regLambda=0, regOrder=1,
                 weight_decay=3e-3, losstype='predMSE', bptttrunc=100,
                 neuralTimescale=2, f=0.5,
                 dropp=0.15, trainNoiseMeanStd=(0,0.03),
                 target_rate=None, target_sparsity=None, decorrelate=False,
                 trainBias=True, identityInit=False, dataloader=False):
        """
        Initalize your predictive net. Requires passing an environment gym
        object that includes env.observation_space and env.action_space

        suppObs: any unpredicted observation key from the environment that is input and
        not predicted. Added to the action input
        """
        #Some defaults
        self.regLambda = regLambda
        self.regOrder = regOrder
        self.trainNoiseMeanStd = trainNoiseMeanStd

        #Set up the environmental I/O parms
        self.EnvLibrary = []
        self.env_shell = env
        self.act_size = env.getActSize()
        self.obs_size = env.getObsSize()
        self.addEnvironment(env)
        self.useDataLoader = dataloader

        #Set up the network and optimization stuff
        self.hidden_size = hidden_size
        self.pRNN = netOptions[pRNNtype](self.obs_size, self.act_size, self.hidden_size,
                                        trunc = bptttrunc, neuralTimescale=neuralTimescale,
                                        dropp=dropp, f=f)
        if identityInit: 
            self.pRNN.W = nn.Parameter(torch.eye(hidden_size))

        self.loss_fn  = lossOptions[losstype]()
        self.resetOptimizer(learningRate, weight_decay,
                            trainBias=trainBias, bias_lr=bias_lr)

        self.loss_fn_spont = LPLLoss(lambda_decorr=0,lambda_hebb=0.02)

        #Set up the training trackers
        self.TrainingSaver = pd.DataFrame()
        self.numTrainingTrials = -1
        self.numTrainingEpochs = -1

        #The homeostatic targets
        self.target_rate = target_rate
        self.target_sparsity = target_sparsity
        self.decorrelate = decorrelate

        #For single-step prediction
        self.state = torch.tensor([])
        self.phase = 0
        self.phase_k = len(self.pRNN.inMask)

    def predict(self, obs, act, state=torch.tensor([]), mask=None, randInit=True):
        """
        Generate predicted observation sequence from an observation and action
        sequence batch. Obs_pred is for the next timestep. 
        Note: state input is used for CANN control in internal functions
        """
        device = self.pRNN.W.device
        
        k=0
        if hasattr(self.pRNN,'k'):
            k= self.pRNN.k
            
        #NOTE: this should be done in Architectures.
        #TODO: make batched
        if hasattr(self,'trainNoiseMeanStd') and self.trainNoiseMeanStd != (0,0):
            noise = self.trainNoiseMeanStd
            timesteps = obs.size(1)
            noise_t = noise[0] + noise[1]*torch.randn((k+1,timesteps,self.hidden_size),
                                                                    device=device)
            if randInit and len(state) == 0:
                state = noise[0] + noise[1]*torch.randn((1,1,self.hidden_size),
                                                            device=device)
                state = self.pRNN.rnn.cell.actfun(state)
        else:
            noise_t = torch.tensor([])
        
        obs_pred, h, obs_next = self.pRNN(obs, act, noise_t=noise_t, state=state, mask=mask)
        #TODO: option to put current h0, and predict forward
        return obs_pred, obs_next, h
    
    def predict_single(self, obs, act):
        """
        Generate pRNN activation from single observation and action.
        There is no dropout step here, so the output will be different from predict().
        """
        device = self.pRNN.W.device
        
        k=0
        if hasattr(self.pRNN,'k'):
            k= self.pRNN.k
            
        if hasattr(self,'trainNoiseMeanStd') and self.trainNoiseMeanStd != (0,0):
            noise = self.trainNoiseMeanStd
            noise_t = noise[0] + noise[1]*torch.randn((k+1,1,self.hidden_size),
                                                                    device=device)
        else:
            noise_t = torch.tensor([])

        # Mask observations and actions according to current phase
        obs = obs * self.pRNN.inMask[self.phase]
        act = act * self.pRNN.actMask[self.phase]
        self.phase = (self.phase + 1) % self.phase_k
        
        _, self.state, _ = self.pRNN(obs, act, noise_t=noise_t, state=self.state, single=True)
        return self.state
    
    def reset_state(self, randInit=True, device='cpu'):
        self.state = torch.tensor([], device=device)
        if randInit:
                noise = self.trainNoiseMeanStd
                self.state = noise[0] + noise[1]*torch.randn((1,1,self.hidden_size), device=device)
                self.state = self.pRNN.rnn.cell.actfun(self.state)
        self.phase = 0


    def spontaneous(self, timesteps, noisemean, noisestd, wgain=1, agent=None, randInit=True):
        device = self.pRNN.W.device
        #Noise
        #NOTE: this should be done in Architectures. 
        #TODO Future update: move noise to pRNN.forward, pRNN.internal as parameters,
        #                    move theta dimension to dim=3, to allow for batched input
        noise_t = noisemean + noisestd*torch.randn((1,timesteps,self.hidden_size),
                                                   device=device)
        if randInit:
            state = noisemean + noisestd*torch.randn((1,1,self.hidden_size),
                                                       device=device)
            state = self.pRNN.rnn.cell.actfun(state)
        else:
            state = torch.tensor([])
                
        #Weight Gain
        with torch.no_grad():
            offdiags = self.pRNN.W.mul(1-torch.eye(self.hidden_size))
            self.pRNN.W.add_(offdiags*(wgain-1)) 
            
        #Action. This should be done separately, similar to self.predict...
        if agent is not None:
            env = self.EnvLibrary[-1]
            obs,act,_,_ = self.collectObservationSequence(env, 
                                                          agent, 
                                                          timesteps)
            obs,act = obs.to(device),act.to(device)
            obs = torch.zeros_like(obs)
            
            obs_pred, h_t, _ = self.pRNN(obs, act, noise_t=noise_t, state=state, theta=0)
            noise_t = (noise_t,act)
        else:
            obs_pred,h_t = self.pRNN.internal(noise_t, state=state)
        
        with torch.no_grad():
            self.pRNN.W.subtract_(offdiags*(wgain-1))
            
        return obs_pred,h_t,noise_t

    def trainStep(self, obs, act, 
                  with_homeostat = False,
                  learningRate=None,
                  mask=None):
        """
        One training step from an observation and action sequence
        (collected via obs, act = agent.getObservations(env,tsteps))
        """

        obs_pred, obs_next, h = self.predict(obs, act, mask=mask)
        predloss = self.loss_fn(obs_pred, obs_next, h)

        if with_homeostat:
            target_sparsity = self.target_sparsity
            target_rate = self.target_rate
            decor = self.decorrelate
        else:
            target_sparsity, target_rate, decor = None, None, False

        homeoloss, sparsity, meanrate = self.homeostaticLoss(h,target_sparsity,target_rate,decor)

        loss = with_homeostat*homeoloss+predloss   

        if learningRate is not None:
            oldlr = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = learningRate
        
        #Backpropagation (through time)
        self.optimizer.zero_grad()  #Clear the gradients
        loss.backward()             #Backpropagate w.r.t the loss
        self.optimizer.step()       #Adjust the parameters
        
        if learningRate is not None:
            self.optimizer.param_groups[0]['lr'] = oldlr

        steploss = predloss.item()
        self.recordTrainingTrial(steploss)

        return steploss, sparsity, meanrate
    
    def sleepStep(self, timesteps, noisemean, noisestd,
                  with_homeostat = False):
        """
        One training step from internally-generated activity
        """
        
        obs_pred,h_t,noise_t = self.spontaneous(timesteps, noisemean, noisestd)
        spontloss = self.loss_fn_spont(None,None,h_t)
        
        if with_homeostat:
            target_sparsity = self.target_sparsity
            target_rate = self.target_rate
            decor = self.decorrelate
        else:
            target_sparsity, target_rate, decor = None, None, False

        homeoloss, sparsity, meanrate = self.homeostaticLoss(h_t,target_sparsity,target_rate,decor)

        loss = with_homeostat*homeoloss+spontloss
        
        #Backpropagation (through time)
        self.optimizer.zero_grad()  #Clear the gradients
        loss.backward()             #Backpropagate w.r.t the loss
        self.optimizer.step()       #Adjust the parameters
        
        steploss = spontloss.item()
        self.recordTrainingTrial(steploss)

        return steploss, sparsity, meanrate
        

    def homeostaticStep(self,h,target_sparsity=None,target_rate=None,
                        decorrelate = False):
        """
        One step optimizing the homeostatic loss, using activations h that have
        not had an optimizer.step()
        """
        self.optimizer.zero_grad()  #Clear the gradients
        homeoloss, sparsity, meanrate = self.homeostaticLoss(h,target_sparsity=target_sparsity,target_rate=target_rate)
        homeoloss.backward()
        self.optimizer.step()

        #self.addTrainingData('sparsity',sparsity.mean().item())
        #self.addTrainingData('meanrate',meanrate.mean().item())
        #self.addTrainingData('normrate',normrate.mean().item())
        return homeoloss, sparsity, meanrate

    def homeostaticLoss(self,h,target_sparsity=None,target_rate=None,
                        decorrelate = False):

        meanrate = torch.mean(h,dim=1)
        sparsity = torch.linalg.vector_norm(h,ord=0,dim=2)/h.size(1)

        if target_sparsity:
            normh = h/(meanrate+1e-16)
            normh = torch.minimum(normh,torch.Tensor([1]))
            L1sparsity = torch.linalg.vector_norm(normh,ord=1,dim=2)/h.size(1)
            target_sparsity = target_sparsity*torch.ones_like(L1sparsity)
            sparseloss = self.loss_fn(L1sparsity,target_sparsity)
            #sparseloss.backward(retain_graph=True) #Backprop w.r.t. sparsity loss
        else:
            sparseloss = 0

        if target_rate is not None:
            target_rate = torch.Tensor(target_rate)
            normrate = meanrate/target_rate
            rateloss = self.loss_fn(normrate,torch.ones_like(target_rate))
            #rateloss.backward() #Add gradient w.r.t. rate loss
        else:
            rateloss = 0

        if decorrelate:
            corr = torch.corrcoef(h.squeeze())
            target = torch.eye(corr.size(0))
            decorloss = self.loss_fn(corr,target)
        else:
            decorloss = 0

        homeoloss = sparseloss + rateloss + decorloss
        #sparsity.mean().item(), meanrate.mean().item()
        return homeoloss, sparsity.mean().item(), meanrate.mean().item()


    def collectObservationSequence(self, env, agent, tsteps, batch_size=1,
                                   obs_format='pred', includeRender=False,
                                   discretize=False, inv_x=False, inv_y=False,
                                   seed = None, dataloader=False):
        """
        A placeholder for backward compatibility, actual function is moved to Shell
        """
        obs, act, state, render = env.collectObservationSequence(agent, tsteps,
                                                                batch_size, obs_format,
                                                                includeRender,
                                                                discretize, inv_x, inv_y,
                                                                seed,
                                                                dataloader=dataloader)

        return obs, act, state, render


    def trainingEpoch(self, env, agent,
                            sequence_duration=2000, num_trials=100,
                            with_homeostat=False,
                            learningRate=None,
                            forceDevice=None,
                            batch_size=1):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if forceDevice is not None:
            device = forceDevice
        self.pRNN.to(device)
        print(f'Training pRNN on {device}...')

        for bb in range(num_trials//batch_size):
            tic = time.time()
            obs,act,_,_ = self.collectObservationSequence(env, 
                                                          agent, 
                                                          sequence_duration,
                                                          dataloader=self.useDataLoader)
            obs,act = obs.to(device),act.to(device)
            steploss, sparsity, meanrate = self.trainStep(obs,act, 
                                                          with_homeostat,
                                                          learningRate=learningRate)
            #self.addTrainingData('sequence_duration',sequence_duration)
            #self.addTrainingData('clocktime',time.time()-tic)
            c=100
            #Until batch is implemented....
            #if (bb*batch_size+c) >= 100 or bb==num_trials//batch_size-1:
            if (100*bb /num_trials) % 10 == 0 or bb==num_trials-1:
                c-=100
                print(f"loss: {steploss:>.2}, sparsity: {sparsity:>.2}, meanrate: {meanrate:>.2} [{bb:>5d}\{num_trials:>5d}]")
        
        self.numTrainingEpochs +=1
        self.pRNN.to('cpu')
        print("Epoch Complete. Back to the cpu")


    def sleepEpoch(self, noisemean, noisestd,
                   sequence_duration=2000, num_trials=100,
                   with_homeostat = False):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        #device = "cpu"
        self.pRNN.to(device)
        print(f'Training pRNN on {device}...')

        for bb in range(num_trials):
            # obs_pred,h_t,noise_t = self.spontaneous(sequence_duration, noisemean, noisestd)

            # homeoloss,sparsity, meanrate = self.homeostaticStep(h_t,
            #                                                 self.target_sparsity,
            #                                                 self.target_rate)
            steploss, sparsity, meanrate = self.sleepStep(sequence_duration, noisemean, noisestd,
                                                     with_homeostat = False)

            if (100*bb /num_trials) % 10 == 0 or bb==num_trials-1:
                print(f"loss: {steploss:>.2}, sparsity: {sparsity:>.2}, meanrate: {meanrate:>.2} [{bb:>5d}\{num_trials:>5d}]")

        self.pRNN.to('cpu')
        print("Epoch Complete. Back to the cpu")


    def recordTrainingTrial(self,loss):
        self.numTrainingTrials += 1 #Increase the counter
        newTrial = pd.DataFrame({'loss':loss}, index=[0])#, 'learning_rate':self.learningRate}
        try:
            self.TrainingSaver = pd.concat((self.TrainingSaver,newTrial), ignore_index=True)
        except:
            self.TrainingSaver = pd.concat((self.TrainingSaver.to_frame(),newTrial), ignore_index=True)
        return

    def addTrainingData(self,key,data):
        if isinstance(data,pd.Series) or isinstance(data,pd.DataFrame):
            data = data.values
        if not key in self.TrainingSaver and (isinstance(data,np.ndarray)
                                              or isinstance(data,list)
                                              or isinstance(data,dict)):
            self.TrainingSaver.at[self.numTrainingTrials,key] = 0
            self.TrainingSaver[key] = self.TrainingSaver[key].astype('object')

        self.TrainingSaver.at[self.numTrainingTrials,key] = data
        return


    def loadEnvironment(self, idx):
        #Throw an error if the environment doesn't exist
        if idx >= len(self.EnvLibrary):
            raise ValueError(f"Environment {idx} does not exist")
        return self.EnvLibrary[idx]


    """ I/O Functions """
    def addEnvironment(self,env):
        """
        Add an environment to the library. If it's not environment 0, check
        that it's Observation/Action space matches environment 0.
        Generate trajectories if DataLoader is used
        and they are not generated already.
        """
        self.EnvLibrary.append(env)
        return
    

    def resetOptimizer(self, learningRate, weight_decay, trainBias=False, bias_lr=1):
        self.learningRate = learningRate
        self.weight_decay = weight_decay
        rootk_h = np.sqrt(1./self.pRNN.rnn.cell.hidden_size)
        rootk_i = np.sqrt(1./self.pRNN.rnn.cell.input_size)

        self.optimizer = torch.optim.RMSprop([
                        {'params': self.pRNN.W, 'name': 'RecurrentWeights', 
                         'lr': learningRate*rootk_h, 
                         'weight_decay': weight_decay*learningRate*rootk_h},
                        {'params': self.pRNN.W_out, 'name': 'OutputWeights', 
                         'lr': learningRate*rootk_h, 
                         'weight_decay': weight_decay*learningRate*rootk_h},
                        {'params': self.pRNN.W_in, 'name': 'InputWeights', 
                         'lr': learningRate*rootk_i, 
                         'weight_decay': weight_decay*learningRate*rootk_i}
                    ],
                        alpha=0.95, eps=1e-7)
                        #Parms from Recanatesi
                        #lr=1e-4, alpha=0.95, eps=1e-7
        
        if trainBias:
            self.bias_lr = bias_lr
            biasparmgroup = {
                'params' : self.pRNN.bias,
                'name' : 'biases',
                'lr' : learningRate*bias_lr,#Note: most papers use same as recurrent weights..., 
                                            #but seems better to have no scaling by rootk?
                'weight_decay': weight_decay*learningRate*bias_lr  
            }
            
            self.optimizer.add_param_group(biasparmgroup)
        else:
            self.pRNN.bias.requires_grad = False
    #TODO: convert these to general.savePkl and general.loadPkl (follow SpatialTuningAnalysis.py)
    def saveNet(self,savename,savefolder=''):
        # Collect the iterators that cannot be pickled
        iterators = [env.killIterator() for env in self.EnvLibrary]
        # Save the net
        filename = savefolder+'nets/'+savename+'.pkl'
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename,'wb') as f:
            pickle.dump(self, f)
        # Restore the iterators
        for i,env in enumerate(self.EnvLibrary):
            env.DL_iterator = iterators[i]
        print("Net Saved to pathname")

    def copy(self):
        """
        Create a copy of the current net,
        except for the Data Loader iterators which cannot be copied
        """
        iterators = [env.killIterator() for env in self.EnvLibrary]
        clone = deepcopy(self)
        for i,env in enumerate(self.EnvLibrary):
            env.DL_iterator = iterators[i]
        return clone


    def loadNet(savename, savefolder='', suppressText=False):
        #TODO Load in init... from filename
        filename = savefolder+'nets/'+savename+'.pkl'
        with open(filename,'rb') as f:
            predAgent = pickle.load(f)
        if not hasattr(predAgent, "env_shell"): # backward compatibility for the nets trained before Shell update
            from prnn.utils.Shell import GymMinigridShell
            for i in range(len(predAgent.EnvLibrary)):
                predAgent.EnvLibrary[i] = GymMinigridShell(predAgent.EnvLibrary[i],
                                                            predAgent.encodeAction.__name__)
            predAgent.env_shell = predAgent.EnvLibrary[0]
        if not suppressText:
            print("Net Loaded from pathname")
        return predAgent


    """ Basic Analysis """
    def calculateSpatialRepresentation(self,env,agent, timesteps=15000,
                                       saveTrainingData=False, trainDecoder=False,
                                       trainHDDecoder = False,
                                       numBatches=5000, inputControl=False,
                                       calculatesRSA = False, bitsec=False,
                                       sleepstd = 0.1, onsetTransient=20,
                                       activeTimeThreshold=200):
        """
        Use an agent to calculate spatial representation of an environment
        """
        obs, act, state, render = self.collectObservationSequence(env,agent,timesteps,discretize=True)

        if hasattr(self, 'current_state'): # easy way to check if it's CANN
            obs_pred, obs_next, h  = self.predict(obs,act,state)
        else:
            obs_pred, obs_next, h  = self.predict(obs,act)
        
        #for now: take only the 0th theta window...
        #Try: mean
        #THETA UPDATE NEEDED
        #h = h[0:1,:,:]
        h = torch.mean(h,dim=0,keepdims=True)
        ##FIX ABOVE HERE FOR k

        position = nap.TsdFrame(t = np.arange(onsetTransient,timesteps),
                                d = state['agent_pos'][onsetTransient:-1,:],
                                columns = ('x','y'), time_units = 's')
        rates = nap.TsdFrame(t = np.arange(onsetTransient,h.size(1)),
                             d = h.squeeze().detach().numpy()[onsetTransient:,:], time_units = 's')
        

        width = env.width
        height = env.height
        nb_bins_x, nb_bins_y, minmax = env.get_map_bins()

        place_fields,xy = nap.compute_2d_tuning_curves_continuous(rates,position,
                                                                  ep=rates.time_support,
                                                                  nb_bins=(nb_bins_x, nb_bins_y),
                                                                  minmax=minmax
                                                                  )
        SI = nap.compute_2d_mutual_info(place_fields, position, position.time_support,
                                        bitssec=bitsec)
        #Remove units that aren't active in enough timepoints
        numactiveT = np.sum((h>0).numpy(),axis=1)
        inactive_cells = numactiveT<activeTimeThreshold
        SI.iloc[inactive_cells.flatten()]=0


        if inputControl:
            rates_input = nap.TsdFrame(t = np.arange(onsetTransient,timesteps),
                                 d = obs.squeeze().detach().numpy()[onsetTransient:-1,:], time_units = 's')
            pf_input,xy = nap.compute_2d_tuning_curves_continuous(rates_input,position,
                                                                  ep=rates.time_support,
                                                                  nb_bins=(nb_bins_x, nb_bins_y),
                                                                  minmax=minmax
                                                                  )
            SI_input = nap.compute_2d_mutual_info(pf_input, position, position.time_support,
                                            bitssec=bitsec)
            SI_input['pfs'] = pf_input.values()
            SI['inputCtrl'] = SI_input['SI']
            SI['inputFields'] = SI_input['pfs']#pd.DataFrame.from_dict(pf_input)

        if calculatesRSA:
            WAKEactivity = {
                'state' : state,
                'h'     : np.squeeze(h.detach().numpy())
            }
            #sRSA
            (sRSA,_),_,_,_ = RGA.calculateRSA_space(RGA,WAKEactivity,
                                    cont=env.continuous,
                                    max_dist=env.max_dist)
            
            #SW Distance
            noisemag = 0
            noisestd = sleepstd
            timesteps_sleep = 500
            _,h_t,_ = self.spontaneous(timesteps_sleep,noisemag,noisestd)
            SLEEPactivity = {'h'     : np.squeeze(h_t.detach().numpy())}
            SWdist,_,_ = RGA.calculateSleepWakeDist(WAKEactivity['h'],
                                                    SLEEPactivity['h'],
                                                    metric='cosine')
            
            #EV_s
            FAKEinputdata = STA.makeFAKEdata(WAKEactivity,place_fields)
            EVs = FAKEinputdata['TCcorr']
            if saveTrainingData:
                self.addTrainingData('sRSA',sRSA)
                self.addTrainingData('SWdist',SWdist)
                self.addTrainingData('EVs',EVs)
        
        decoder = None
        if trainDecoder:
            #Consider - this has extra weights for walls...
            decoder = linearDecoder(self.hidden_size,
                                    height*width)
            #Reformat inputs
            h_decoder = torch.squeeze(torch.tensor(h.detach().numpy()))
            pos_decoder = np.array([state['agent_pos'][:h_decoder.size(0),0],
                                    state['agent_pos'][:h_decoder.size(0),1]])
            
            pos_decoder = np.ravel_multi_index(pos_decoder,
                                            (width,height))
            pos_decoder = torch.tensor(pos_decoder)

            decoder.train(h_decoder, pos_decoder, batchSize=0.5, numBatches=numBatches)

            def unravel_pos(pos):
                pos = np.vstack(np.unravel_index(pos, (width, height))).T
                return pos
            def unravel_p(p):
                p = np.reshape(p.detach().numpy(),(-1, width, height))
                return p
            decoder.unravel_pos = unravel_pos
            decoder.unravel_p = unravel_p
            decoder.gridheight = height
            decoder.gridwidth = width

            totalPF = np.array(list(place_fields.values())).sum(axis=0)
            decoder.mask = np.array((totalPF>0)*1.)
            decoder.mask = np.pad(decoder.mask, 1)
           # decoder.mask_p = #here:use pos_decoder to build mask
        
            if trainHDDecoder:
                numHDs = env.numHDs
                HDdecoder = linearDecoder(self.hidden_size, numHDs)
                #Reformat inputs
                pos_decoder = np.array(state['agent_dir'][:h_decoder.size(0)])                     
                pos_decoder = torch.tensor(pos_decoder)
                if env.continuous:
                    pos_decoder = (pos_decoder*numHDs).long()
                    pos_decoder = torch.clamp(pos_decoder, 0, numHDs-1)

                HDdecoder.train(h_decoder, pos_decoder, batchSize=0.5, numBatches=numBatches)
                
                def unravel_pos_HD(pos):
                    pos = np.vstack(np.unravel_index(pos, (numHDs))).T
                    return pos
                def unravel_p_HD(p):
                    p = np.reshape(p.detach().numpy(), (-1, numHDs))
                    return p
                HDdecoder.unravel_pos = unravel_pos_HD
                HDdecoder.unravel_p = unravel_p_HD
                
                decoder.HDdecoder = HDdecoder

        if saveTrainingData:
            self.addTrainingData('place_fields',place_fields)
            self.addTrainingData('SI',SI)
        return place_fields, SI, decoder


    def decode(self, h, decoder, withHD = False):
        """
        """
        #OLD PYNAPPLE WAY
        timesteps = h.size(1)
        # rates = nap.TsdFrame(t = np.arange(timesteps), d = h.squeeze().detach().numpy(),
                             # time_units = 's')
        #TODO: fix this... bins should be stored with place_fields
        #TODO: provide occupancy of the available environment
        # xy2 = [np.arange(1,np.size(place_fields[0],0)+1),
        #        np.arange(1,np.size(place_fields[1],0)+1)]
        #decoded, p = nap.decode_2d(place_fields,rates,rates.time_support,1,xy2)

        h_decoder = torch.squeeze(h)
        decoded, p = decoder.decode(h_decoder,withSoftmax=True)

        p = decoder.unravel_p(p)
        decoded = decoder.unravel_pos(decoded)
        
        dims = ('x','y')
        
        if withHD:
            decodedHD, pHD = decoder.HDdecoder.decode(h_decoder,withSoftmax=True)
            decodedHD = decoder.HDdecoder.unravel_pos(decodedHD)
            decoded = np.concatenate((decoded,decodedHD),axis=1)
            dims = ('x','y','HD')

        decoded = nap.TsdFrame(
            t = np.arange(timesteps),
            d = decoded,
            time_units = 's',
            columns=dims
            )

        return decoded,p

    def decode_error(self, decoded, state, restrictPos=None):
        cols = decoded.columns
        data = np.vstack((state['agent_pos'][:-1,0],state['agent_pos'][:-1,1])).T
        state = nap.TsdFrame(t=decoded.index.values,
                                 d=data,
                                 time_units='us',
                                 time_support=decoded.time_support,
                                 columns=cols)
        if restrictPos is not None:
            state = state[(state['x'].values==restrictPos[0]) &
                          (state['y'].values==restrictPos[1])]
            decoded = state.value_from(decoded,ep=decoded.time_support)
        #TODO: random positions should only be those available to the agent
        #TODO: rantint values should not be hard coded...
        randpos = nap.TsdFrame(t=decoded.index.values,
                                 d=np.random.randint(1,15,np.shape(data)),
                                 time_support=decoded.time_support,
                                 columns=cols)

        derror = np.sum(np.abs(decoded.values - state.values), axis=1)
        dshuffle = np.sum(np.abs(randpos.values - state.values), axis=1)
        return derror, dshuffle

    def calculateDecodingPerformance(self,env,agent,decoder,timesteps = 2000,
                                     savefolder=None,savename=None,saveTrainingData=False,
                                     showFig=True, trajectoryWindow=100,
                                     seed = None):

        obs,act,state,render = self.collectObservationSequence(env,agent,
                                                                        timesteps,
                                                                        includeRender=True,
                                                                        discretize=True,
                                                                        seed=seed
                                                                        )
        obs_pred, obs_next, h  = self.predict(obs,act)
        
        #for now: take only the 0th theta window...
        #THETA UPDATE NEEDED
        k=0
        if hasattr(self.pRNN,'k'):
            k= self.pRNN.k
        state['agent_pos'] = state['agent_pos'][:h.shape[1]+1,:]
        h = h[0:1,:,:]
        obs_pred = obs_pred[0:1,:,:]
        timesteps = timesteps-(k+1)
        ##FIX ABOVE HERE FOR K
        
        decoded, p = self.decode(h,decoder)
        if showFig:
            self.plotObservationSequence(obs,render,obs_pred,state,
                                         p_decode=p,decoded=decoded,mask=decoder.mask,
                                         timesteps=range(timesteps-6,timesteps),
                                         savefolder=savefolder, savename=savename,
                                         trajectoryWindow=trajectoryWindow)
        derror, dshuffle = self.decode_error(decoded, state)
        if saveTrainingData:
            self.addTrainingData('derror',derror)
        return


    def calculateActivationStats(self,h,onset=100):
        """
        Calculates the gross statistics of neuronal activations from a recurrent
        activity tensor h
        """
        h = h.detach().numpy()

        actStats = {}
        #actStats['ratedist_t']
        actStats['poprate_t'] = np.squeeze(np.mean(h,2))
        actStats['popstd_t'] = np.squeeze(np.std(h,2))

        h = h[:,onset:,:]
        actStats['meanrate']  = np.mean(h)
        actStats['stdrate']  = np.std(h)
        actStats['meanpoprate'] = np.mean(actStats['poprate_t'][onset:])
        actStats['stdpoprate'] = np.std(actStats['poprate_t'][onset:])
        actStats['meancellrates'] = np.squeeze(np.mean(h,1))
        actStats['cellstds'] = np.std(h,1)
        actStats['meancellstds'] = np.mean(np.std(h,1))
        actStats['stdcellrates'] = np.std(actStats['meancellrates'])

        return actStats

    """Plot Functions"""
    def plotSampleTrajectory(self,env,agent,timesteps=100,decoder=False,
                             savename=None,savefolder=None, plot=True,
                             plotCANN=False):
        #Consider - separate file for the more compound plots
        obs,act,state, render  = self.collectObservationSequence(env,agent,timesteps,
                                                           includeRender = True)
        obs_pred, obs_next ,h = self.predict(obs,act)

        decoded = None
        if decoder:
            decoded, p = self.decode(h,decoder)

            
        #for now: take only the 0th theta window...
        #THETA UPDATE NEEDED
        k=0
        if hasattr(self.pRNN,'k'):
            k= self.pRNN.k
        state['agent_pos'] = state['agent_pos'][:state['agent_pos'].shape[0]-k+1,:]
        #h = h[0:1,:,:]
        h = torch.mean(h,dim=0,keepdims=True) #this is what's used to train the decoder...
        if obs_pred is not None:
            obs_pred = obs_pred[0:1,:,:]
        timesteps = timesteps-(k+1)
        ##FIX ABOVE HERE FOR K
            
            
        if plotCANN is not False:
            h = h.squeeze().detach().numpy()
        else:
            h = None            
            
        if plot:
            self.plotObservationSequence(obs,render,obs_pred,
                                         timesteps=range(timesteps-5,timesteps),
                                         p_decode = None,state=state,h=h,
                                         savename=savename,savefolder=savefolder)


        return state, decoded

    def plotSpontaneousTrajectory(self,noisemag,noisestd,timesteps=100,decoder=None,
                             savename=None,savefolder=None, plot=True,
                             plotCANN=False):
        #Consider - separate file for the more compound plots
        obs_pred,h,noise_t = self.spontaneous(timesteps,noisemag,noisestd)

        p = None
        if decoder:
            decoded, p = self.decode(h,decoder)

            maxp = np.max(p,axis=(1,2))
            dx = np.sum(np.abs(decoded.values[:-1,:] - decoded.values[1:,:]),
                        axis=1)
            dd = delaydist(decoded.values,numdelays=10)

            pdhist,pbinedges,dtbinedges = np.histogram2d(dx,np.log10(maxp[:-1]),
                                    bins=[np.arange(0,15),np.linspace(-1.25,-0.75,11)])
            pdhist = pdhist/np.sum(pdhist,axis=0)

            lochist,_,_ = np.histogram2d(decoded['x'].values,decoded['y'].values,
                                    bins=[np.arange(0,decoder.gridheight)-0.5,np.arange(0,decoder.gridheight-1)])

        if plotCANN is not False:
            h = h.squeeze().detach().numpy()
        else:
            h = False

        if plot:
            plt.figure(figsize=(10,10))
            self.plotSequence(self.env_shell.pred2np(obs_pred),
                              range(timesteps-5,timesteps),3,label='Predicted')
            self.plotSequence(p, range(timesteps-5,timesteps),4,
                              label='Decoded')

            if savename is not None:
                saveFig(plt.gcf(),savename+'_SpontaneousTrajectory',savefolder,
                        filetype='pdf')
            plt.show()

        return decoded

    def plotSequence(self, sequence, timesteps, row, label=None, mask=None,
                     cmap='viridis', vmin=None, vmax=None, numrows=6):

        numtimesteps = len(timesteps)
        for tt,timestep in enumerate(timesteps):
            plt.subplot(numrows, numtimesteps, tt+1+(row-1)*numtimesteps )
            plt.imshow(sequence[timestep], alpha=mask, cmap=cmap,
                      vmin=vmin, vmax=vmax)
            if tt==0:
                plt.ylabel(label)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

    def plotObservationSequence(self, obs, render, obs_pred, state=None,
                                p_decode = None, decoded=None, mask=None,
                                timesteps=None, h=None,
                                savename=None, savefolder=None,
                                obs_next=None, trajectoryWindow=100):
        """
        Plots the gridworld render, observation, and predicted observation from
        an observation sequence
        (note, this maybe doesn't need to be in the predictiveNet class...')
        """
        timedim = 0 #make this a self.variable
        if timesteps is None:
            timesteps = range(np.size(obs,timedim))
        start = max(timesteps[-1]-trajectoryWindow,0)

        numtimesteps = len(timesteps)
        
        
        obs = self.env_shell.pred2np(obs[:,timesteps,...])
        if obs_pred is not None:
            obs_pred = self.env_shell.pred2np(obs_pred[:,timesteps,...])
        #obs_next = self.pred2np(obs_next[:,timesteps,...])

        #figure bigger, get rid of axis numbers, add labels
        fig = plt.figure(figsize=(10,10))

        if decoded is not None and state is not None:
            derror,dshuffle = self.decode_error(decoded,state)
            plt.subplot(4,2,2)
            plt.hist(derror,bins=np.arange(0,20)-0.5)
            plt.hist(dshuffle,edgecolor='k',fill=False,bins=np.arange(0,20)-0.5)
            plt.legend(('Actual','Random'))
            plt.xlabel('Decode Error')

        if state is not None:
            ax = plt.subplot(3,3,1)
            self.env_shell.show_state_traj(start=start,
                                           end=timesteps[-1],
                                           state=state,
                                           render=render,
                                           fig=fig,
                                           ax=ax)
            #TODO
            #if p_decode is not None:
            #     plt.plot((p_decode['x'][trajectory_ts]+0.5)*512/16,
            #              (p_decode['y'][trajectory_ts]+0.5)*512/16,
            #              linestyle=':',color='b')
            plt.xticks([])
            plt.yticks([])

        for tt in range(numtimesteps):
            ax = plt.subplot(6,numtimesteps,tt+1+2*numtimesteps)
            self.env_shell.show_state(render=render,
                                      t=timesteps[tt],
                                      fig=fig,
                                      ax=ax)
            if tt==0:
                plt.ylabel('State')
            plt.xticks([])
            plt.yticks([])
            plt.title(timesteps[tt])

            plt.subplot(6,numtimesteps,numtimesteps+tt+1+2*numtimesteps )
            plt.imshow(obs[tt,:,:,:])
            if tt==0:
                plt.ylabel('Observation')
            plt.xticks([])
            plt.yticks([])
            if obs_pred is not None:
                plt.subplot(6,numtimesteps,2*numtimesteps+tt+1+2*numtimesteps )
                plt.imshow(obs_pred[tt,:,:,:])
                if tt==0:
                    plt.ylabel('Predicticted')
                plt.xticks([])
                plt.yticks([])

            if obs_next is not None:
                plt.subplot(6,numtimesteps,3*numtimesteps+tt+1+2*numtimesteps)
                plt.imshow(obs_next[tt,:,:,:])
                if tt==0:
                    plt.ylabel('obs_next')
                plt.xticks([])
                plt.yticks([])

            if p_decode is not None:
                plt.subplot(6,numtimesteps,3*numtimesteps+tt+1+2*numtimesteps)
                plt.imshow(np.log10(p_decode[timesteps[tt],:,:].transpose()),
                           interpolation='nearest',alpha = mask.transpose(),
                          cmap='bone',vmin=-3.25,vmax=0)
                if state is not None:
                    plt.plot(state['agent_pos'][timesteps[tt],0],
                             state['agent_pos'][timesteps[tt],1],
                             marker=(3, 0, self.env_shell.dir2deg(state['agent_dir'][timesteps[tt]])),
                             color='r',
                             markersize=6)
                #plt.plot(decoded.iloc[timesteps[tt]]['x'],
                #          decoded.iloc[timesteps[tt]]['y'],'y+', markersize=5)
                if tt==0:
                    plt.ylabel('decoded')
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')

            if h is not None:
                plt.subplot(6,numtimesteps,3*numtimesteps+tt+1+2*numtimesteps)
                plt.scatter(self.pRNN.locations[0][:,0],
                         -self.pRNN.locations[0][:,1],s=15,c=h[timesteps[tt],:])
                plt.xticks([])
                plt.yticks([])

        if savename is not None:
            #plt.savefig(savename+'_ObservationSequence.pdf',format='pdf')
            saveFig(plt.gcf(),savename+'_ObservationSequence',savefolder,
                    filetype='pdf')
        plt.show()

        return


    def plotLearningCurve(self,onsettransient=0, incSI = True, incDecode = True,
                          savename=None,savefolder=None, axis=None, maxBoxes=10):

        if axis is None:
            fig, axis = plt.subplots()
        if incSI:
            ax2 = axis.twinx()
            self.plotSpatialInfo(ax2, maxBoxes=maxBoxes)
        if incDecode:
            ax3 = axis.twinx()
            ax3.spines.right.set_position(("axes", 1.12))
            self.plotDecodePerformance(ax3, maxBoxes=maxBoxes)
        axis.plot(np.log10(self.TrainingSaver['loss'][onsettransient:]),'k-')
        axis.set_xlabel('Training Steps')
        axis.set_ylabel('log10(Loss)')
        #plt.xticks([0,self.numTrainingTrials+1])

        if savename is not None:
            #plt.savefig(savename+'_LerningCurve.pdf',format='pdf')
            saveFig(fig,savename+'_LearningCurve',savefolder,
                    filetype='pdf')
        if axis is None:
            plt.show()

        return

    def plotSpatialInfo(self,ax, color='red', maxBoxes=np.inf):
        trials = ~self.TrainingSaver['SI'].isna()
        index = self.TrainingSaver['SI'].index[trials]
        if len(index) > maxBoxes:
            idx = np.int32(np.linspace(0,len(index)-1,maxBoxes))
            index = index[idx]

        SI = self.TrainingSaver.loc[index,'SI'].values.tolist()
        SI = [np.array(i) for i in SI]
        SI = [i[~np.isnan(i)] for i in SI]

        if len(index)>2:
            widths = (index[1]-index[0])/3
        else:
            widths = index[-1]/3

        bp = ax.boxplot(SI,positions=index-widths/2,
                        widths=widths, manage_ticks=False, sym='.',showfliers=False)

        #Graphics stuff
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=color)
        plt.setp(bp["fliers"], markeredgecolor=color)
        ax.set_ylabel("Spatial Info")
        #ax.spines['right'].set_color('red')
        ax.yaxis.label.set_color('red')
        ax.tick_params(axis='y', colors='red')
        return

    def plotDecodePerformance(self,ax, color='blue', maxBoxes=np.inf):
        trials = ~self.TrainingSaver['derror'].isna()
        index = self.TrainingSaver['derror'].index[trials]
        if len(index) > maxBoxes:
            idx = np.int32(np.linspace(0,len(index)-1,maxBoxes))
            index = index[idx]

        derror = self.TrainingSaver.loc[index,'derror'].values.tolist()
        derror = [np.array(i) for i in derror]
        derror = [i[~np.isnan(i)] for i in derror]

        if len(index)>2:
            widths = (index[1]-index[0])/3
        else:
            widths = index[0]/3

        bp = ax.boxplot(derror,positions=index+widths/2,
                        widths=widths, manage_ticks=False, sym='.',showfliers=False)

        #Graphics stuff
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=color)
        plt.setp(bp["fliers"], markeredgecolor=color)
        ax.set_ylabel("Decode Error")
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis='y', colors=color)
        return


    def plotTuningCurvePanel(self, fig=None, gridsize = 5,
                             place_fields=None, SI=None, whichcells=None,
                             savename=None,savefolder=None, sithresh=None):
        
        
        if whichcells is None:
            whichcells = np.arange(1,gridsize**2+1)
        sortidx = whichcells
        if place_fields is None:
            place_fields=self.TrainingSaver['place_fields'].iloc[-1]
            SI=self.TrainingSaver['SI'].iloc[-1]
            #sortidx = np.squeeze(np.argsort(SI[whichcells],0))[::-1]+1
            sortidx = whichcells[np.squeeze(np.argsort(SI[whichcells],0))[::-1]]

        totalPF = np.array(list(place_fields.values())).sum(axis=0)
        mask = np.array((totalPF>0)*1.)

        nofig = False
        if fig is None:
            fig = plt.figure(figsize=(gridsize,gridsize))
            nofig =True
        ax = fig.subplots(gridsize,gridsize)
        plt.setp(ax, xticks=[], yticks=[])
        for x in range(gridsize):
            for y in range(gridsize):
                try:
                    idx = sortidx[x*gridsize+y]
                    ax[x,y].imshow(place_fields[idx].transpose(),
                                interpolation='nearest',alpha=mask.transpose())
                    ax[x,y].axis('off')
                    if SI is not None:
                        ax[x,y].text(0, 3, f"{SI[idx][0]:0.1}", fontsize=10,color='r')
                except:
                    break

        if savename is not None:
            saveFig(fig,savename+'_TuningCurves',savefolder,
                    filetype='pdf')

        if nofig:
            plt.show()
        return


    def plotDelayDist(self, env, agent, decoder, numdelays = 10,
                      timesteps=2000,noisemag=0,noisestd=[0,1e-2,1e-1,1e0,1e1],
                      savename=None,savefolder=None):

        wake_pos, wake_decoded = self.plotSampleTrajectory(env,agent,
                                                           decoder = decoder,
                                                           timesteps=timesteps,
                                                           plot=False)
        sleep_decoded = {}
        for noise in noisestd:
            sleep_decoded[noise] = self.plotSpontaneousTrajectory(noisemag,noise,
                                                               decoder=decoder,
                                                               timesteps=timesteps,
                                                               plot=False)

        dd={}
        dkl={}
        dd['wake'],dkl['wake'] = delaydist(wake_decoded.values,numdelays=numdelays)
        dd['pos'],dkl['pos'] = delaydist(wake_pos['agent_pos'],numdelays=numdelays)
        for noise in noisestd:
            dd[f'sleep{noise}'],dkl[f'sleep{noise}'] = delaydist(sleep_decoded[noise].values,numdelays=numdelays)

        fig=plt.figure(figsize=(10,10))
        for i,k in enumerate(dd):
            plt.subplot(5,5,i+4)
            plt.imshow(dd[k],origin='lower',
                       extent=(0.5,numdelays+0.5,-0.5,numdelays-0.5),aspect='auto')
            plt.xlabel('dt')
            plt.ylabel('dx')

        if savename is not None:
            saveFig(fig,savename+'_DelayDist',savefolder,
                    filetype='pdf')
        plt.show()

        return dd


    def plotActivationTimeseries(self,h):
        timesteps = h.size(1)
        actStats = self.calculateActivationStats(h)
        hnp = h.squeeze().detach().numpy()
        timeIDX = np.arange(timesteps-25,timesteps-1)
        neuronIDX = np.arange(0,5)

        plt.plot(hnp[timeIDX,0:5],linewidth=0.5)
        plt.plot(actStats['poprate_t'][timeIDX],color='k',linewidth=2)
        plt.plot(actStats['poprate_t'][timeIDX]+actStats['popstd_t'][timeIDX],
                 '--k',linewidth=1)
        plt.xlabel('t')
        plt.ylabel('Activations')
