import torch
from torch import nn
import numpy as np

from ratinabox.utils import get_angle

forwardIDX = 2
LIDX = 0
RIDX = 1

#Note: can remove extra 0s..... but will break old networks. Fix before reruning all networks?
#Or add backwards compatiblity option. Not a huge deal, just extra parameters.

def OneHot(act, obs, numActs = 7, **kwargs):
    #Action of -1 means no action
    noActFlag = False
    if act[0]<0:
        act = np.ones_like(act)
        noActFlag = True
    
    act = torch.tensor(act, requires_grad=False, dtype=torch.int64)
    act = nn.functional.one_hot(act, num_classes=numActs)
    act = torch.unsqueeze(act, dim=0)
    
    if noActFlag:
        act = torch.zeros_like(act)
        
    return act


def addHD(act, obs, numSuppObs = 4, suppOffset=False, **kwargs):
    #Unpredictied Observation (e.g. head direction) at time t, 
    #passed in as action
        
    if suppOffset:
        #TODO: make this be a function of which pRNN is used
        suppIdx = range(1,len(obs))
    else:
        suppIdx = range(len(obs)-1)
        
    suppObs = obs[suppIdx]
    suppObs = torch.tensor(suppObs, dtype=torch.int64, requires_grad=False)
    suppObs = nn.functional.one_hot(suppObs, num_classes=numSuppObs)
    suppObs = torch.unsqueeze(suppObs, dim=0)
    
    act = torch.cat((act,suppObs), dim=2)
    return act


def HDOnly(act, obs, numSuppObs = 4, **kwargs):
    act = NoAct(act, obs)
    act = addHD(act, obs, numSuppObs)
    return act

def OneHotHD(act, obs, numSuppObs = 4, numActs = 7):
    act = OneHot(act, obs, numActs)
    act = addHD(act, obs, numSuppObs)
    return act

def OneHotHDPrevAct(act, obs, numSuppObs = 4, numActs = 7):
    act = OneHot(act, obs, numActs)
    # Shift actions to the right by 1
    act = nn.ConstantPad1d((0,0,1,0),0)(act)[...,:-1,:]
    act = addHD(act, obs, numSuppObs)
    return act


def SpeedHD(act, obs, numSuppObs = 4, numActs = 7):
    act = OneHot(act, obs, numActs)
    act[:,:,forwardIDX+1:] = 0
    act[:,:,:forwardIDX] = 0
    act = addHD(act, obs, numSuppObs)
    return act

def SpeedNextHD(act, obs, numSuppObs = 4, numActs = 7):
    act = OneHot(act, obs, numActs = numActs)
    act[:,:,forwardIDX+1:] = 0
    act[:,:,:forwardIDX] = 0
    act = addHD(act, obs, numSuppObs, suppOffset=True)
    return act


def Velocities(act, obs, numActs = 7, **kwargs):
    act = OneHot(act, obs, numActs)
    act[:,:,forwardIDX+1:] = 0
    act[:,:,LIDX] = act[:,:,LIDX]-act[:,:,RIDX]
    act[:,:,RIDX] = 0
    return act


def NoAct(act, obs, numActs = 7, **kwargs):
    act = OneHot(act, obs, numActs)
    act = torch.zeros_like(act)
    return act

def Continuous(act):
    # Should work with any type of continuous actions
    act = torch.tensor(act, requires_grad=False)
    act = torch.unsqueeze(act, dim=0)
    return act

def ContSpeedRotation(act, meanspeed, **kwargs):
    # Made for RiaB envs, transforms 2D velocity to linear speed along head direction, then adds rotation
    speed = np.linalg.norm(act[:,1:], axis=1)/meanspeed
    act = np.concatenate((speed[:,None], act[:,0][:,None]/(2*np.pi)), axis=-1)
    act = torch.tensor(act, dtype=torch.float, requires_grad=False)
    act = torch.unsqueeze(act, dim=0)
    return act

def ContSpeedHD(act, meanspeed, **kwargs):
    # Made for RiaB envs, transforms 2D velocity to linear speed along HD, then adds HD
    speed = np.linalg.norm(act[:,1:], axis=1)/meanspeed
    HD = get_angle(act[:,1:], is_array=True)/(2*np.pi)
    act = np.concatenate((speed[:,None], HD[:,None]), axis=-1)
    act = torch.tensor(act, dtype=torch.float, requires_grad=False)
    act = torch.unsqueeze(act, dim=0)
    return act

def ContSpeedOnehotHD(act, meanspeed, nbins=12):
    act = ContSpeedHD(act, meanspeed)
    HD = (act[...,-1]*nbins).long()
    HD = torch.clamp(HD, min=0, max=nbins-1)
    HD = nn.functional.one_hot(HD, num_classes=nbins)
    act = torch.cat((act[...,:-1], HD), dim=-1)
    return act