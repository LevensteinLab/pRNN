
from torch import nn
import torch
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import norm
from prnn.utils.thetaRNN import thetaRNNLayer, RNNCell, LayerNormRNNCell, AdaptingLayerNormRNNCell, AdaptingRNNCell, SparseGatedRNNCell, LogNRNNCell

from prnn.utils.pytorchInits import CANN_
from abc import ABC, abstractmethod

## Helper Functions --


def generate_noise(noise_params, shape, w_device):
    if noise_params != (0,0):
        noise = noise_params[0] + noise_params[1]*torch.randn(shape, device=w_device)
    else:
        noise = torch.zeros(shape, device=w_device)

    return noise


#base class for future architectures unlike pRNN
class Base_pRNN(nn.Module, ABC):
    def __init__(*args, **kwargs):
        super().__init__()
        
    @abstractmethod
    def create_layers(*args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(*args, **kwargs):
        raise NotImplementedError


class pRNN(nn.Module):
    """
    A general predictive RNN framework that takes observations and actions, and
    returns predicted observations, as well as the actual observations to train
    Observations are inputs that are predicted. Actions are inputs that are
    relevant to prediction but not predicted.

    predOffset: the output at time t s matched to obs t+predOffset (defulat: 1)
    actOffset:  the action input at time t is the action took at t-actOffset (default:0)
    inMask:     boolean list, length corresponding to the prediction cycle period. (default: [True])
    outMask, actMask: list, same length as inMask (default: None)

    All inputs should be tensors of shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
    """
    def __init__(self, obs_size, act_size, hidden_size=500,
                 cell=RNNCell,  dropp=0, bptttrunc=50, k=0, f=0.5,
                 predOffset=1, inMask=[True], outMask=None, 
                 actOffset=0, actMask=None, neuralTimescale=2,
                 continuousTheta=False,
                 **cell_kwargs):
        super(pRNN, self).__init__()

        #pRNN architecture parameters
        self.predOffset = predOffset
        self.actOffset = actOffset
        self.actpad = nn.ConstantPad1d((0,0,self.actOffset,0),0)
        self.batched_actpad = nn.ConstantPad1d((0,0,0,0,self.actOffset,0),0)
        self.inMask = inMask
        if outMask is None:
            outMask = [True for i in inMask]
        if actMask is None:
            actMask = [True for i in inMask]
        self.outMask = outMask
        self.actMask = actMask
        self.hidden_size = hidden_size
        
        self.droplayer = nn.Dropout(p=dropp)
        #self.droplayer_act = nn.Dropout(p=dropp_act)
        
        # if there's a predefined input_size or output_size specified (e.g. by pRNN multimodal), use that. Else, use the standard
        self.input_size = cell_kwargs["input_size"] if "input_size" in cell_kwargs else obs_size + act_size
        self.output_size = cell_kwargs["output_size"] if "output_size" in cell_kwargs else obs_size 

        self.create_layers(self.input_size, self.output_size, hidden_size,
                           cell, bptttrunc, continuousTheta,
                           k, f, **cell_kwargs)

        self.W_in = self.rnn.cell.weight_ih
        self.W = self.rnn.cell.weight_hh
        self.W_out = self.outlayer[0].weight
        self.bias = self.rnn.cell.bias

        if hasattr(self.rnn.cell,'weight_is'):
            self.W_is = self.rnn.cell.weight_is
            self.W_sh = self.rnn.cell.weight_sh
            self.W_hs = self.rnn.cell.weight_hs
        
        self.neuralTimescale = neuralTimescale

        with torch.no_grad():
            self.W.add_(torch.eye(hidden_size).mul_(1-1/self.neuralTimescale).to_sparse())

    def create_layers(self, input_size, output_size, hidden_size,
                    cell, bptttrunc, continuousTheta,
                    k, f, **cell_kwargs):
        
        #Sparsity via layernorm subtraction
        mu = norm.ppf(f)
        musig = [mu,1]

        self.rnn = thetaRNNLayer(cell, bptttrunc, input_size, hidden_size,
                                    defaultTheta=k, continuousTheta=continuousTheta,
                                    musig=musig, **cell_kwargs)
        
        self.outlayer = nn.Sequential(
            nn.Linear(hidden_size, output_size, bias=False),
            nn.Sigmoid())
        
    def restructure_inputs(self, obs_in, obs_target, act, batched=False): #obs --> obs-in, obs_target 
        """
        Join obs and act into a single input tensor shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
        obs should be one timestep longer than act, for the [t+1] observation
        after the last action
        """

        #Apply the action and prediction offsets
        act = self.batched_actpad(act) if batched else self.actpad(act)
       
        if self.actOffset: 
            act = act[:,:-self.actOffset,...]

        if obs_target is None: # if obs_target not already provided (e.g. through multimodal...)
            obs_target = obs_in[:,self.predOffset:,:] 

        #Make everything the same size
        minsize = min(obs_in.size(1),act.size(1),obs_target.size(1))
        obs_in, act = obs_in[:,:minsize,:], act[:,:minsize,:]
        obs_target = obs_target[:,:minsize,:]

        #Apply the masks (this is ugly.)
        actmask = np.resize(np.array(self.actMask),minsize)
        outmask = np.resize(np.array(self.outMask),minsize)
        obsmask = np.resize(np.array(self.inMask),minsize)

        obs_out = torch.zeros_like(obs_in, requires_grad=False)
        act_out = torch.zeros_like(act, requires_grad=False)
        obs_target_out = torch.zeros_like(obs_target, requires_grad=False)

        obs_out[:,obsmask,:] = obs_in[:,obsmask,:]
        act_out[:,actmask,:] = act[:,actmask,:]
        obs_target_out[:,outmask,:] = obs_target[:,outmask,:]
        
        obs_out = self.droplayer(obs_out) #dropout without action
        #act_out = self.droplayer_act(act_out) #dropout without action

        #Concatenate the obs/act into a single input
        x_t = torch.cat((obs_out,act_out), 2)
        return x_t, obs_target_out, outmask

class pRNN_th(pRNN): 
    def __init__(self, obs_size, act_size, k, hidden_size=500,
                cell=RNNCell,  dropp=0, bptttrunc=50, f=0.5,
                predOffset=0, inMask=[True], outMask=None,
                actOffset=0, actMask=None, neuralTimescale=2,
            continuousTheta=False, actionTheta=False,
            **cell_kwargs):
        

        super(pRNN_th, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                cell=cell,  dropp=dropp, bptttrunc=bptttrunc, k=k, f=f,
                predOffset=predOffset, inMask=inMask, outMask=outMask,
                actOffset=actOffset, actMask=actMask,
                neuralTimescale=neuralTimescale,
                continuousTheta=continuousTheta,
                **cell_kwargs)
        
        self.k = k
        self.actionTheta = actionTheta
        self.obspad=(0,0,0,0,0,k)
        self.batched_obspad=(0,0,0,0,0,0,0,k)
    
    def restructure_inputs(self, obs_in, obs_target, act, batched=False):
        #TODO: Define!!
        return super().restructure_inputs(obs_in, obs_target, act, batched)

class pRNN_multimodal(pRNN):

    def __init__(self, obs_size, act_size, hidden_size=500,
                 cell=RNNCell,  dropp=0, bptttrunc=50, k=0, f=0.5,
                 predOffset=1, inMask=[True], outMask=None, 
                 actOffset=0, actMask=None, neuralTimescale=2,
                 continuousTheta=False, inIDs=None, outIDs=None,
                 **cell_kwargs):
        
        self.obs_size = obs_size
        self.inIDs = inIDs
        self.outIDs = outIDs
        
        n=0
        self.obs_out_slices = []
        for i in outIDs: #TODO: clarify with dan what this is
            self.obs_out_slices.append(slice(n,n+obs_size[i]))
            n += obs_size[i]

        #calculate sizes for multimodal "create_layers"
        self.input_size = 0
        self.output_size = 0
        for i in self.inIDs:
                self.input_size += obs_size[i]
        for i in self.outIDs:
                self.output_size += obs_size[i]
        self.input_size += act_size

        super(pRNN_multimodal, self).__init__(obs_size, act_size, hidden_size,
                                              cell,  dropp, bptttrunc, k, f,
                                              predOffset, inMask, outMask, 
                                              actOffset, actMask, neuralTimescale,
                                              continuousTheta,  input_size = self.input_size, 
                                              output_size = self.output_size, #fix in a seconds
                                              **cell_kwargs)

    def restructure_inputs(self, obs, act, batched=False):

        # Specify inputs... 
        obs_in = []
        for i in self.inIDs:
            obs_in.append(obs[i])
        obs_in = torch.cat(obs_in, 2)

        #... and outputs in the observation
        obs_target = []
        for i in self.outIDs:
            obs_target.append(obs[i][:,self.predOffset:,:])
        obs_target = torch.cat(obs_target, 2)

        return super().restructure_inputs(obs_in, obs_target, act, batched)