
from torch import nn
import torch
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import norm
from prnn.utils.thetaRNN import thetaRNNLayer, RNNCell, LayerNormRNNCell, AdaptingLayerNormRNNCell, AdaptingRNNCell, SparseGatedRNNCell, LogNRNNCell

from prnn.utils.pytorchInits import CANN_
from abc import ABC, abstractmethod
from functools import partial

## TODOs 
# - go over internal and spontaneous with dan, helper or class method?
# - tests
# - next_step network, masked network, theta sweep

## Helper Functions --




"""
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
"""

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
    Args:
        nn (class): PyTorch nn module
        Base_pRNN (class): Abstract pRNN class
    """
    def __init__(self, obs_size, act_size, hidden_size=500,
                 cell=RNNCell,  dropp=0, bptttrunc=50, k=0, f=0.5,
                 predOffset=1, inMask=[True], outMask=None, 
                 actOffset=0, actMask=None, neuralTimescale=2,
                 continuousTheta=False,
                 **cell_kwargs):
        """_summary_

        Args:
            obs_size (int): Size of each agent observation. Flattened? #TODO confirm with dan
            act_size (int): Size of action vector.
            hidden_size (int, optional): Number of recurrent neurons. Defaults to 500.
            cell (RNNCell, optional): Specified RNNCell type for layers. Defaults to RNNCell.
            dropp (int, optional): Dropout probability. Defaults to 0.
            bptttrunc (int, optional): Backpropagation Through Time, Truncated. Defaults to 50.
            k (int, optional): Number of predictions in rollout.. Defaults to 0.
            f (float, optional): Cumulative probaility, used as an argument into ppf. Defaults to 0.5.
            predOffset (int, optional): At timestep t, how many steps forward do we predict?. Defaults to 1.
            inMask (list, optional): Mask to cover FUTURE timesteps. Defaults to [True].
            outMask (_type_, optional): Mask to cover FUTURE predictions. Defaults to None.
            actOffset (int, optional): Number by which actions should be shifted backward before incorporation into hidden state. Defaults to 0.
            actMask (_type_, optional):  Mask to cover FUTURE actions.. Defaults to None.
            neuralTimescale (int, optional): Decay for timescale. Defaults to 2.
            continuousTheta (bool, optional): Carry over hidden state from the kth rollout to the t+1'th timestep. Defaults to False.
        """
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
                           k, f, **cell_kwargs) #init scheme for the thetaRNNLayer gets passed through here

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
        
    def restructure_inputs(self, obs_in, act, obs_target=None, batched=False): #obs --> obs-in, obs_target 
        """
        Join obs and act into a single input tensor shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
        obs should be one timestep longer than act, for the [t+1] observation
        after the last action

        OBS TARGET SHOULD BE PASSED
        """

        #Apply the action and prediction offsets
        act = self.batched_actpad(act) if batched else self.actpad(act)
       
        if self.actOffset: 
            act = act[:,:-self.actOffset,...]

        if obs_target is None: # if obs_target not already provided (e.g. through multimodal...)
            obs_target = obs_in[:,self.predOffset:,:] 

        x_t, obs_target_out, outmask = self.clip_mask(obs_in, act, obs_target)
        return x_t, obs_target_out, outmask 
        
    def clip_mask(self, obs_in, act, obs_target):
        """ Takes observation tensor, action tensor, and observation targets and clips them.
        Also applies relevant masks if they exist and no rollout is done.
        """

        #clip
        minsize = min(obs_in.size(1),act.size(1),obs_target.size(1))
        obs_in, act = obs_in[:,:minsize,:], act[:,:minsize,:]
        obs_target = obs_target[:,:minsize,:]

        #new tensors for return
        obs_out = torch.zeros_like(obs_in, requires_grad=False)
        act_out = torch.zeros_like(act, requires_grad=False)
        obs_target_out = torch.zeros_like(obs_target, requires_grad=False)

        if self.actMask is not None and self.outMask is not None and self.inMask is not None:
            
            #Apply the masks if exist
            actmask = np.resize(np.array(self.actMask),minsize)
            outmask = np.resize(np.array(self.outMask),minsize)
            obsmask = np.resize(np.array(self.inMask),minsize)

            obs_out[:,obsmask,:] = obs_in[:,obsmask,:]
            act_out[:,actmask,:] = act[:,actmask,:]
            obs_target_out[:,outmask,:] = obs_target[:,outmask,:]
            
            obs_out = self.droplayer(obs_out)
        
        else: #if no masks, or doing roll out
            outmask = True

            obs_out[:] = obs_in[:]
            act_out[:] = act[:]
            obs_target_out[:] = obs_target[:]
            
            obs_in = self.droplayer(obs_in)

        #Concatenate the obs/act into a single input
        x_t = torch.cat((obs_out,act_out), 2)
        return x_t, obs_target_out, outmask

    def forward(self, obs, act, noise_params=(0,0), state=torch.tensor([]), theta=None,
                single=False, mask=None, batched=False, fullRNNstate=False):
        #Determine the noise shape
        k=0
        if hasattr(self,'k'):
            k= self.k
        
        if batched:
            noise_shape = (k+1, obs.size(1), self.hidden_size, obs.size(-1))
        else:
            noise_shape = (k+1, obs.size(1), self.rnn.cell.hidden_size)
            #^^^for backwards compadiblity. change to self.hidden_size later

        noise_t = self.generate_noise(noise_params, noise_shape)

        if single: #if single = true, we only care about updating the hidden state, don't need the prediction
            if hasattr(self, "inIDs"): #if multimodal pRNN...
                for i in self.inIDs:
                    x_t.append(obs[i])
            else:
                x_t = torch.cat((obs,act), 2) 

            h_t,_ = self.rnn(x_t, internal=noise_t, state=state, theta=theta)
            if not fullRNNstate: 
                h_t = h_t[:,:,:self.hidden_size] #For RNNcells that output more than the hidden RNN units
            y_t = None
            obs_target = None
        else:
            x_t, obs_target, outmask = self.restructure_inputs(obs_in=obs, act=act, obs_target = None, batched = batched) #passed as None now, will get filled in restructure_inputs
            #x_t = self.droplayer(x_t) # (should it count action???) dropout with action
            h_t,_ = self.rnn(x_t, internal=noise_t, state=state,
                             theta=theta, mask=mask, batched=batched)
            if not fullRNNstate: 
                h_t = h_t[:,:,:self.hidden_size] #For RNNcells that output more than the hidden RNN units (ugly)
            if batched:# change shape to include batch first and theta last, if we're doing batching. consider this just a black box for making the shape all proper
                h_t = h_t.permute(-1,*[i for i in range(len(h_t.size())-1)])
                allout = self.outlayer(h_t[:,:,:,:self.hidden_size])
                allout = allout.permute(*[i for i in range(1,len(allout.size()))],0)
                h_t = h_t.permute(*[i for i in range(1,len(h_t.size()))],0)
            else:
                allout = self.outlayer(h_t[:,:,:self.hidden_size])

            if hasattr(self, "inIDs") and hasattr(self, "outIDs"): #multimodal model --> may want to change this, not good convention...
                    #Apply the mask to the output
                pred = torch.zeros_like(allout)
                pred[:,outmask,:] = allout[:,outmask,:] #The predicted outputs.
                y_t = [] # Outputs disentangled
                for i in set((*self.inIDs, *self.outIDs)):
                    if i in set(self.inIDs)-set(self.outIDs):
                        y_t.append(torch.zeros_like(obs[i]))
                    else:
                        y_t.append(pred[...,self.obs_out_slices[self.outIDs.index(i)]])
                y_t = (pred, y_t)
            else:
                #Apply the mask to the output
                y_t = torch.zeros_like(allout)
                y_t[:,outmask,:] = allout[:,outmask,:] #The predicted outputs.
        
        return y_t, h_t, obs_target

    def generate_noise(self, noise_params, shape):
        if noise_params != (0,0):
            noise = noise_params[0] + noise_params[1]*torch.randn(shape, device=self.W.device)
        else:
            noise = torch.zeros(shape, device=self.W.device)

        return noise

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
    
    #override
    def restructure_inputs(self, obs_in, act, obs_target = None, batched=False):
        """
        Join obs and act into a single input tensor shape (N,L,H)
        N: Batch size/Theta Size
        L: timesamps
        H: input_size
        obs should be one timestep longer than act, for the [t+1] observation
        after the last action
        """
        #Apply the action and prediction offsets
        if batched:
            act = self.batched_actpad(act)
            obspad = self.batched_obspad
        else:
            act = self.actpad(act)
            if ~hasattr(self,'obspad'):   #For backwards compadibility - remove later
                self.obspad = (0,0,0,0,0,self.k)
            obspad = self.obspad

        obs_target = obs_in[:,self.predOffset:,:]
        
        #Apply the theta prediction for target observation
        theta_idx = np.flip(toeplitz(np.arange(self.k+1),
                                     np.arange(obs_target.size(1))),0)
        theta_idx = theta_idx[:,self.k:,]
        obs_target = obs_target[:,theta_idx.copy()]
        obs_target = torch.squeeze(obs_target,0)
        
        if self.actionTheta == 'hold':
            size = [x for x in act.size()] # So it works with batched and non-batched
            size[0] = self.k+1
            act = act.expand(*size)
            obs_in = nn.functional.pad(input=obs_in, pad=obspad, 
                                    mode='constant', value=0)

        elif self.actionTheta is True:
            theta_idx = np.flip(toeplitz(np.arange(self.k+1),
                                         np.arange(act.size(1))),0)
            theta_idx = theta_idx[:,self.k:,]
            act = act[:,theta_idx.copy()]
            act = torch.squeeze(act,0)
            obs_in = nn.functional.pad(input=obs_in, pad=obspad, 
                                    mode='constant', value=0)
            
        
        x_t, obs_target_out, outmask = self.clip_mask(obs_in, act, obs_target)
        return x_t, obs_target_out, outmask 


class pRNN_multimodal(pRNN):

    def __init__(self, obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell,  dropp=0, bptttrunc=50, k=0, f=0.5,
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

        #backwards compatibility: all constructor args stay the same, 
        # but if you can use cell_kwargs to pass in some additional ones

        if "use_LN" in cell_kwargs:
            cell = LayerNormRNNCell if cell_kwargs["use_LN"] else RNNCell

        if "inMask_length" in cell_kwargs: #make the mask boolean arrays
            inMask_length = cell_kwargs["inMask_length"]
            
            inMask = np.full(inMask_length + 1, False)
            inMask[0] = True #timestep t

            outMask = np.full(inMask_length + 1, True)


        super(pRNN_multimodal, self).__init__(obs_size, act_size, hidden_size,
                                              cell,  dropp, bptttrunc, k, f,
                                              predOffset, inMask= inMask, outMask = outMask, 
                                              actOffset, actMask, neuralTimescale,
                                              continuousTheta,  input_size = self.input_size, 
                                              output_size = self.output_size, #fix in a seconds
                                              **cell_kwargs)
    #override
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

        return super().restructure_inputs(obs_in=obs_in, act=act, obs_target=obs_target, batched = batched)

 
class NextStepRNN(pRNN):
    """
    A predictive RNN that uses an observation and action at timestep t to predict the next observation (at t+1).

    Args:
        pRNN (class): Extends regular pRNN class.
    """
    def __init__(self, obs_size, act_size, hidden_size = 500, 
                 bptttrunc = 100, neuralTimescale = 2, 
                 dropp = 0.15, f = 0.5, 
                 use_LN = True, use_FF = False, **cell_kwargs):
        """Initialize NextStepRNN.

        Args:
            obs_size (int): Size of each agent observation. Flattened? #TODO confirm with dan
            act_size (int): Size of action vector.
            hidden_size (int, optional): Number of recurrent neurons. Defaults to 500. Defaults to 500.
            bptttrunc (int, optional): Backpropagation Through Time, Truncated.. Defaults to 100.
            neuralTimescale (int, optional): Decay for timescale.. Defaults to 2.
            dropp (float, optional): Dropout probability. Defaults to 0.15.
            f (float, optional): Cumulative probaility, used as an argument into ppf. Defaults to 0.5.
            use_LN (bool, optional): Use LayerNorm? Defaults to True.
            use_FF (bool, optional): Use Feed Forward network? Defaults to False. If True, we get rid of recurrence by zeroing out the hidden-to-hidden weight matrix.
        """
        cell = LayerNormRNNCell if use_LN else RNNCell
        predOffset = cell_kwargs["predOffset"] if "predOffset" in cell_kwargs else 1
        print("inside the init function of NextStepRNN...")
        
        super().__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, 
                          neuralTimescale=neuralTimescale, 
                          dropp=dropp, f=f, predOffset=predOffset, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)        
        if use_FF:
            self.W.requires_grad_(False)
            self.W.zero_()

class MaskedRNN(pRNN):
    """ 
    A predictive RNN with some combination of masked input observations, actions, or predictions.
    Also accepts an offset for actions.

    Args:
        pRNN (class): extends base pRNN class
    """
 
    def __init__(self, obs_size, act_size, hidden_size=500,
                bptttrunc=100, neuralTimescale=2, 
                dropp = 0.15, f=0.5, 
                use_LN = True, mask_actions = False, actOffset = 0, inMask_length= 0, **cell_kwargs): #new additions
        """Initialize MaskedRNN.

        Args:
            obs_size (int): Size of each agent observation. Flattened? #TODO confirm with dan
            act_size (int): Size of action vector.
            hidden_size (int, optional): Number of recurrent neurons. Defaults to 500.
            bptttrunc (int, optional): Backpropagation Through Time, Truncated. Defaults to 100.
            neuralTimescale (int, optional): Decay for timescale. Defaults to 2.
            dropp (float, optional): Dropout probability. Defaults to 0.15.
            f (float, optional): Cumulative probaility, used as an argument into ppf. Defaults to 0.5.
            use_LN (bool, optional): Use Layer Norm? Defaults to True (no LayerNorm).
            mask_actions (bool, optional): Mask Actions as well? Note that the action mask will the same as the input observation mask. Defaults to False.
            actOffset (int, optional): Number of timesteps to offset actions by (backwards). Defaults to 0.
            inMask_length (int, optional): Number of FUTURE timesteps to mask. Model will continue output predictions. Defaults to 0.
        """
        cell = LayerNormRNNCell if use_LN else RNNCell
    
        inMask = np.full(inMask_length + 1, False)
        inMask[0] = True #timestep t

        actMask = inMask if mask_actions else None #set the action mask to be the same as the input obs mask
        outMask = np.full(inMask_length + 1, True)

        super().__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=actOffset,
                          inMask=inMask, outMask=outMask,
                          actMask=actMask)

class RolloutRNN(pRNN_th):
    """
    A predictive RNN that does a rollout of k predictions at each timestep.
    Options to offset actions, use the same future actions for the k rollouts, use the same action, or no actions.

    Args:
        pRNN_th (class): extends pRNN class with "theta" rollouts.
    """
    def __init__(self,obs_size, act_size, hidden_size=500,
                bptttrunc=100, neuralTimescale=2, dropp = 0.15, f=0.5,
                use_ALN = False, k = 5, rollout_action = "first", continuousTheta = False, actOffset = 0, **cell_kwargs):
        """Initialize RolloutRNN.

        Args:
            obs_size (int): Size of each agent observation. Flattened? #TODO confirm with dan
            act_size (int): Size of action vector.
            hidden_size (int, optional): Number of recurrent neurons. Defaults to 500.
            bptttrunc (int, optional): Backpropagation Through Time, Truncated. Defaults to 100.
            neuralTimescale (int, optional): Decay for timescale. Defaults to 2.
            dropp (float, optional): Dropout probability. Defaults to 0.15.
            f (float, optional): Cumulative probaility, used as an argument into ppf. Defaults to 0.5.
            use_ALN (bool, optional): Use Adaptive Layer Norm?. Defaults to False (plain LayerNorm)
            k (int, optional): Number of predictions in rollout. Defaults to 5.
            rollout_action (str, optional): Action structure. Defaults to "first" (use only the one action). Other options: "full" (use real future actions);  "hold" (use same action for k steps?)
            continuousTheta (bool, optional): Carry over hidden state from the kth rollout to the t+1'th timestep. Defaults to False (carry hidden state from t to t+1).
            actOffset (int, optional): Number of timesteps to offset actions by (backwards). Defaults to 0.
        """
        cell = AdaptingLayerNormRNNCell if use_ALN else LayerNormRNNCell

        actionTheta = True if rollout_action == "full" else \
                    False if rollout_action == "first" else \
                    "hold"
                    
        super().__init__(obs_size, act_size,  k=k, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=actOffset,
                                       continuousTheta=continuousTheta, actionTheta=actionTheta)


""" Next-step Prediction Networks"""

nAutoencoderFF = partial(NextStepRNN, use_LN = False, use_FF = True, predOffset = 0)
nAutoencoderRec = partial(NextStepRNN, use_LN = False, use_FF = False, predOffset = 0)
nAutoencoderPred = partial(NextStepRNN, use_LN = False, use_FF = False, predOffset = 1)
nAutoencoderFFPred = partial(NextStepRNN, use_LN = False, use_FF = True, predOffset = 1)
nAutoencoderFF_LN = partial(NextStepRNN, use_LN = True, use_FF = True, predOffset = 0)
nAutoencoderRec_LN = partial(NextStepRNN, use_LN = True, use_FF = False, predOffset = 0)
nAutoencoderPred_LN = partial(NextStepRNN, use_LN = True, use_FF = False, predOffset = 1)
nAutoencoderFFPred_LN = partial(NextStepRNN, use_LN = True, use_FF = True, predOffset = 1)

""" Masked Prediction Networks """

nthRNN = partial(MaskedRNN, use_LN = False, inMask_length = 1) #has no extra stuff like neuralTimescale, bptttrunc...etc

nthRNN_0win_noLN = partial(MaskedRNN, use_LN = False, inMask_length = 0)
nthRNN_1win_noLN = partial(MaskedRNN, use_LN = False, inMask_length = 1)
nthRNN_2win_noLN = partial(MaskedRNN, use_LN = False, inMask_length = 2)
nthRNN_3win_noLN = partial(MaskedRNN, use_LN = False, inMask_length = 3)
nthRNN_4win_noLN = partial(MaskedRNN, use_LN = False, inMask_length = 4)
nthRNN_5win_noLN = partial(MaskedRNN, use_LN = False, inMask_length = 5)
nthRNN_6win_noLN = partial(MaskedRNN, use_LN = False, inMask_length = 6)

nthRNN_0win = partial(MaskedRNN, use_LN = True, inMask_length = 0)
nthRNN_1win = partial(MaskedRNN, use_LN = True, inMask_length = 1)
nthRNN_2win = partial(MaskedRNN, use_LN = True, inMask_length = 2)
nthRNN_3win = partial(MaskedRNN, use_LN = True, inMask_length = 3)
nthRNN_4win = partial(MaskedRNN, use_LN = True, inMask_length = 4)
nthRNN_5win = partial(MaskedRNN, use_LN = True, inMask_length = 5)
nthRNN_6win = partial(MaskedRNN, use_LN = True, inMask_length = 6)
nthRNN_7win = partial(MaskedRNN, use_LN = True, inMask_length = 7)
nthRNN_8win = partial(MaskedRNN, use_LN = True, inMask_length = 8)
nthRNN_9win = partial(MaskedRNN, use_LN = True, inMask_length = 9)
nthRNN_10win = partial(MaskedRNN, use_LN = True, inMask_length = 10)

nthRNN_0win_mask = partial(MaskedRNN, use_LN = True, inMask_length = 0, mask_actions = True)
nthRNN_1win_mask = partial(MaskedRNN, use_LN = True, inMask_length = 1, mask_actions = True)
nthRNN_2win_mask = partial(MaskedRNN, use_LN = True, inMask_length = 2, mask_actions = True)
nthRNN_3win_mask = partial(MaskedRNN, use_LN = True, inMask_length = 3, mask_actions = True)
nthRNN_4win_mask = partial(MaskedRNN, use_LN = True, inMask_length = 4, mask_actions = True)
nthRNN_5win_mask = partial(MaskedRNN, use_LN = True, inMask_length = 5, mask_actions = True)

nthRNN_0win_prevAct = partial(MaskedRNN, use_LN = True, inMask_length = 0, actOffset=1)
nthRNN_1win_prevAct = partial(MaskedRNN, use_LN = True, inMask_length = 1, actOffset=1)
nthRNN_2win_prevAct = partial(MaskedRNN, use_LN = True, inMask_length = 2, actOffset=1)
nthRNN_3win_prevAct = partial(MaskedRNN, use_LN = True, inMask_length = 3, actOffset=1)
nthRNN_4win_prevAct = partial(MaskedRNN, use_LN = True, inMask_length = 4, actOffset=1)
nthRNN_5win_prevAct = partial(MaskedRNN, use_LN = True, inMask_length = 5, actOffset=1)
nthRNN_6win_prevAct = partial(MaskedRNN, use_LN = True, inMask_length = 6, actOffset=1)
nthRNN_7win_prevAct = partial(MaskedRNN, use_LN = True, inMask_length = 7, actOffset=1)
nthRNN_8win_prevAct = partial(MaskedRNN, use_LN = True, inMask_length = 8, actOffset=1)
nthRNN_9win_prevAct = partial(MaskedRNN, use_LN = True, inMask_length = 9, actOffset=1)
nthRNN_10win_prevAct = partial(MaskedRNN, use_LN = True, inMask_length = 10, actOffset=1)

""" Rollout Prediction Networks """

nthcycRNN_3win = partial(RolloutRNN, use_ALN = False, k = 3)
nthcycRNN_4win = partial(RolloutRNN, use_ALN = False, k = 4)
nthcycRNN_5win = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = True) #TODO check with dan whether its supposed to be continueous theta

nthcycRNN_5win_holdc = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = True, rollout_action = "hold")
nthcycRNN_5win_fullc = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = True, rollout_action = "full")
nthcycRNN_5win_firstc = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = True, rollout_action = "first")

nthcycRNN_5win_hold = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = False, rollout_action = "hold")
nthcycRNN_5win_full = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = False, rollout_action = "full")
nthcycRNN_5win_first = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = False, rollout_action = "first")

nthcycRNN_5win_holdc_adapt = partial(RolloutRNN, use_ALN = True, k = 5, continuousTheta = True, rollout_action = "hold")
nthcycRNN_5win_fullc_adapt = partial(RolloutRNN, use_ALN = True, k = 5, continuousTheta = True, rollout_action = "full")
nthcycRNN_5win_firstc_adapt = partial(RolloutRNN, use_ALN = True, k = 5, continuousTheta = True, rollout_action = "first")

nthcycRNN_5win_hold_adapt = partial(RolloutRNN, use_ALN = True, k = 5, continuousTheta = False, rollout_action = "hold")
nthcycRNN_5win_full_adapt = partial(RolloutRNN, use_ALN = True, k = 5, continuousTheta = False, rollout_action = "full")
nthcycRNN_5win_first_adapt = partial(RolloutRNN, use_ALN = True, k = 5, continuousTheta = False, rollout_action = "first")

nthcycRNN_5win_holdc_prevAct = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = True, rollout_action = "hold", actOffset = 1)
nthcycRNN_5win_fullc_prevAct = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = True, rollout_action = "full", actOffset = 1)
nthcycRNN_5win_firstc_prevAct = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = True, rollout_action = "first", actOffset = 1)

nthcycRNN_5win_hold_prevAct = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = False, rollout_action = "hold", actOffset = 1)
nthcycRNN_5win_full_prevAct = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = False, rollout_action = "full", actOffset = 1)
nthcycRNN_5win_first_prevAct = partial(RolloutRNN, use_ALN = False, k = 5, continuousTheta = False, rollout_action = "first", actOffset = 1)

""" Log Normal Initialization"""

#use LayerNormCell, no more LogNRNNCell
nlognRNN_rollout = partial(RolloutRNN, k = 5, continuousTheta = False, rollout_action = "full", init = "log_normal")
nlognRNN_mask = partial(MaskedRNN, use_LN = True, inMask_length = 5, init = "log_normal")


""" Multimodal pRNNs """ 

#though this could be made more efficient, i'm keeping the pRNN_multimodal arguments the same for the sake of backwards compatibility

nmultRNN_5win_i01_o01 = partial(pRNN_multimodal, use_LN = True, inMask_length = 5, predOffset = 0, inIDs=(0,1), outIDs=(0,1))
nmultRNN_5win_i1_o0 = partial(pRNN_multimodal, use_LN = True, inMask_length = 5, predOffset = 0, inIDs=(1,), outIDs=(0))
nmultRNN_5win_i01_o0 = partial(pRNN_multimodal, use_LN = True, inMask_length = 5, predOffset = 0, inIDs=(0,1), outIDs=(0,))
nmultRNN_5win_i0_o1 = partial(pRNN_multimodal, use_LN = True, inMask_length = 5, predOffset = 0, inIDs=(0,), outIDs=(1,))


""" OLD ARCHITECTURES: TESTING CONTRUCTORS FOR BACK COMPATIBILITY """

class oAutoencoderPred_LN(pRNN):
    """
    Autoencoder prediction, next step prediction, no obs or action masks, no offset. Yes LayerNorm
    """
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(oAutoencoderPred_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                                                 f=f,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)

class othRNN_5win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(othRNN_5win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)
        
class othcycRNN_5win_full(pRNN_th):
    """
    Rollout RNN; k = 5 predictions per timestep, use true future actions through rollout, continue to t+1 after kth rollout.
    """
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(othcycRNN_5win_full, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta=True)

class lognRNN_rollout(pRNN_th):
    """
    log normal weight initializations, rollout rnn with k =5 predictions per time steps, use true future action policies, no continue after k predictions
    """
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LogNRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(lognRNN_rollout, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta=True,
                                       **cell_kwargs)

class lognRNN_mask(pRNN):
    """
    log normal weight initializations, rollout rnn with k =5 predictions per time steps, use true future action policies, mask input obs  
    """
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LogNRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(lognRNN_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None,
                          **cell_kwargs)
        