from torch import nn
import torch
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import norm
from prnn.utils.thetaRNN import thetaRNNLayer, RNNCell, LayerNormRNNCell, AdaptingLayerNormRNNCell, AdaptingRNNCell, SparseGatedRNNCell, LogNRNNCell

from prnn.utils.pytorchInits import CANN_
from prnn.utils.Architectures import pRNN, pRNN_th, pRNN_multimodal



class vRNN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell):
        super(vRNN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=None, actMask=None)

class thRNN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell):
        super(thRNN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[False,True], actMask=None)

class vRNN_0win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_0win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)

class vRNN_1win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_1win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,True], outMask=[True,True], actMask=None)

class vRNN_2win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_2win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=None)

class vRNN_3win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_3win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=None)

class vRNN_4win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_4win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=None)

class vRNN_5win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_5win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)

class vRNN_1win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_1win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,False])

class vRNN_2win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_2win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=[True,False,False])

class vRNN_3win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_3win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True], 
                          actMask=[True,False,False,False])

class vRNN_4win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_4win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=[True,False,False,False,False])

class vRNN_5win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(vRNN_5win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=[True,False,False,False,False,False])

class thRNN_0win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_0win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)

class thRNN_1win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_1win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=None)

class thRNN_2win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_2win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=None)

class thRNN_3win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_3win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=None)

class thRNN_4win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_4win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=None)

class thRNN_5win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_5win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_6win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_6win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True],
                          actMask=None)
        
        
        
        
        




class thRNN_0win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_0win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)

class thRNN_1win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(thRNN_1win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=None)

class thRNN_2win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(thRNN_2win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=None)

class thRNN_3win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(thRNN_3win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=None)

class thRNN_4win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(thRNN_4win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=None)

class thRNN_5win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(thRNN_5win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_6win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(thRNN_6win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False], outMask=[True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_7win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_7win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_8win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_8win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_9win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_9win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_10win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_10win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
        

class thRNN_1win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thRNN_1win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,False])

class thRNN_2win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thRNN_2win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=[True,False,False])

class thRNN_3win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thRNN_3win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=[True,False,False,False])

class thRNN_4win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thRNN_4win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=[True,False,False,False,False])

class thRNN_5win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thRNN_5win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=[True,False,False,False,False,False])

class thRNN_0win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_0win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True], outMask=[True], actMask=None)

class thRNN_1win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_1win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False], outMask=[True,True],
                          actMask=None)

class thRNN_2win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_2win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=None)

class thRNN_3win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_3win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=None)

class thRNN_4win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_4win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=None)

class thRNN_5win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_5win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_6win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_6win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_7win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_7win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_8win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_8win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_9win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_9win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_10win_prevAct(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thRNN_10win_prevAct, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=1,
                          inMask=[True,False,False,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True,True,True],
                          actMask=None)






class AutoencoderFF(pRNN): 
    """
    Feed-forward autoencoder. 
    Predict obs t at time t, no masked observations. 
    Also zero out gradients and weights for the hidden state, so no temporal dependency, 
    just input --> prediction autoencoder
    """ 
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=None):
        super(AutoencoderFF, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()
        
class AutoencoderRec(pRNN):
    """
    Autoencoder "Recurrent". 
    i.e. Don't zero out the gradients or the weights, keep temporal dependency
    """
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=None):
        super(AutoencoderRec, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderPred(pRNN): #next timestep prediction
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=None):
        super(AutoencoderPred, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderFFPred(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=None):
        super(AutoencoderFFPred, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()
        
        
        
class AutoencoderFF_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderFF_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()
        
class AutoencoderRec_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderRec_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderPred_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(AutoencoderPred_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                                                 f=f,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderFFPred_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15, f=0.5):
        super(AutoencoderFFPred_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                                                    cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, 
                                                    dropp=dropp,f=f,
                                                    predOffset=1, actOffset=0,
                                                    inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()

class AutoencoderMaskedO(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderMaskedO, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[False,True],
                          actMask=[True,True])
        
class AutoencoderMaskedOA(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderMaskedOA, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[False,True],
                          actMask=[True,False])

class AutoencoderMaskedO_noout(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderMaskedO_noout, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,True])
        
class AutoencoderMaskedOA_noout(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(AutoencoderMaskedOA_noout, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,False])


        

        
        
class thcycRNN_3win(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thcycRNN_3win, self).__init__(obs_size, act_size,  k=3, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       predOffset=0, actOffset=0,
                                      )
        
class thcycRNN_4win(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thcycRNN_4win, self).__init__(obs_size, act_size,  k=4, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       predOffset=0, actOffset=0,
                                      )
        
class thcycRNN_5win(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15):
        super(thcycRNN_5win, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True)
        
        
        
class thcycRNN_5win_holdc(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_holdc, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta='hold')
        
class thcycRNN_5win_fullc(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_fullc, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta=True)
        
class thcycRNN_5win_firstc(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_firstc, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta=False)
        
        
class thcycRNN_5win_hold(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_hold, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta='hold')
        
class thcycRNN_5win_full(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(thcycRNN_5win_full, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta=True)
        
class thcycRNN_5win_first(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_first, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta=False)
        
        
     
        
class thcycRNN_5win_holdc_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_holdc_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta='hold')
        
class thcycRNN_5win_fullc_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_fullc_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta=True)
        
class thcycRNN_5win_firstc_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_firstc_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=True, actionTheta=False)
        
        
class thcycRNN_5win_hold_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_hold_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta='hold')
        
class thcycRNN_5win_full_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_full_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta=True)
        
class thcycRNN_5win_first_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_first_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       continuousTheta=False, actionTheta=False)
        
        
class thcycRNN_5win_holdc_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_holdc_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=True, actionTheta='hold')
        
class thcycRNN_5win_fullc_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_fullc_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=True, actionTheta=True)
        
class thcycRNN_5win_firstc_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_firstc_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=True, actionTheta=False)
        
        
class thcycRNN_5win_hold_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_hold_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=False, actionTheta='hold')
        
class thcycRNN_5win_full_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_full_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=False, actionTheta=True)
        
class thcycRNN_5win_first_prevAct(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5):
        super(thcycRNN_5win_first_prevAct, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=cell, bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=1,
                                       continuousTheta=False, actionTheta=False)
        


class lognRNN_rollout(pRNN_th):
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
        
        
        
              
        

class vRNN_LayerNorm(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_LayerNorm, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=LayerNormRNNCell)


class thRNN_LayerNorm(thRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(thRNN_LayerNorm, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=LayerNormRNNCell)


class vRNN_LayerNormAdapt(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_LayerNormAdapt, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=AdaptingLayerNormRNNCell)



class vRNN_CANN(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_CANN, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=LayerNormRNNCell)

        #TODO Clean this
        size = [20,20,20]
        Nmaps = 1
        self.locations = CANN_(self.W, size, Nmaps, selfconnect=False)
        with torch.no_grad():
            self.W.add_(torch.eye(hidden_size).mul_(0.5))



class vRNN_adaptCANN(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_adaptCANN, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=AdaptingLayerNormRNNCell)

        #TODO Clean this
        size = [15,15,15]
        Nmaps = 1
        self.locations = CANN_(self.W, size, Nmaps, selfconnect=False)
        with torch.no_grad():
            self.W.add_(torch.eye(hidden_size).mul_(0.5))




class vRNN_CANN_FFonly(vRNN_CANN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_CANN_FFonly, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias)

        rootk = np.sqrt(1/hidden_size)
        with torch.no_grad():
            self.W.add_(torch.rand(hidden_size, hidden_size)*0.5*rootk)
        self.W.requires_grad=False


class vRNN_adptCANN_FFonly(vRNN_adaptCANN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_adptCANN_FFonly, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias)

        rootk = np.sqrt(1/hidden_size)
        with torch.no_grad():
            self.W.add_(torch.rand(hidden_size, hidden_size)*0.2*rootk)
        self.W.requires_grad=False



class sgpRNN_5win(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                 f=0.5,
                 sparse_size=1000, sparse_beta=1,
                 lambda_direct=1, lambda_context=1, lambda_sparse=1):
        super(sgpRNN_5win, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                       cell=SparseGatedRNNCell,
                                       continuousTheta=False, actionTheta=True, 
                                       bptttrunc=bptttrunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       f=f,
                                       predOffset=0, actOffset=0,
                                       sparse_size=sparse_size, sparse_beta=sparse_beta,
                                       lambda_direct=lambda_direct, lambda_context=lambda_context,
                                       lambda_sparse=lambda_sparse)



class multRNN_5win_i01_o01(pRNN_multimodal):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(multRNN_5win_i01_o01, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None,
                          inIDs=(0,1), outIDs=(0,1),
                          )



class multRNN_5win_i1_o0(pRNN_multimodal):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(multRNN_5win_i1_o0, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None,
                          inIDs=(1,), outIDs=(0,),
                          )



class multRNN_5win_i01_o0(pRNN_multimodal):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(multRNN_5win_i01_o0, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None,
                          inIDs=(0,1), outIDs=(0,),
                          )



class multRNN_5win_i0_o1(pRNN_multimodal):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, bptttrunc=100, neuralTimescale=2, dropp = 0.15,
                f=0.5, **cell_kwargs):
        super(multRNN_5win_i0_o1, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, bptttrunc=bptttrunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None,
                          inIDs=(0,), outIDs=(1,),
                          )