import numpy as np
from prnn.utils.agent import RandomActionAgent
import torch


class decodeAnalysis:
    def __init__(self, predictiveNet, decoder='train', timesteps=2000, agent=None):
        
        self.pN = predictiveNet
        self.decoder = decoder
        
        env = predictiveNet.EnvLibrary[0]
        # this is for backward compatibility, better provide the agent as an arg
        if not agent:
            action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
            agent = RandomActionAgent(env.action_space, action_probability)
        
        self.WAKEactivity = self.runWAKE(env, agent, timesteps)
        
        self.decoded, self.p = self.pN.decode(self.WAKEactivity['h'],self.decoder)
        #derror, dshuffle = self.pN.decode_error(decoded, state)
        
        
        
        
    def runWAKE(self, env, agent, timesteps_wake, theta='expand'):
        print('Running WAKE')
        a = {}
        a['obs'],a['act'],a['state'],_ = self.pN.collectObservationSequence(env,
                                                             agent,
                                                             timesteps_wake)
        a['obs_pred'], a['obs_next'], h = self.pN.predict(a['obs'],a['act'])
        
        if theta == 'mean':
            h = h.mean(axis=0,keepdims=True)
            a['act'] = a['act'][:,:h.size(dim=1),:]
            a['state']['agent_pos'] = a['state']['agent_pos'][:h.size(dim=1)+1,:]
        if theta == 'expand':
            k = h.size(dim=0)
            nt = h.size(dim=1)
            h = h.transpose(0,1).reshape((-1,1,h.size(dim=2))).swapaxes(0,1)
            a['obs_pred'] = a['obs_pred'].transpose(0,1).reshape((-1,1,a['obs_pred'].size(dim=2))).swapaxes(0,1)
            obs_temp = torch.zeros_like(a['obs_pred'])
            obs_temp[:,::k,:]=a['obs'][:,:nt,:]
            a['obs'] = obs_temp
            a['state']['agent_pos'] = np.repeat( a['state']['agent_pos'], k, axis=0)
            a['state']['agent_pos'] = a['state']['agent_pos'][:h.size(dim=1)+1,:]
            a['state']['agent_dir'] = np.repeat( a['state']['agent_dir'], k, axis=0)
            a['state']['agent_dir'] = a['state']['agent_dir'][:h.size(dim=1)+1]
            
            
        #a['h'] = np.squeeze(h.detach().numpy())
        a['h'] = h
        return a
    
    def decodeSequenceFigure(self, netname=None, savefolder=None, 
                             timesteps = 2002,trajectoryWindow=100):
        obs = self.WAKEactivity['obs']
        render = None
        obs_pred = self.WAKEactivity['obs_pred']
        state = self.WAKEactivity['state']
        p = self.p
        decoded = self.decoded
        decoder = self.decoder
        
        self.pN.plotObservationSequence(obs, render, obs_pred, state,
                                         p_decode=p, decoded=decoded, mask=decoder.mask,
                                         timesteps=range(timesteps-10,timesteps),
                                         savefolder=savefolder, savename=netname,
                                         trajectoryWindow=trajectoryWindow)