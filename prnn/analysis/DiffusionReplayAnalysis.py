from prnn.utils.general import delaydist
import numpy as np
from prnn.utils.agent import RandomActionAgent, RandomHDAgent
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from prnn.utils.general import state2nap
import prnn.analysis.trajectoryAnalysis as trajectoryAnalysis
from matplotlib import cm
import copy
from prnn.utils.general import saveFig
from prnn.analysis.representationalGeometryAnalysis import representationalGeometryAnalysis as rg
from prnn.analysis.OfflineTrajectoryAnalysis import calculateSpatialCoherence


class DiffusionReplayAnalysis:
    def __init__(self, predictiveNet, noisemag = 0, noisestd=0.05,
                 timesteps_sleep=300, num_trials = 15, dtmax=15, decoder = 'train', 
                 actionAgent=None, sleepOnsetTransient=10, 
                 decoderbatches = 10000, compareWake=False, wakeAgent=None, 
                 withAdapt=False, b_adapt = 0.4, tau_adapt=5):
        self.pN = predictiveNet
        self.pN_sleep = self.pN
        if withAdapt:
            self.pN_sleep = makeAdaptingNet(self.pN, b_adapt, tau_adapt)

        self.noisemag = noisemag
        self.noisestd = noisestd
        self.timesteps_sleep = timesteps_sleep
        self.num_trials = num_trials
        self.actionAgent = actionAgent
        # this is for backward compatibility, better provide the agent as an arg
        if wakeAgent:
            self.wakeAgent = wakeAgent
        else:
            action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
            self.wakeAgent = RandomActionAgent(self.pN.env_shell.action_space, action_probability)
        self.sleepOnsetTransient = sleepOnsetTransient
        
        if decoder == 'train':
            self.decoder = self.trainDecoder(decoderbatches)
        else:
            self.decoder = decoder
            
        (self.msd, self.delays, 
         self.delay_dist, self.coverage,
         self.coherence,self.extent) = self.runOfflineTrials(num_trials, noisemag, noisestd, 
                                                 timesteps_sleep, actionAgent, sleepOnsetTransient,
                                                           dtmax=dtmax)
        self.diffusionFit = self.calculateDiffusionFit(self.msd, self.delays)
        
        self.compareWake = compareWake
        if compareWake:
            (self.msd_WAKE, _, 
             self.delay_dist_WAKE,
            self.coverage_WAKE,_,_) = self.runOfflineTrials(num_trials, noisemag, noisestd, 
                                                           timesteps_sleep, actionAgent, 
                                                           sleepOnsetTransient,
                                                           wakeControl = True,
                                                           dtmax=dtmax)
            self.diffusionFit_WAKE = self.calculateDiffusionFit(self.msd_WAKE, self.delays)



    def runSTDPanel(self, minStd = 0.01, maxStd= 1, numStds=11):
        noisestds = np.logspace(np.log10(minStd),np.log10(maxStd),numStds)
        STDPanel = {
            'noisestds' : noisestds,
            'alpha'     : np.zeros_like(noisestds),
            'r_sq'      : np.zeros_like(noisestds),
            'intercept' : np.zeros_like(noisestds),
            'msd'       : [[] for x in noisestds],
            'delay_dist': [[] for x in noisestds],
            'coverage': [[] for x in noisestds],
            'coherence': [[] for x in noisestds],
            'extent': [[] for x in noisestds],
        }
        
        for nidx, noise in enumerate(noisestds):
            print(nidx)
            (msd, delays, delay_dist, coverage, 
            coherence, extent) = self.runOfflineTrials(self.num_trials, 
                                                self.noisemag, noise,
                                                self.timesteps_sleep, 
                                                self.actionAgent, 
                                                self.sleepOnsetTransient,
                                                dtmax=self.delays.size)
            diffusionFit = self.calculateDiffusionFit(msd, delays)
            STDPanel['alpha'][nidx] = diffusionFit['alpha']
            STDPanel['r_sq'][nidx] = diffusionFit['r_sq']
            STDPanel['intercept'][nidx] = diffusionFit['intercept']
            STDPanel['msd'][nidx] = msd
            STDPanel['delay_dist'][nidx] = delay_dist
            STDPanel['coverage'][nidx] = coverage
            STDPanel['coherence'][nidx] = coherence
            STDPanel['extent'][nidx] = extent
        self.STDPanel = STDPanel
        
    
        
    def runOfflineTrials(self, num_trials, noisemag, noisestd, 
                         timesteps_sleep, actionAgent, sleepOnsetTransient,
                         dtmax=15, wakeControl=False):
        msd = np.zeros((num_trials,dtmax))
        delays = np.arange(1,dtmax+1)
        cohere = []
        extent = []
        #print('Running Sleep Trials')
        for trial in range(num_trials):
            if wakeControl:
                activity = self.runWAKE(timesteps_sleep)
                decoded = state2nap(activity['state'])
                spatialCoherence = {'cohere':None,
                                   'extent': None}
            else:
                actvity = self.runSLEEP(noisemag, noisestd,
                                        timesteps_sleep, actionAgent,
                                        suppressPrint = True,
                                        onsetTransient = sleepOnsetTransient)
                decoded = self.decodeSLEEP(actvity['h'],self.decoder)
                spatialCoherence = calculateSpatialCoherence(decoded)
                decoded = decoded[0]
            
            
            delay_dist, msd[trial,:] = delaydist(decoded.values,dtmax, sqdist=True,
                                                dist='euclidian')
            cohere.append(spatialCoherence['cohere'])
            extent.append(spatialCoherence['extent'])
        
        cohere = np.vstack(cohere)
        extent = np.vstack(extent)
            
        coverage = trajectoryAnalysis.calculateCoverage(decoded,
                                                        [0,self.decoder.gridheight,
                                                         0,self.decoder.gridwidth])
        
        return msd, delays, delay_dist, coverage, cohere, extent
        
    def trainDecoder(self, numbatches = 10000):
        env = self.pN.EnvLibrary[0]
        _,_,decoder = self.pN.calculateSpatialRepresentation(env,
                                                             self.wakeAgent,
                                                             numBatches=numbatches,
                                                             trainDecoder=True)
        return decoder
    
    
    def runWAKE(self, timesteps_wake):
        env = self.pN.EnvLibrary[0]
        #print('Running WAKE')
        a = {}
        a['obs'],a['act'],a['state'],_ = self.pN.collectObservationSequence(env,
                                                             self.wakeAgent,
                                                             timesteps_wake)
        a['obs_pred'], a['obs_next'], h = self.pN.predict(a['obs'],a['act'])
        a['h'] = np.squeeze(h.detach().numpy())
        return a
    
    def runSLEEP(self, noisemag, noisestd, timesteps_sleep, 
                 actionAgent, suppressPrint=False, onsetTransient=0):
        # Not adapted for RiaB and farama-minigrid yet
        if not suppressPrint:
            print('Running SLEEP')
        if actionAgent is True:
            env = self.pN.EnvLibrary[0]
            #action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
            action_probability = np.array([0.1,0.1,0.7,0.1,0,0,0])
            actionAgent = RandomActionAgent(env.action_space,action_probability)
        if actionAgent == 'HDOnly':
            env = self.pN.EnvLibrary[0]
            action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
            actionAgent = RandomHDAgent(env.action_space,action_probability)
        if actionAgent == 'RandHDForward':
            env = self.pN.EnvLibrary[0]
            action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
            actionAgent = RandomHDAgent(env.action_space,action_probability,constantAction=2)
        if actionAgent == 'OneHDForward':
            env = self.pN.EnvLibrary[0]
            action_probability = np.array([0,0,1,0,0,0,0])
            actionAgent = RandomActionAgent(env.action_space,action_probability)
        if actionAgent == 'OneHDStop':
            env = self.pN.EnvLibrary[0]
            action_probability = np.array([0,0,0,1,0,0,0])
            actionAgent = RandomActionAgent(env.action_space,action_probability)
            
        a = {}
        a['obs_pred'],a['h'],a['noise_t'] = self.pN_sleep.spontaneous(timesteps_sleep,
                                                   noisemag,
                                                   noisestd,agent=actionAgent)
        
        #Remove the onset transient
        a['obs_pred'] = a['obs_pred'][:,onsetTransient:,:]
        a['h'] = a['h'][:,onsetTransient:,:]
        if type(a['noise_t']) is tuple:
            a['noise_t'] = (a['noise_t'][0][:,onsetTransient:,:],
                            a['noise_t'][1][:,onsetTransient:,:])
        else:
            actions = a['noise_t'][:,onsetTransient:,:]
            
        return a
    
    def decodeSLEEP(self,h,decoder):
        decoded, p = self.pN.decode(h,decoder)
        return decoded, p
    
    
    
    @staticmethod
    def calculateDiffusionFit(msd, delays):
        #Fit the linear regression between log t and log mean distance squared
        delays = delays.repeat(msd.shape[0])
        msd = msd.T.flatten()
        
        if (msd==0).any():
            print('setting zero-values to min (0.01)')
            zerovals = msd==0
            msd[zerovals] = 0.01
            #delays = np.delete(delays,zerovals)
            
        x = np.log10(delays).reshape(-1, 1)
        y = np.log10(msd).reshape(-1, 1)
        
        diffusionFit = {
            'delays' : delays,
            'msd' : msd,
            'r_sq' : 0,
            'intercept' : np.nan,
            'alpha' : 0,
        }
        
        try:
            model = LinearRegression().fit(x, y)
            diffusionFit['r_sq'] = model.score(x, y)
            diffusionFit['intercept'] = model.intercept_[0]        
            diffusionFit['alpha'] = model.coef_[0][0]
        except:
            print('fit fail. sorry kiddo')
            
        return diffusionFit
    
    
    
    def DiffusionFigure(self, netname=None, savefolder=None, halflims=True):
        msd_wake = None
        if self.compareWake is not False:
            msd_wake = self.msd_WAKE
        
        plt.figure()
        
        plt.subplot(2,2,1)
        self.diffusionFitPanel(self.delays, self.msd, self.diffusionFit, msd_wake,
                              showTrialPoints=False, halflims=halflims)
        
        #plt.subplot(2,4,1)
        #self.extentPanel
        
        if netname is not None:
            saveFig(plt.gcf(),netname+'_DiffusionFigure',savefolder,
                    filetype='png')
        
        plt.show()
        
    def diffusionFitPanel(self,delays, msd, diffusionFit = None, msd_wake=None,
                          showTrialPoints=True, color='k', halflims = True, errorshade=False):
        
        mean_msd = np.mean(np.log10(msd),axis=0, where=msd>0)
        std_msd = np.nanstd(np.log10(msd), where=msd>0)
        
        if msd_wake is not None:
            mean_msd_W = np.mean(np.log10(msd_wake),axis=0)
            std_msd_W = np.std(np.log10(msd_wake))
            #plt.plot(np.log10(delays),np.log10(mean_msd_W),'ko')
            if errorshade:
                plt.fill_between(np.log10(delays), mean_msd_W-std_msd_W, mean_msd_W+std_msd_W,
                                color='grey', alpha=0.1)
                plt.plot(np.log10(delays),mean_msd_W,'-',
                             color='grey')
            else:
                plt.errorbar(np.log10(delays),mean_msd_W,
                             yerr=std_msd_W,uplims=halflims,
                             elinewidth = 1,
                             fmt='.',color='grey', )

        if showTrialPoints:
            plt.plot(np.log10(delays),np.log10(msd.T),'.',color='lightgrey')
        
        if errorshade:
            plt.fill_between(np.log10(delays), mean_msd-std_msd, mean_msd+std_msd,
                                color=color, alpha=0.1)
            plt.plot(np.log10(delays),mean_msd,'-',
                         color=color,)
        else:
            plt.errorbar(np.log10(delays),mean_msd,
                         yerr=std_msd,lolims=halflims,
                         elinewidth = 1,
                         color=color,fmt='.')
        
        if diffusionFit is not None:
            intercept = diffusionFit['intercept']
            alpha = diffusionFit['alpha']
            r_sq = diffusionFit['r_sq']
        
        
            plt.plot([0, np.log10(delays[-1])],[intercept, intercept+alpha*np.log10(delays[-1])],'r')
            plt.text(.01, .98, f'alpha: {alpha:.2f}', ha='left', va='top', 
                     color='r', transform=plt.gca().transAxes)
            plt.text(.01, .89, f'R^2:   {r_sq:.2f}', ha='left', va='top', 
                     color='r', transform=plt.gca().transAxes)

        #plt.plot(np.log10(delays),np.log10(mean_msd),'-',color=color)
        #plt.plot(np.log10(delays),np.log10(mean_msd),'.',color=color)
        
        
        plt.xlabel('log(dt)')
        plt.ylabel('log(mean(dx^2))')
        
        

            
            
    def STDPanelFigure(self, netname=None, savefolder=None):
        wakeNoise = self.pN.trainNoiseMeanStd[1]
        if self.compareWake is not False:
            msd_wake = self.msd_WAKE
        
        numnoise = len(self.STDPanel['noisestds'])
        cmap = cm.get_cmap('viridis', numnoise)  
        
        plt.figure(figsize=(13,8))
        plt.subplot(3,3,8)
        plt.plot(np.log10(self.STDPanel['noisestds']),self.STDPanel['alpha'],'grey')
        plt.plot(np.log10(self.STDPanel['noisestds']),np.zeros(numnoise),'k--')
        plt.plot(np.log10(self.STDPanel['noisestds']),
                 self.diffusionFit_WAKE['alpha']*np.ones(numnoise),'k')
        plt.scatter(np.log10(self.STDPanel['noisestds']),self.STDPanel['alpha'],
                    c= np.arange(numnoise))
        plt.plot(np.log10(wakeNoise),self.diffusionFit_WAKE['alpha'],'kv')
        plt.xlabel('log(noise)')
        plt.ylabel('alpha')
        
        
        #plt.subplot(3,3,9)
        #plt.plot(np.log10(self.STDPanel['noisestds']),self.STDPanel['r_sq'],'grey')
        #plt.scatter(np.log10(self.STDPanel['noisestds']),self.STDPanel['r_sq'],
        #            c= np.arange(numnoise))
        #plt.ylim([0,1])
        #plt.xlabel('log(noise)')
        #plt.ylabel('R^2')
        
        
        plt.subplot(3,3,9)
        plt.plot(np.log10(self.STDPanel['noisestds']),self.STDPanel['intercept'],'grey')
        plt.plot(np.log10(self.STDPanel['noisestds']),
                 self.diffusionFit_WAKE['intercept']*np.ones(numnoise),'k')
        plt.scatter(np.log10(self.STDPanel['noisestds']),self.STDPanel['intercept'],
                    c= np.arange(numnoise))
        plt.plot(np.log10(wakeNoise),self.diffusionFit_WAKE['intercept'],'k^')
        # plt.ylim([0,1])
        plt.xlabel('log(noise)')
        plt.ylabel('Intercept')
        
        
        extents = [i.flatten() for i in self.STDPanel['extent']]
        plt.subplot(3,3,7)
        plt.plot(np.log10(self.STDPanel['noisestds']),np.mean(extents,axis=1),'grey')
        #plt.plot(np.log10(self.STDPanel['noisestds']),
        #         self.diffusionFit_WAKE['extent']*np.ones(numnoise),'k')
        plt.scatter(np.log10(self.STDPanel['noisestds']),np.mean(extents,axis=1),
                    c= np.arange(numnoise))
        #plt.boxplot(extents,
        #           labels=np.log10(self.STDPanel['noisestds']).round(decimals=2),
        #           showfliers=False)
        #plt.plot(np.log10(wakeNoise),self.diffusionFit_WAKE['extent'],'k^')
        #plt.ylim([0,1])
        plt.xlabel('log(noise)')
        plt.ylabel('extent')
        
        
        
                       
        plt.subplot(3,3,3)
        self.diffusionFitPanel(self.delays, self.STDPanel['msd'][0],
                               None, msd_wake, showTrialPoints=False,
                               color=cmap(0/numnoise),
                              halflims = False)
        for sidx,std in enumerate(self.STDPanel['msd']):
            if sidx in [4,6,8]:
                self.diffusionFitPanel(self.delays, self.STDPanel['msd'][sidx], 
                                       None, None, showTrialPoints=False,
                                       color=cmap(sidx/numnoise),
                                      halflims = False)
            
            
        plt.subplot(6,numnoise, 1)
        self.continuityDistPanel(self.delay_dist_WAKE)
                
        plt.subplot(6,numnoise, 2)
        self.occupancyDistPanel(self.coverage_WAKE)
        
        
        for sidx, delay_dist in enumerate(self.STDPanel['delay_dist']):
            plt.subplot(6,numnoise, sidx+3*numnoise+1)
            self.continuityDistPanel(delay_dist)
            if sidx > -1:
                plt.gca().set_ylabel(None)
                plt.gca().set_xlabel(None)
                plt.gca().axis('off')
                
            plt.subplot(6,numnoise,sidx+2*numnoise+1)
            self.occupancyDistPanel(self.STDPanel['coverage'][sidx])
         
        #plt.tight_layout()
        if netname is not None:
            saveFig(plt.gcf(),netname+'_STDPanel',savefolder,
                    filetype='png')
        plt.show()
        
        
    def continuityDistPanel(self,delay_dist):
        numdelays = delay_dist.shape[1]
        maxdist = delay_dist.shape[0]
        
        plt.imshow(delay_dist,origin='lower',
                   extent=(0.5,numdelays+0.5,-0.5,maxdist-0.5),
                   aspect='auto', cmap='binary')
        plt.xlabel('dt')
        plt.ylabel('dx')
        
    def occupancyDistPanel(self,coverage):

            palette = copy.copy(plt.get_cmap('viridis'))
            palette.set_under('grey', 1.0)  # 1.0 represents not transparent

            plt.imshow(coverage['occupancy'].transpose(),
                    interpolation='nearest',alpha=self.decoder.mask.transpose(),
                       cmap=palette, vmin=0.001)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            
            
            
            
            
def makeAdaptingNet(pN, b, tau_a):
    from prnn.utils.thetaRNN import LayerNormRNNCell, AdaptingLayerNormRNNCell, RNNCell, AdaptingRNNCell
    from torch.nn import Parameter
    import torch
    
    adapting_pN = pN.copy()

    input_size = pN.pRNN.rnn.cell.input_size
    hidden_size = pN.pRNN.rnn.cell.hidden_size
    if isinstance(pN.pRNN.rnn.cell,LayerNormRNNCell):
        musig = [pN.pRNN.rnn.cell.layernorm.mu,pN.pRNN.rnn.cell.layernorm.sig]
        adapting_pN.pRNN.rnn.cell = AdaptingLayerNormRNNCell(input_size, hidden_size, musig)
    elif isinstance(pN.pRNN.rnn.cell,RNNCell):
        adapting_pN.pRNN.rnn.cell = AdaptingRNNCell(input_size, hidden_size)
    else:
        print('Your RNN cell is not yet supported')
    adapting_pN.pRNN.rnn.cell.weight_ih = copy.deepcopy(pN.pRNN.rnn.cell.weight_ih)
    adapting_pN.pRNN.rnn.cell.weight_hh = copy.deepcopy(pN.pRNN.rnn.cell.weight_hh)
    adapting_pN.pRNN.rnn.cell.bias = copy.deepcopy(pN.pRNN.rnn.cell.bias)

    adapting_pN.pRNN.rnn.cell.b = Parameter(torch.ones(1)*b)
    adapting_pN.pRNN.rnn.cell.tau_a = Parameter(torch.ones(1)*tau_a)
    
    return adapting_pN