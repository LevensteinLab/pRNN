#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from prnn.utils.agent import RandomActionAgent, RandomHDAgent
import prnn.analysis.trajectoryAnalysis as trajectoryAnalysis
from prnn.utils.general import saveFig, savePkl, loadPkl
from sklearn import manifold
from sklearn.linear_model import LinearRegression
import copy
from prnn.utils.general import state2nap
from scipy.stats import entropy
from prnn.utils.general import delaydist
from torch import tensor, zeros_like
from scipy.stats import spearmanr

class OfflineTrajectoryAnalysis:
    def __init__(self, predictiveNet, noisemag = 0, noisestd=0.1,
                 timesteps_sleep=2000, decoder = 'train', actionAgent=None,
                 withIsomap = False, timesteps_wake = 2000,
                 suppressPrint = False, withTransitionMaps = True,
                 skipContinuity = False, sleepOnsetTransient=0, compareWake=False,
                 compareDiffusion=False, decoderbatches = 10000, wakeAgent=None, 
                 withAdapt=False, b_adapt = 0.5, tau_adapt=8,
                 calculateDiffusionFit=True, calculateViewSimilarity=False):
        self.pN = predictiveNet
        self.pN_sleep = self.pN
        if withAdapt:
            self.pN_sleep = makeAdaptingNet(self.pN, b_adapt, tau_adapt)
        env = predictiveNet.EnvLibrary[0]

        # this is for backward compatibility, better provide the wakeAgent as an arg
        if not wakeAgent:
            action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
            wakeAgent = RandomActionAgent(env.action_space, action_probability)
        
        if decoder == 'train':
            self.decoder = self.trainDecoder(decoderbatches, wakeAgent,
                                             trainHDDecoder = calculateViewSimilarity)
        else:
            self.decoder = decoder
            
        self.SLEEPactvity = self.runSLEEP(noisemag, noisestd, 
                                         timesteps_sleep, actionAgent,
                                         suppressPrint,
                                         onsetTransient = sleepOnsetTransient)
        self.SLEEPdecoded = self.decodeSLEEP(self.SLEEPactvity['h'],self.decoder)
        self.coverageContinuity = self.calculateCoverageContinuity(self.SLEEPdecoded,
                                                                  skipContinuity=skipContinuity)
        

        self.diffusionFit=False
        if calculateDiffusionFit:
            self.diffusionFit = self.calculateDiffusionFit(self.SLEEPdecoded[0])
        
        self.transitionMaps = False
        if withTransitionMaps:
            self.transitionMaps = self.calculateTransitionMaps(self.SLEEPdecoded,
                                                                 self.SLEEPactvity)
        if compareWake or withIsomap or calculateViewSimilarity:
            self.WAKEactity = self.runWAKE(env, wakeAgent, timesteps_wake)
        
        self.compareWake = compareWake
        if compareWake:
            #WAKEdecoded = self.decodeSLEEP(self.WAKEactity['h'],self.decoder)
            state_nap = state2nap(self.WAKEactity['state'])
            self.coverageContinuity_WAKE = self.calculateCoverageContinuity((state_nap,None),
                                                                  skipContinuity=skipContinuity)
            
            self.SWsimilarity = self.compareSW(self.coverageContinuity[1],self.coverageContinuity_WAKE[1])
            
        if compareDiffusion:
            continuity_DIFFUS = self.makeDiffusion()
            self.SDsimilarity = self.compareSW(self.coverageContinuity[1],continuity_DIFFUS)

        self.compareViewSimilarity = calculateViewSimilarity
        if calculateViewSimilarity:
            self.ViewSimilarity = self.calculateViewSimilarity(env, self.WAKEactity, self.SLEEPactvity)
            
            
        self.Isomap = withIsomap
        if self.Isomap:
            self.Isomap = self.fitIsomap()
    
    
    
    def calculateViewSimilarity(self, env, WAKEactivity, SLEEPactivity):
        h = tensor(np.expand_dims(WAKEactivity['h'],0))
        decodedViewObs, decoded = getDecodedViewObs(self.pN, h, self.decoder)
        #error = decodedViewObs.detach().numpy() - WAKEactivity['obs_pred'].detach().numpy()
        #MSE_wake = np.mean(np.square(error),axis=2)
        (MSE_wake,mean_wake,std_wake) = delayCorr(np.squeeze(decodedViewObs.detach().numpy()).T, 
                                    np.squeeze(WAKEactivity['obs_pred'].detach().numpy()).T)
        
        #error = WAKEactivity['obs_next'].detach().numpy() - WAKEactivity['obs_pred'].detach().numpy()
        #MSE_real = np.mean(np.square(error),axis=2)
        (MSE_real,mean_real,std_real) = delayCorr(np.squeeze(WAKEactivity['obs_next'].detach().numpy()).T, 
                                    np.squeeze(WAKEactivity['obs_pred'].detach().numpy()).T)
        
        h = SLEEPactivity['h']
        decodedViewObs, decoded = getDecodedViewObs(self.pN, h, self.decoder)
        #error = decodedViewObs.detach().numpy() - SLEEPactivity['obs_pred'].detach().numpy()
        #MSE_sleep = np.mean(np.square(error),axis=2)
        (MSE_sleep,mean_sleep,std_sleep) = delayCorr(np.squeeze(decodedViewObs.detach().numpy()).T, 
                                    np.squeeze(SLEEPactivity['obs_pred'].detach().numpy()).T)
        
        viewSimilarity = {
            'MSE_wake' : MSE_wake,
            'MSE_real' : MSE_real,
            'MSE_sleep': MSE_sleep,
            'meanstd_wake' : (mean_wake,std_wake),
            'meanstd_real' : (mean_real,std_real),
            'meanstd_sleep' : (mean_sleep,std_sleep),
            'decodedViewObs' : decodedViewObs,
        }
        return viewSimilarity      
        
    
    @staticmethod
    def calculateDiffusionFit(decodedpos, dtmax = 15):
        delays = np.arange(1,dtmax+1)
        _, meandistsq = delaydist(decodedpos.values, dtmax, 
                                  sqdist=True, dist='euclidian')
        #Fit the linear regression between log t and log mean distance squared
        x = np.log10(delays).reshape(-1, 1)
        y = np.log10(meandistsq).T
        
        diffusionFit = {
            'delays' : delays,
            'msd' : meandistsq,
            'r_sq' : 0,
            'intercept' : np.nan,
            'alpha' : 0,
        }
        
        try:
            model = LinearRegression().fit(x, y)
            diffusionFit['r_sq'] = model.score(x, y)
            diffusionFit['intercept'] = model.intercept_       
            diffusionFit['alpha'] = model.coef_[0]
        except:
            print('fit fail. sorry kiddo')
            
        return diffusionFit
    
    @staticmethod
    def compareSW(continuity_SLEEP,continuity_WAKE):
        S = continuity_SLEEP['delay_dist']
        W = continuity_WAKE['delay_dist']
        eps = 1e-12
        SWsimilarity = np.sum(entropy(S+eps,W+eps,axis=0))
        FP = np.zeros_like(S)
        FP[0,:] = 1
        FPsimilarity = np.sum(entropy(FP+eps,W+eps,axis=0))
        SWsimilarity = SWsimilarity/FPsimilarity
        #Here: normalize to fixed point dx=0
        return SWsimilarity
    
    @staticmethod
    def makeDiffusion(dtmax = 11, dxmax = 15, D=0.6):
        #Bounds of x and r should match what's calculated for slep.
        #D should match the probability of forward action...
        #None of this should be hard coded ;)
        t = np.arange(1,dtmax)
        r = np.arange(0,dxmax)
        T,R = np.meshgrid(t,r)
        diffuse = np.exp(-(R**2)/(4*D*T))/(2*np.pi*D*T)
        diffuse = diffuse/np.sum(diffuse,axis=0)
        continuity_DIFFUS = {'delay_dist':diffuse}
        return continuity_DIFFUS
    
    def calculateTransitionMaps(self, decoded, activity, transrange = 3):
        # Not done for the continuous envs yet
        decodedpos = decoded[0].values
        transitionprob,edges = self.calculateTransitionProbability(decodedpos,
                                                                   transrange=transrange)
        
        transitionMaps = {
            'transitionprob': transitionprob,
            'edges': edges,
        }
        
        if type(activity['noise_t']) is tuple:
            actions = activity['noise_t'][1]
        
            numacts = 4
            numHDs = 4
            
            #FOr HD-only cue
            if not actions[:,:,:numacts].any():
                actions[:,:,numacts-1] = 1
            
            HDaligned = [None for act in range(4)]
            AllTransProb = [[]for act in range(4)]
            for act in range(numacts):
                thisact = actions[:,:,act]==1
                HDaligned[act] = np.zeros((2*transrange+1,2*transrange+1))
                for HD in range(numHDs):
                    thisHD = actions[:,:,-HD-1]==1
                    restrict = np.squeeze(thisact&thisHD)
                    transitionprob, edges = self.calculateTransitionProbability(decodedpos,
                                                                            restrict=restrict,
                                                                            transrange=transrange)
                    AllTransProb[act].append(transitionprob)
                    HDaligned[act] = HDaligned[act] + np.rot90(transitionprob,HD-1)
                HDaligned[act] = HDaligned[act]/numHDs
        
            transitionMaps['HDaligned'] = HDaligned
            transitionMaps['AllTransProb'] = AllTransProb

        return transitionMaps
        
            
    def calculateTransitionProbability(self, decodedpos,
                                       restrict=None, transrange=3):

        xdif = decodedpos[1:,0]-decodedpos[:-1,0]
        ydif = decodedpos[1:,1]-decodedpos[:-1,1]
        
        if restrict is not None:
            xdif = xdif[restrict[:len(xdif)]]
            ydif = ydif[restrict[:len(ydif)]]

        bins = np.linspace(-transrange-0.5,transrange+0.5,2*transrange+2)
        transitionhist,edges,_ = np.histogram2d(xdif,ydif,bins)
        transitionhist = transitionhist/np.sum(transitionhist[:])
        
        return transitionhist,edges
            
    def transitionProbabilityPanel(self, transitionprob,marker='+',
                                  vmax=None, incLabels = True):
        transitionhist, edges = transitionprob
        
        plt.imshow(transitionhist,extent=(edges[0],edges[-1],
                                  edges[0],edges[-1]),
                  vmin=0, vmax=vmax)
        plt.plot(0,0,'r',marker=marker)
        plt.xticks([])
        plt.yticks([])
        if incLabels:
            plt.xlabel('dx')
            plt.ylabel('dy')
            plt.colorbar(label='P[dx]')
            
            
    def transitionProbabilityFigure(self,netname, savefolder):
        transitionMaps = self.transitionMaps
        arrowicons = ['<','^','>','v']
        actionLabels = ['Turn R','Turn L','Forward','Hold']
        numacts = 4
        numHDs = 4
            
        plt.figure()
        for act in range(numacts):
            plt.subplot(4,4,act+1)
            self.transitionProbabilityPanel((transitionMaps['HDaligned'][act],
                                                 transitionMaps['edges']),
                                                marker='^',vmax=0.2, incLabels=False)
            plt.title(actionLabels[act])
            if act == 3:
                plt.colorbar(label='P[dx]')
            
            for HD in range(numHDs):
                plt.subplot(6,4,act+4*HD+9)
                self.transitionProbabilityPanel((transitionMaps['AllTransProb'][act][HD],
                                                 transitionMaps['edges']),
                                                marker=arrowicons[HD],
                                                vmax=0.2, incLabels=False)
        if netname is not None:
            saveFig(plt.gcf(),netname+'_TransitionProbability',savefolder,
                    filetype='pdf')
        plt.show()
        
    
    
    def trainDecoder(self, numbatches = 10000, trainHDDecoder = False, agent=None):
        env = self.pN.EnvLibrary[0]
        _,_,decoder = self.pN.calculateSpatialRepresentation(env,
                                                             agent,
                                                             numBatches=numbatches,
                                                             trainDecoder=True,
                                                             trainHDDecoder=trainHDDecoder)
        return decoder
    
    def runSLEEP(self, noisemag, noisestd, timesteps_sleep, actionAgent, suppressPrint=False,
                onsetTransient=0):
        # Not adapted for RiaB yet
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
        #a['h'] = np.squeeze(h_t.detach().numpy())
        #a['h'] = h_t
        
        #Remove the onset transient
        a['obs_pred'] = a['obs_pred'][:,onsetTransient:,:]
        a['h'] = a['h'][:,onsetTransient:,:]
        if type(a['noise_t']) is tuple:
            a['noise_t'] = (a['noise_t'][0][:,onsetTransient:,:],
                            a['noise_t'][1][:,onsetTransient:,:])
        else:
            actions = a['noise_t'][:,onsetTransient:,:]
            
        return a
    
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
            h = h.transpose(0,1).reshape((-1,1,h.size(dim=2))).swapaxes(0,1)
            a['obs_pred'] = a['obs_pred'].transpose(0,1).reshape((-1,1,a['obs_pred'].size(dim=2))).swapaxes(0,1)
            a['obs_next'] = a['obs_next'].transpose(0,1).reshape((-1,1,a['obs_next'].size(dim=2))).swapaxes(0,1)
            obs_temp = zeros_like(a['obs_pred'])
            obs_temp[:,::k,:]=a['obs'][:,:-k,:]
            a['obs'] = obs_temp
            a['state']['agent_pos'] = np.repeat( a['state']['agent_pos'], k, axis=0)
            a['state']['agent_pos'] = a['state']['agent_pos'][:h.size(dim=1)+1,:]
            a['state']['agent_dir'] = np.repeat( a['state']['agent_dir'], k, axis=0)
            a['state']['agent_dir'] = a['state']['agent_dir'][:h.size(dim=1)+1]


        a['h'] = np.squeeze(h.detach().numpy())
        return a
    
    def decodeSLEEP(self,h,decoder):
        decoded, p = self.pN.decode(h,decoder)
        return decoded, p
    
    def calculateCoverageContinuity(self,decoded, skipContinuity=False):
        decoded, p = decoded
        
        continuity = None
        if skipContinuity is False:
            continuity = trajectoryAnalysis.calculateContinuity(decoded)
        
        coverage = trajectoryAnalysis.calculateCoverage(decoded,
                                                        [0,self.decoder.gridheight,
                                                         0,self.decoder.gridwidth])
        
        maxp,dx,pdhist = None,None,None
        if p is not None:
            maxp = np.max(p,axis=(1,2))
            dx = np.sum(np.abs(decoded.values[:-1,:] - decoded.values[1:,:]), axis=1)

            pdhist,pbinedges,dtbinedges = np.histogram2d(dx,np.log10(maxp[:-1]),
                                    bins=[np.arange(0,15),np.linspace(-1,0,11)])
        
        return pdhist, continuity, coverage, maxp, dx, p
    
    def fitIsomap(self):
        print('Fitting Isomap')
        #X = self.WAKEactity['h']
        h_wake = self.WAKEactity['h']
        h_sleep = np.squeeze(self.SLEEPactvity['h'].detach().numpy())
        X = np.concatenate((h_wake,h_sleep))
        method = manifold.Isomap(n_neighbors=50, n_components=3,p=1)
        method.fit(X)
        return method

    
    
    def SpontTrajectoryFigure(self, netname, savefolder, trajRange=(0,20)):
        pdhist, continuity, coverage, maxp, dx, p = self.coverageContinuity
        obs_pred = self.SLEEPactvity['obs_pred']
        obs_pred = self.pN.env_shell.pred2np(obs_pred[:,range(trajRange[0],trajRange[0]+10),...])
        h = self.SLEEPactvity['h']
        decoded, _ = self.SLEEPdecoded
        if self.transitionMaps:
            transitionprob = (self.transitionMaps['transitionprob'],
                              self.transitionMaps['edges'])
        
        
        
        
        plt.figure(figsize=(10,10))
        plt.subplot(6,5,30)
        self.pdPanel(pdhist)

        plt.subplot(6,5,10)
        self.continuityDistPanel(continuity)

        plt.subplot(6,4,7)
        self.occupancyDistPanel(coverage)
        
        if self.compareWake:
            plt.subplot(6,4,24)
            self.occupancyDistPanel(self.coverageContinuity_WAKE[2])

        plt.subplot(6,2,1)
        self.pdxTimeseries(maxp, dx)
        
        plt.subplot(6,4,3)
        self.trajectoryPanel(decoded,trajRange=trajRange)

        plt.subplot(6,2,3)
        self.pN.plotActivationTimeseries(h)

        self.pN.plotSequence(obs_pred, range(10), 5, label='Predicted', numrows=9)
        
        self.pN.plotSequence(np.transpose(np.log10(p),axes=[0,2,1]), 
                             range(trajRange[0],trajRange[0]+10),
                             4, label='Decoded',
                             mask=self.decoder.mask.transpose(),
                             cmap='bone', vmin=-3.25, vmax=0, numrows=9)
        if self.compareViewSimilarity:
            obs_view = self.ViewSimilarity['decodedViewObs']
            obs_view = self.pN.env_shell.pred2np(obs_view[:,range(trajRange[0],trajRange[0]+10),...])
            self.pN.plotSequence(obs_view, range(10), 6, label='Predicted', numrows=9)
        #plt.colorbar()
        plt.subplot(6,5,5)
        self.diffusionPanel(self.diffusionFit)

        if self.Isomap:
            plt.subplot(3,3,7,projection='3d')
            self.isomapPanel()
            
        if self.transitionMaps:
            plt.subplot(3,3,8)
            self.transitionProbabilityPanel(transitionprob, vmax=0.2)
        
        if self.compareViewSimilarity:
            plt.subplot(6,3,13)
            self.compareViewStatsPanel(self.ViewSimilarity)
        
            plt.subplot(6,3,16)
            self.compareViewDelay(self.ViewSimilarity)
        
        if netname is not None:
            saveFig(plt.gcf(),netname+'_SpontaneousTrajectory',savefolder,
                    filetype='pdf')

        plt.show()

    def compareViewStatsPanel(self,ViewSimilarity):
        MSE_real = ViewSimilarity['MSE_real'][0]
        MSE_sleep = ViewSimilarity['MSE_sleep'][0]
        MSE_wake = ViewSimilarity['MSE_wake'][0]
        
        plt.boxplot([np.squeeze(MSE_real),np.squeeze(MSE_wake),np.squeeze(MSE_sleep)],
                   labels=['W-Real','W-Decode','S-Decode'],
                   showfliers=False)
        plt.ylabel('pixel corr')
        
    def compareViewDelay(self,ViewSimilarity):
        MSE_real = ViewSimilarity['meanstd_real'][0]
        MSE_sleep = ViewSimilarity['meanstd_sleep'][0]
        MSE_wake = ViewSimilarity['meanstd_wake'][0]
        
        plt.plot(MSE_real,label='W-real')
        plt.plot(MSE_wake,label='W-decode')
        plt.plot(MSE_sleep,label='S-decode')
        plt.ylabel('pixel corr')
        plt.xlabel('Delay')
        plt.legend()

    def diffusionPanel(self, diffusionFit):
        delays = diffusionFit['delays']
        msd = diffusionFit['msd']
        intercept = diffusionFit['intercept']
        alpha = diffusionFit['alpha']
        
        plt.plot(np.log10(delays),np.log10(msd),'o')
        plt.plot([0,np.log10(delays[-1])],
                 [intercept,intercept+alpha*np.log10(delays[-1])],'r')
        plt.text(0.75,intercept,
                 f"{np.round(alpha,2)}",
                 fontsize=10,color='r')
        plt.xlabel('log(time)')
        plt.ylabel('log(mean sq dist)')
        
    
    def isomapPanel(self, duration=20):
        X = self.WAKEactity['h']
        Y_WAKE = self.Isomap.transform(X)
        state = self.WAKEactity['state']
        c_WAKE = np.arctan(state['agent_pos'][:-1,0]/state['agent_pos'][:-1,1])
        
        X = np.squeeze(self.SLEEPactvity['h'].detach().numpy())
        Y_SLEEP = self.Isomap.transform(X)
        c_SLEEP = np.ones((X.shape[0]))            
        
        ax = plt.gca()
        ax.scatter(Y_WAKE[:, 0], Y_WAKE[:, 1], Y_WAKE[:, 2], 
                   c=c_WAKE, marker='.', s=4)
        ax.scatter(Y_SLEEP[:, 0], Y_SLEEP[:, 1], Y_SLEEP[:, 2], 
                   c=[0.5,0.5,0.5], marker='.', s=2)
        ax.plot3D(Y_SLEEP[:duration,0], Y_SLEEP[:duration,1], 
                   Y_SLEEP[:duration,2],'k')
        ax.scatter(Y_SLEEP[:duration,0], Y_SLEEP[:duration,1], 
                   Y_SLEEP[:duration,2], c=range(duration),s=30,
                  cmap='cubehelix')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        #ax.set_title(colorvar)
        return
        
    def trajectoryPanel(self, decoded, trajRange=(0,20)):
        x = decoded['x'].values
        y= decoded['y'].values
        plt.imshow(np.ones_like(self.decoder.mask.transpose()),
                   interpolation='nearest',alpha=self.decoder.mask.transpose())
        plt.plot(x[trajRange[0]:trajRange[1]],
                 y[trajRange[0]:trajRange[1]],'r')
        plt.scatter(x[trajRange[0]:trajRange[1]],
                    y[trajRange[0]:trajRange[1]],
                    c=range(trajRange[1]-trajRange[0]), marker='.',
                    cmap='cubehelix')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.xlim(0,self.decoder.gridwidth)
        plt.ylim(0,self.decoder.gridheight)
                                                        
    
    def pdPanel(self, pdhist):
        #plt.plot(np.log10(maxp[:-1]),dx,'.')
        plt.imshow(pdhist[0:9,:],origin='lower',
                   extent=(-1,-0,-0.5,9.5),aspect='auto')
        plt.xlabel('log p_x')
        plt.ylabel('dx')
        
    def continuityDistPanel(self, continuity):
        numdelays = continuity['delays'][-1]
        maxdist = continuity['dxbins'][-1]+1
        
        ax1 = plt.gca()
        # ax2 = ax1.twinx()
        ax1.imshow(continuity['delay_dist'],origin='lower',
                   extent=(0.5,numdelays+0.5,-0.5,maxdist-0.5),
                   aspect='auto', cmap='binary')
        # ax2.plot(continuity['delays'],continuity['delay_kl'],'.--r')
        # ax1.text(numdelays-2,maxdist-2, f"{continuity['underthresh']}",
        #          fontsize=10,color='r')
        ax1.set_xlabel('dt')
        ax1.set_ylabel('dx')
        # ax2.set_ylabel('K-L Div from P[dx]')
    
    def occupancyDistPanel(self, coverage):
        
        palette = copy.copy(plt.get_cmap('viridis'))
        palette.set_under('grey', 1.0)  # 1.0 represents not transparent
        
        plt.imshow(coverage['occupancy'].transpose(),
                interpolation='nearest',alpha=self.decoder.mask.transpose(),
                   cmap=palette, vmin=0.001)
        plt.text(1, 1, f"{1-coverage['nonuniformity']:0.1}", fontsize=10,color='r')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.title('Occupancy')
        
    def pdxTimeseries(self, maxp, dx):
        timesteps = len(dx)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(dx[range(30)],color='grey')
        ax1.plot(maxp[range(30)],color='k')
        plt.xlabel('t')
        ax2.set_ylabel('dx')
        ax1.set_ylabel('px')
        
        
    def saveAnalysis(self, savename, savefolder=None):
        #TODO: default to an analysis folder in pN.savefolder
        savename = savename+'_OfflineTrajectoryAnalysis'
        savePkl(self,savename,savefolder)
        print("Analysis Saved to pathname")
        
    def loadAnalysis(savename, savefolder=None, suppressText=False):
        savename = savename+'_OfflineTrajectoryAnalysis'
        analysis = loadPkl(savename,savefolder)
        if not suppressText:
            print("Analysis Loaded from pathname")
        return analysis
        


def getDecodedViewObs(pN, h, decoder):
    decoded, p = pN.decode(h, decoder, withHD=True)
    
    decodedViewObs = []
    for decodedView in decoded.values:
        fromdecode = pN.env_shell.get_viewpoint(decodedView[0:2], decodedView[2])
        decodedViewObs.append(fromdecode.flatten())
    
    decodedViewObs = np.expand_dims(np.stack(decodedViewObs),0)
    decodedViewObs = tensor(decodedViewObs, requires_grad=False)
    return decodedViewObs, decoded


def delayCorr(a, b, maxDelay=15):
    corr,p = spearmanr(a, b)
        
    N = a.shape[1]
    delaycorr = []
    delaycorr_mean = []
    delaycorr_std = []
    delays = range(0,maxDelay)
    for t in delays:
        delaycorr.append(np.diag(corr,k=N+t))
        delaycorr_mean.append(np.mean(delaycorr[-1]))
        delaycorr_std.append(np.mean(delaycorr[-1]))
    delaycorr_mean= np.array(delaycorr_mean)
    delaycorr_std= np.array(delaycorr_std)
        
    return delaycorr, delaycorr_mean, delaycorr_std
        
        
        
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