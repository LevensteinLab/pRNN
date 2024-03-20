#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:35:12 2022

@author: dl2820
"""
import numpy as np
from utils.general import saveFig
import matplotlib.pyplot as plt
from utils.agent import RandomActionAgent
from analysis.representationalGeometryAnalysis import representationalGeometryAnalysis as rg
import analysis.trajectoryAnalysis as trajectoryAnalysis
from utils.general import state2nap
from analysis.OfflineTrajectoryAnalysis import OfflineTrajectoryAnalysis
compareSW = OfflineTrajectoryAnalysis.compareSW
makeDiffusion = OfflineTrajectoryAnalysis.makeDiffusion
calculateDiffusionFit = OfflineTrajectoryAnalysis.calculateDiffusionFit

class SpontaneousActivityAnalysis:
    def __init__(self, predictiveNet, compareWAKEagent=None,
                 murange=0.5, stdmin =0.01, stdmax=1, numpoints=11, timesteps=1000,
                 examples=((0,0.1),(0,0.25),(0,0.4)), actionAgent=None,
                 wgain = 1, onset_transient=100,
                 savename=None, savefolder=None, decoder=None):
        self.pN = predictiveNet
        self.actionAgent = actionAgent
        self.wgain = wgain
        self.onset = onset_transient
        
        self.decoder = decoder
        if decoder == 'train':
            decoder = self.trainDecoder()
            self.decoder = decoder
        
        self.wakeStats = self.runWakeComparison(timesteps,decoder)  
        
        self.noisemags = np.linspace(-murange,murange,numpoints)
        self.noisestds = np.logspace(np.log10(stdmin),np.log10(stdmax),numpoints)
        
        self.spontStats = self.runSpontaneousSims(self.noisemags, self.noisestds, 
                                             decoder, timesteps, self.wakeStats[2], 
                                                 self.wakeStats[6])
        
        self.exampleSims = self.runSamples(examples,timesteps, decoder, self.wakeStats[2],
                                          self.wakeStats[6])
        

        
    def trainDecoder(self):
        env = self.pN.EnvLibrary[0]
        action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
        agent = RandomActionAgent(env.action_space,action_probability)
        _, _, decoder = self.pN.calculateSpatialRepresentation(env,
                                                               agent,
                                                               numBatches=10000,
                                                               trainDecoder=True)
        return decoder
        
    def runSim_getResults(self, timesteps, noisemag, noisestd, decoder, 
                          h_wake, n_trials=1, wake_continuity=None):
        wgain=self.wgain
        actionAgent = self.actionAgent
        if actionAgent is True:
            env = self.pN.EnvLibrary[0]
            action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
            actionAgent = RandomActionAgent(env.action_space,action_probability)
            
        _,h,_ = self.pN.spontaneous(timesteps, noisemag, noisestd,
                                    wgain=wgain, agent=actionAgent)
        actStats = self.pN.calculateActivationStats(h)
        meanrate = actStats['meanrate']
        stdrate = actStats['meancellstds']
        
        h_sleep = np.squeeze(h.detach().numpy())
        
        SWdist = rg.calculateSleepWakeDist(h_wake, h_sleep, metric='cosine')
        SWdist = SWdist[0]
        
        if decoder:
            onset = self.onset
            decoded, p = self.pN.decode(h,decoder)
            decoded_slice = decoded.__getitem__(range(onset,decoded.shape[0]),
                                                columns=decoded.columns)
            c_cov = trajectoryAnalysis.calculateCoverage(decoded_slice,
                                                            [0,decoder.gridheight,
                                                             0,decoder.gridwidth],
                                                    mask=decoder.mask)
            coverage = 1-c_cov['nonuniformity']
            c_cont = trajectoryAnalysis.calculateContinuity(decoded_slice,
                                                            showFig=False,numdelays=25,
                                                            maxdist=20)
            continuity = c_cont['underthresh']
            
            transitionmap = self.calculateTransitionMap(decoded_slice, transrange = 3)
            
            SWsimilarity = np.nan
            if wake_continuity is not None:
                SWsimilarity = compareSW(c_cont,wake_continuity)
            
            continuity_DIFFUS = makeDiffusion(dtmax = 26, dxmax = 20, D=0.6)
            SDsimilarity = compareSW(c_cont,continuity_DIFFUS)
            
            diffFit = calculateDiffusionFit(decoded,dtmax = 20)
            diffusioncoeff = diffFit['alpha']
            diffusionRsq = diffFit['r_sq']
            diffusionG = diffFit['intercept']
            
                        
        return (meanrate, stdrate, SWdist, coverage, continuity,
                h_sleep, actStats, c_cov, c_cont, transitionmap, 
                SWsimilarity, SDsimilarity, diffusioncoeff, diffusionRsq, diffusionG)
        
    def runSpontaneousSims(self,noisemags,noisestds,decoder,timesteps, h_wake, wake_continuity=None):
        meanrate, stdrate, simcounts, coverage, continuity, SWdist, SWsim, SDsim, diffCoef, diffRsq, diffG = self.initalizeOutputs(noisemags,noisestds)
        
        for i,noisemag in enumerate(noisemags):
            print(i)
            for j,noisestd in enumerate(noisestds):
                while meanrate[i,j] > 5 and simcounts[i,j]<5:
                    (meanrate[i,j], stdrate[i,j], SWdist[i,j],
                    coverage[i,j],  continuity[i,j],
                    _,_,_,_,_,SWsim[i,j], SDsim[i,j],
                    diffCoef[i,j], diffRsq[i,j], diffG[i,j]) = self.runSim_getResults(timesteps,noisemag,
                                                      noisestd,decoder,
                                                      h_wake, wake_continuity=wake_continuity)
                            
                    if np.isnan(meanrate[i,j]): 
                        meanrate[i,j] = np.inf
                    simcounts[i,j]+=1
                    
        return meanrate, stdrate, simcounts, coverage, continuity, SWdist, SWsim, SDsim, diffCoef, diffRsq, diffG
    
    def initalizeOutputs(self,noisemags,noisestds):
        meanrate = np.ones((len(noisemags),len(noisestds)))*np.inf
        stdrate = np.zeros((len(noisemags),len(noisestds)))
        simcounts = np.zeros((len(noisemags),len(noisestds)))
        coverage = np.zeros((len(noisemags),len(noisestds)))
        continuity = np.zeros((len(noisemags),len(noisestds)))
        SWdist = np.ones((len(noisemags),len(noisestds)))*np.inf
        SWsim = np.ones((len(noisemags),len(noisestds)))*np.inf
        SDsim = np.ones((len(noisemags),len(noisestds)))*np.inf
        diffCoef = np.zeros((len(noisemags),len(noisestds)))
        diffR = np.zeros((len(noisemags),len(noisestds)))
        diffG = np.zeros((len(noisemags),len(noisestds)))
        return meanrate, stdrate, simcounts, coverage, continuity, SWdist, SWsim, SDsim, diffCoef, diffR, diffG
        
    def runWakeComparison(self,timesteps,decoder):
        env = self.pN.EnvLibrary[0]
        action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
        agent = RandomActionAgent(env.action_space,action_probability)
        
        obs, act, state, _ = self.pN.collectObservationSequence(env,
                                                            agent,
                                                            timesteps,
                                                            includeRender = False)
        obs_pred, obs_next, h_wake = self.pN.predict(obs, act)
        actStats_wake = self.pN.calculateActivationStats(h_wake)
        meanWAKE = actStats_wake['meanrate']
        stdWAKE = actStats_wake['meancellstds']
        h_wake = np.squeeze(h_wake.detach().numpy())
            
        state_nap = state2nap(state)
        c_cov = trajectoryAnalysis.calculateCoverage(state_nap,
                                                        [0,decoder.gridheight,
                                                         0,decoder.gridwidth],
                                                    mask=decoder.mask)
        coverage_wake = 1-c_cov['nonuniformity']
        c_cont = trajectoryAnalysis.calculateContinuity(state_nap,showFig=False,numdelays=25,
                                                        maxdist=20)
        continuity_wake = c_cont['underthresh']
        
        return (meanWAKE, stdWAKE, h_wake, coverage_wake, continuity_wake, 
                c_cov, c_cont)
        
        
    def runSamples(self,examples,timesteps, decoder, h_wake, wake_continuity=None):
        exampleStuff = []
        for s,sample in enumerate(examples):
            a = {}
            (a['meanrate'],
            a['stdrate'],
            a['SleepSimilarity'],
            a['coverage'],
            a['continuity'],
            a['h_sleep'],
            a['actStats'],
            a['c_cov'],
            a['c_cont'],
            a['transitionmap'],
            a['SWsim'],
            a['SDsim'],
            a['diffCoef'],
            a['diffR'],
            a['diffG']) = self.runSim_getResults(timesteps,
                                                sample[0],
                                                sample[1],
                                                decoder, h_wake, 
                                                 wake_continuity=wake_continuity)
            a['noisemean']=sample[0]
            a['noisestd']=sample[1]
            exampleStuff.append(a)
        return exampleStuff
        

        
        
    def OfflineActivityFigure(self, netname, savefolder):
        meanrate, stdrate, simcounts, coverage, continuity, SleepSimilarity, SWsim, SDsim, diffCoef, diffRsq, diffG = self.spontStats
        meanWAKE,stdWAKE,_,_,_,_,_ = self.wakeStats
        
        plt.figure()
        
        #plt.subplot(3,3,1)
        #self.musigPanel(meanrate,meanWAKE,logScale=True)
        #plt.title('Mean Rate')
        
        #plt.subplot(3,3,2)
        #self.musigPanel(stdrate,stdWAKE,logScale=True)
        #plt.title('Std Rate')
        
        plt.subplot(3,3,1)
        self.musigPanel(np.log10(SleepSimilarity),None,clim=[-1.25,-0.25])
        plt.title('Sleep-Wake Distance')
        
        plt.subplot(3,3,2)
        self.musigPanel(coverage,None,clim=[0,1])
        plt.title('Coverage')
        
        plt.subplot(3,3,3)
        self.musigPanel(continuity,None)
        plt.title('Continuity')
        
        plt.subplot(3,3,7)
        self.musigPanel(diffCoef,None, clim=[0,1],cmap='twilight')
        plt.title('Diff. Coef')
        
        plt.subplot(3,3,8)
        self.musigPanel(diffG,None)
        plt.title('Diff. G')
        
        plt.subplot(3,3,9)
        self.musigPanel(diffRsq,None, clim=[0,1])
        plt.title('Diff. R')
        
        plt.subplot(3,3,4)
        self.musigPanel(SWsim,None,logScale=True,clim=[-1,1],cmap='bwr')
        plt.title('S-W Similarity')
        
        plt.subplot(3,3,5)
        self.musigPanel(SDsim,None,logScale=True,clim=[-1,1],cmap='bwr')
        plt.title('S-D Similarity')
        
        plt.tight_layout()
        if netname is not None:
            saveFig(plt.gcf(),netname+'_OfflineActivity',savefolder,
                    filetype='pdf')

        plt.show()
    
    
    def OfflineActivityStatsFigure(self,netname,savefolder):
        meanrate, stdrate, simcounts, coverage, continuity, SleepSimilarity, SWsim, SDsim, diffCoef = self.spontStats
        meanWAKE, stdWAKE, h_wake, coverage_wake, continuity_wake, c_cov, c_cont = self.wakeStats
        
        plt.figure(figsize=(10,6))
        
        plt.subplot(4,2,7)
        plt.scatter(coverage,SDsim,c=SleepSimilarity,
                    vmin=0, vmax=1)
        for sample in self.exampleSims:
            plt.plot(sample['coverage'],sample['SDsim'],'k+')
        #plt.plot(coverage_wake,continuity_wake,'r+')
        plt.xlabel('Coverage')
        plt.ylabel('Continuity')
        plt.colorbar(label='SW Dist (cos)')
        
        plt.subplot(4,len(self.exampleSims)+1,1+2*(len(self.exampleSims)+1))
        self.continuityPanel(c_cont,SWsim=0)
        
        plt.subplot(4,len(self.exampleSims)+1,1)
        self.coveragePanel(c_cov)

        for s,sample in enumerate(self.exampleSims):
            plt.subplot(4,len(self.exampleSims)+1,2+s+2*(len(self.exampleSims)+1))
            self.continuityPanel(sample['c_cont'],SWsim=sample['SDsim'])
            
            plt.subplot(4,len(self.exampleSims)+1,2+s+1*(len(self.exampleSims)+1))
            self.transitionProbabilityPanel(sample['transitionmap'])
            
            plt.subplot(4,len(self.exampleSims)+1,2+s)
            self.coveragePanel(sample['c_cov'])
            plt.xlabel([])
            plt.ylabel([])
            
        plt.tight_layout()
        if netname is not None:
            saveFig(plt.gcf(),netname+'_OfflineStats',savefolder,
                    filetype='pdf')

        plt.show()

        
    def continuityPanel(self,continuity,SWsim = None):
        numdelays = np.size(continuity['delay_dist'],1)
        maxdist = np.size(continuity['delay_dist'],0)
        
        ax1 = plt.gca()
        
        ax1.imshow(continuity['delay_dist'],origin='lower',
                   extent=(0.5,numdelays+0.5,-0.5,maxdist-0.5),
                   aspect='auto', cmap='binary')
        if SWsim is None:
            ax2 = ax1.twinx()
            ax2.plot(continuity['delays'],continuity['delay_kl'],'.--r')
            ax1.text(numdelays-2,maxdist-2, f"{continuity['underthresh']}",
                     fontsize=10,color='r')
            ax2.set_ylabel('K-L Div from P[dx]')
            
        else:
            ax1.text(numdelays-2,maxdist-3, f"{np.log10(SWsim):0.1}",
                     fontsize=10,color='r')
        ax1.set_xlabel('dt')
        ax1.set_ylabel('dx')
        
        #ax2.set_xlim([0,])
        
    def calculateTransitionMap(self,decoded, transrange = 3):
        decodedpos = decoded.values
        transitionprob,edges = OfflineTrajectoryAnalysis.calculateTransitionProbability(self,
                                                                                        decodedpos,
                                                                   transrange=transrange)

        return transitionprob,edges
        
    def transitionProbabilityPanel(self, transitionprob):
        OfflineTrajectoryAnalysis.transitionProbabilityPanel(self,transitionprob,marker='+',
                                                             vmax=None, incLabels = True)
        
    def coveragePanel(self,coverage):
        OfflineTrajectoryAnalysis.occupancyDistPanel(self,coverage)
        plt.title(None)
    
    def musigPanel(self,metric,WAKEcompare,logScale=False, clim=None,cmap=None):
        murange = self.noisemags[-1]
        noisestds = self.noisestds
        if logScale:
            metric = np.log10(metric)
        im = plt.imshow(metric.T,
                        origin='lower', aspect='auto',
                        extent = (-murange,murange,
                                  np.log10(noisestds[0]),np.log10(noisestds[-1])),)
        if WAKEcompare is not None:
            im.set_clim(vmin=np.log10(WAKEcompare)-2.5,
                         vmax=np.log10(WAKEcompare)+2.5)
            im.set_cmap('twilight')
        elif clim is not None:
            im.set_clim(vmin=clim[0],
                         vmax=clim[1])
        if cmap is not None:
            im.set_cmap(cmap)
        plt.plot([0,0],np.log10(noisestds[[0,-1]]),'w--')
        plt.colorbar()
        plt.xlabel('Mean')
        plt.ylabel('log std')
        
        
        
        
     
        

        
      
    