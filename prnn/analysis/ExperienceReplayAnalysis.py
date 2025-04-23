import pynapple as nap
import prnn.analysis.trajectoryAnalysis as trajectoryAnalysis
import numpy as np
import copy
from prnn.analysis.OfflineTrajectoryAnalysis import OfflineTrajectoryAnalysis as OTA
from scipy.stats import spearmanr
from scipy.spatial import distance
from prnn.utils.agent import RandomActionAgent
import torch
from prnn.utils.general import saveFig
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
from prnn.analysis.OfflineTrajectoryAnalysis import makeAdaptingNet


class ExperienceReplayAnalysis:
    def __init__(self, predictiveNet, seqdur=100, timesteps_sleep=200,  decoder='train',
                 lrmin = 1, lrmax = 16, numlrs = 5, 
                 numExps = 1, numWakeRepeats=20, numSleepRepeats = 30, 
                 sleepNoiseStd = 0.1, sleepAgent = None,
                 lowExampleThresh = 0.05, hiExampleThresh = 0.3, exampleLRthresh = 1,
                 resetOptimizer=False, lrgroups=[0,1,2],
                 withAdapt=False, b_adapt = 0.3, tau_adapt=8):
        
        self.pN = predictiveNet
        self.lowExampleThresh = lowExampleThresh
        self.hiExampleThresh = hiExampleThresh
        self.exampleLRthresh = exampleLRthresh
        self.HiExample = None
        self.LowExample = None
        self.trialExample = None
        
        #Train the decoder 
        if decoder == 'train':
            self.decoder = self.trainDecoder()
        else:
            self.decoder = decoder
        
        #Set the trial learning rates
        self.lrs = np.concatenate((np.array([0]),np.logspace(np.log10(lrmin),np.log10(lrmax),numlrs)))
        
        self.exps = numExps
        if isinstance(numExps,int):
            print(f'Collecting Experiences')
            self.exps = []
            for expIDX in range(numExps):
                self.exps.append(self.collectExperience(self.pN, seqdur))
        numExps = len(self.exps)
        
        self.LRpanel = []
        self.pTimeWake = []
        self.pVisited = []
        self.SWcorr = []
        self.meanSWcorr = np.zeros((numlrs+1,numExps))
        self.meanpWake = np.zeros((numlrs+1,numExps))
        self.meanpVisited = np.zeros((numlrs+1,numExps))
        for expIDX,exp in enumerate(self.exps):
            print(f'Trial {expIDX} of {numExps}')
            LRpanel, pTimeWake, pVisited, SWcorr = self.runLRpanel(self.pN, self.lrs, exp,
                                                    numWakeRepeats, numSleepRepeats,
                                                    noisestd = sleepNoiseStd, 
                                                    actionAgent = sleepAgent,
                                                    timesteps_sleep = timesteps_sleep,
                                                    resetOptimizer = resetOptimizer, 
                                                    lrgroups=lrgroups,
                                                    withAdapt=withAdapt, 
                                                    b_adapt = b_adapt, tau_adapt=tau_adapt,
                                                    seed=expIDX)
            self.LRpanel.append(LRpanel)
            self.pTimeWake.append(pTimeWake)
            self.pVisited.append(pVisited)
            self.SWcorr.append(SWcorr)
            self.meanSWcorr[:,expIDX] = np.mean(LRpanel,axis=0)
            self.meanpWake[:,expIDX] = np.mean(pTimeWake,axis=0)
            self.meanpVisited[:,expIDX] = np.mean(pVisited,axis=0)
            
            #If the experience has a high and low example, save it!
            if self.HiExample is None and expIDX<numExps-1:
                self.LowExample = None
            elif self.trialExample is None:
                traj, coverage, _, _, lastRender = exp
                self.trialExample  = {'traj'     : traj,
                                      'coverage' : coverage,
                                      'trialNum' : expIDX,
                                      'lastRender':lastRender}
                
        self.LRpanel = np.concatenate(self.LRpanel)
        self.SWcorr = np.concatenate(self.SWcorr)
                                        

    def getExamples(self, seqdur, sleepNoiseStd=0.1, hiExampleThresh=0.5, maxSleepRepeats=10000):
        self.HiExample = None
        self.LowExample = None
        self.trialExample = None
        exp = self.collectExperience(self.pN, seqdur)
    
    
    def testNoiseLevel(self, seqdur=100, 
                       lognoisemin = -3, lognoisemax = -0.5, numnoise = 6, 
                       numExps = 1, numWakeRepeats=20, numSleepRepeats = 30, 
                       sleepAgent = None,
                       resetOptimizer = False, lrgroups=[0,1,2],
                       withAdapt=False, b_adapt = 0.4, tau_adapt=5):
        
        #1) Return experience occupany, to plot
        #2) Take out the non-accessable locations!!! (use decoder.mask)
        #2a) consider just using overlap of non-0 entries - fraction of 
        #.   visited locations in the Trial present in the replay...
        #.   what about unvisited locations?...
        #3) Also check noise during experience
        noises = np.logspace(lognoisemin,lognoisemax,numnoise)
        noisePanel = np.zeros((len(self.lrs),len(noises),numExps))
        exps = []
        for expIDX in range(numExps):

            exp = self.collectExperience(self.pN, seqdur)
            exps.append(exp)
            for nIDX, sleepNoiseStd in enumerate(noises):
                SWsims, _, _ = self.runLRpanel(self.pN, self.lrs, exp,
                                         numWakeRepeats, numSleepRepeats,
                                         noisestd = sleepNoiseStd,
                                         actionAgent = sleepAgent,
                                         resetOptimizer = resetOptimizer, 
                                         lrgroups=lrgroups,
                                         withAdapt=withAdapt,
                                         b_adapt=b_adapt, tau_adapt=tau_adapt)
                noisePanel[:,nIDX,expIDX] = np.mean(SWsims,axis=0)
        
        return noisePanel, noises, self.lrs, exps


        
        
    def testRepeats(self, lr=1e-3, seqdur=100, sleepNoiseStd = 0.03,
                    sleepAgent=None, 
                    wakereps = np.arange(1,6,1), 
                    totalreps = np.arange(1,60,7),
                    resetOptimizer = False, lrgroups=[0,1,2],
                    withAdapt=False, b_adapt = 0.4, tau_adapt=5):
        exp = self.collectExperience(self.pN, seqdur)
        
        #wakereps = np.arange(1,6,1)
        #sleepreps = np.arange(1,60,7)
        repPanel = np.zeros((len(wakereps),len(totalreps)))
        toc = np.zeros((len(wakereps),len(totalreps)))
        for wIDX, numWakeRepeats in enumerate(wakereps):
            for sIDX, numTotalRepeats in enumerate(totalreps):
                tic = time.time()
                numSleepRepeats = int(numTotalRepeats/numWakeRepeats)
                SWsims,_,_,_ = self.runLRpanel(self.pN, np.array([lr]), exp,
                                          numWakeRepeats, numSleepRepeats,
                                          noisestd = sleepNoiseStd,
                                          actionAgent = sleepAgent,
                                          resetOptimizer = resetOptimizer, 
                                          lrgroups=lrgroups,
                                          withAdapt=withAdapt,
                                          b_adapt=b_adapt, tau_adapt=tau_adapt)
                repPanel[wIDX,sIDX] = np.mean(SWsims)
                toc[wIDX,sIDX] = time.time()-tic
                
        return repPanel,toc, wakereps, totalreps
        
        
    def calculateOverlap(self,occupancyTrial,occupancyReplay):
        #Option 1: remove mask, use correlation (works well)
        #mask = np.pad(self.decoder.mask,(0,1))
        mask = self.decoder.mask
        occupancyTrial = occupancyTrial[mask!=0]
        occupancyReplay = occupancyReplay[mask!=0]
        
        #Option 2: Remove all neither-locations, use correlation (has negative values, not great)
        #eitherTraj = (occupancyTrial>0) | (occupancyReplay>0)
        #occupancyTrial = occupancyTrial[eitherTraj]
        #occupancyReplay = occupancyReplay[eitherTraj]
        
        occ_corr = spearmanr(occupancyTrial,occupancyReplay)
        occ_corr = occ_corr.correlation
        
        #Option 3: Remove all neither-locations, use hamming distance
        #Problem with hamming distance: (False,False) counts as good
        #Want distance where 1,1 is good, 0,1 or 1,0 is bad, 0,0 is neutral
        #occupancyTrial[occupancyTrial>0]=1
        #occupancyReplay[occupancyReplay>0]=1
        #occ_corr = 1-distance.hamming(occupancyTrial,occupancyReplay)
        #Try removing all (False False) entries...!
        return occ_corr
    
    def replayScore(self,occupancyTrial,occupancyReplay):
        #Option 1: remove mask, use correlation (works well)
        #mask = np.pad(self.decoder.mask,(0,1))
        mask = self.decoder.mask
        occupancyTrial = occupancyTrial[mask!=0]
        occupancyReplay = occupancyReplay[mask!=0]
        
        pTimeInWakeLoc = np.sum(occupancyReplay * (occupancyTrial>0))
        pWakeVisited = np.sum((occupancyReplay>0) * (occupancyTrial>0))/np.sum(occupancyTrial>0)
        replayScore = pWakeVisited*pTimeInWakeLoc
        return replayScore, pTimeInWakeLoc, pWakeVisited
        
        
    def replayTrial(self, pN, decoder, trialOccupancy,
                    noisemag = 0, noisestd = 0.05,
                    actionAgent = None, timesteps_sleep = 100, lr=0):
        
        onsetTransient = 20
        timesteps_sleep = timesteps_sleep+onsetTransient
        
        OTA_trial = OTA(pN, noisemag = noisemag, noisestd=noisestd,
                        decoder=decoder, actionAgent=actionAgent,
                        timesteps_sleep=timesteps_sleep, suppressPrint=True,
                        withTransitionMaps = False, sleepOnsetTransient=onsetTransient,
                        skipContinuity = True,
                        calculateDiffusionFit=False)

        occupancyMap = OTA_trial.coverageContinuity[2]['occupancy']

        occ_corr = self.calculateOverlap(trialOccupancy,occupancyMap)
        replayScore, pTimeInWakeLoc, pWakeVisited = self.replayScore(trialOccupancy,occupancyMap)
        
        if (lr > self.exampleLRthresh and occ_corr > self.hiExampleThresh and 
            self.HiExample is None):
            
            self.HiExample = {'traj' : OTA_trial.SLEEPdecoded[0],
                              'coverage' : OTA_trial.coverageContinuity[2],
                              'occupancyCorr' : occ_corr,
                              'replayScore' : replayScore,
                              'pTimeInWakeLoc' : pTimeInWakeLoc,
                              'pWakeVisited' : pWakeVisited}      
        
        if (lr > self.exampleLRthresh and occ_corr < self.lowExampleThresh and 
            self.LowExample is None):
            
            self.LowExample  = {'traj' : OTA_trial.SLEEPdecoded[0],
                                'coverage' : OTA_trial.coverageContinuity[2],
                                'occupancyCorr' : occ_corr,
                                'replayScore' : replayScore,
                                'pTimeInWakeLoc' : pTimeInWakeLoc,
                                'pWakeVisited' : pWakeVisited}
        
        return occ_corr, replayScore, pTimeInWakeLoc, pWakeVisited

    
    def runReplayTrials(self,pN, exp, lr, numWakeRepeats, numSleepRepeats,
                        noisestd = 0.1, actionAgent = None, timesteps_sleep = 100,
                        resetOptimizer=False, lrgroups=[0,1,2],
                        withAdapt=False, b_adapt = 0.4, tau_adapt=5):
        occ_overlap = np.zeros((numWakeRepeats,numSleepRepeats))
        replayScore = np.zeros((numWakeRepeats,numSleepRepeats))
        pTimeInWakeLoc = np.zeros((numWakeRepeats,numSleepRepeats))
        pWakeVisited = np.zeros((numWakeRepeats,numSleepRepeats))
        traj, coverage, obs, act, _ = exp
        trialOccupancy = coverage['occupancy']
        
        for wakeIDX in range(numWakeRepeats):
            #Copy the network
            pN_post = pN.copy()
            
            if resetOptimizer:
                #print('resetting optimizer')
                pN_post.resetOptimizer(pN_post.trainArgs.lr,
                                       pN_post.trainArgs.weight_decay)
                
            #Update the learning rate
            for lidx,lgroup in enumerate(lrgroups):
                pN_post.optimizer.param_groups[lgroup]['lr'] *= lr
            
            #Annoying device issue...
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "cpu"
            pN_post.pRNN.to(device)
            obs,act = obs.to(device),act.to(device)
            
            steploss,_,_ = pN_post.trainStep(obs,act)
            pN_post.pRNN.to('cpu')
            
            if withAdapt:
                pN_post = makeAdaptingNet(pN_post, b_adapt, tau_adapt)
            
            for sleepIDX in range(numSleepRepeats):
                (occ_overlap[wakeIDX,sleepIDX], 
                 replayScore[wakeIDX,sleepIDX], 
                 pTimeInWakeLoc[wakeIDX,sleepIDX], 
                 pWakeVisited[wakeIDX,sleepIDX]) = self.replayTrial(pN_post,
                                                  self.decoder,
                                                  trialOccupancy,
                                                  lr = lr,
                                                  noisestd = noisestd,
                                                  actionAgent = actionAgent,
                                                  timesteps_sleep = timesteps_sleep)
                
        return occ_overlap, replayScore, pTimeInWakeLoc, pWakeVisited
            
            
            
    def runLRpanel(self, pN, lrs, exp, numWakeRepeats, numSleepRepeats,
                   noisestd = 0.1, actionAgent = None, timesteps_sleep = 100,
                   resetOptimizer = False, lrgroups=[0,1,2],
                   withAdapt=False, b_adapt = 0.4, tau_adapt=5, seed = 'rand'):
        if seed == 'rand':
            seed = np.random.randint(1000)
        LRpanel = np.zeros((numWakeRepeats*numSleepRepeats,len(lrs)))
        pTimeWake = np.zeros((numWakeRepeats*numSleepRepeats,len(lrs)))
        pVisited = np.zeros((numWakeRepeats*numSleepRepeats,len(lrs)))
        SWcorr = np.zeros((numWakeRepeats*numSleepRepeats,len(lrs)))
        for lrIDX,lr in enumerate(lrs):
            #print(f"LR {lrIDX} of {len(lrs)}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            (occ_overlap, 
             replayScore, 
             pTimeInWakeLoc, 
             pWakeVisited) = self.runReplayTrials(pN, exp, lr,
                                               numWakeRepeats, numSleepRepeats,
                                               noisestd = noisestd, 
                                               actionAgent = actionAgent,
                                               timesteps_sleep = timesteps_sleep,
                                               resetOptimizer=resetOptimizer, 
                                               lrgroups=lrgroups,
                                               withAdapt=withAdapt, 
                                               b_adapt = b_adapt, tau_adapt=tau_adapt) 
            LRpanel[:,lrIDX] = replayScore.flatten()
            pTimeWake[:,lrIDX] = pTimeInWakeLoc.flatten()
            pVisited[:,lrIDX] = pWakeVisited.flatten()
            SWcorr[:,lrIDX] = occ_overlap.flatten()
            
        return LRpanel, pTimeWake, pVisited, SWcorr
        
    def collectExperience(self, pN, seqdur):
        #Run the "experienced" training step
        env = pN.EnvLibrary[0]
        action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
        agent = RandomActionAgent(env.action_space,action_probability)
        obs,act,state,render = pN.collectObservationSequence(env,
                                                        agent,
                                                        seqdur,
                                                        includeRender=True)
        lastRender = render[-1]
        
        traj = state['agent_pos']
        timesteps = np.size(traj,0)
        traj = nap.TsdFrame(
            t = np.arange(timesteps),
            d = traj,
            time_units = 's',
            columns=('x','y')
            )
        
        coverage = trajectoryAnalysis.calculateCoverage(traj,
                                                        [0,env.width,
                                                         0,env.height])
        
        return traj, coverage, obs, act, lastRender
    

    def trainDecoder(self):
        env = self.pN.EnvLibrary[0]
        action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
        agent = RandomActionAgent(env.action_space,action_probability)
        _, _, decoder = self.pN.calculateSpatialRepresentation(env,
                                                               agent,
                                                               numBatches=10000,
                                                               trainDecoder=True)
        return decoder
      
        
        
        
        
    def LRPanelFigure(self, netname = None, savefolder = None):
        
        plt.figure()
        plt.subplot(3,3,1)
        self.plotTrajectoryPanel(self.trialExample['traj'],self.trialExample['lastRender'],
                                showRender=True,showOccupancy=False)
        plt.title(f"Trial ({self.trialExample['trialNum']})")
        
        if self.LowExample is not None:
            plt.subplot(3,3,3)
            self.plotTrajectoryPanel(self.LowExample['traj'],self.trialExample['coverage'],
                                    timeMarker=True, showOccupancy=True)
            plt.title(f"Replay: {self.LowExample['occupancyCorr']:.2f}")
        if self.HiExample is not None:
            plt.subplot(3,3,2)
            self.plotTrajectoryPanel(self.HiExample['traj'],self.trialExample['coverage'],
                                    timeMarker=True, showOccupancy=True)
            plt.title(f"Replay: {self.HiExample['occupancyCorr']:.2f}")
        
        plt.subplot(3,2,5)
        self.plotLRPanel(self.SWcorr, self.lrs)
        plt.subplot(3,2,6)
        #self.plotLRPanel(self.LRpanel, self.lrs, meanLR = self.meanSWcorr, metric = 'Replay Score')
        self.plotLRPanel(self.LRpanel, self.lrs,  metric = 'Replay Score')
        
        plt.subplot(3,2,4)
        self.plotReplayScorePanel(self.meanpWake,self.meanpVisited,self.lrs)

        #plt.tight_layout()
        if netname is not None:
            saveFig(plt.gcf(),netname+'_LRReplayPanel',savefolder,
                    filetype='png')
            
        plt.show()
    
    def plotLRPanel(self, occ_overlap, lrs, meanLR=None, metric='Trial-Sleep Corr.',showxlabel=True):
        
        if meanLR is not None:
           plt.plot(np.arange(len(lrs))+1,meanLR,'--',color='grey',linewidth=0.5)
            #plt.plot(np.arange(len(lrs))+1,meanLR[:,0],'-',color='black',linewidth=0.5)
        plt.boxplot(occ_overlap,labels = lrs.astype(int),showfliers=False,whis=[5,95])

        plt.plot(plt.xlim(),[0,0],'k--')
        if showxlabel:
            plt.xlabel('LR During Trial (*Train)')
        plt.ylabel(metric)
        
    def plotLRPanel_violin(self, occ_overlap, lrs, meanLR=None, metric='Trial-Sleep Corr.',showxlabel=True):
        
        plt.violinplot(occ_overlap, showextrema=False)

        plt.plot(plt.xlim(),[0,0],'k--')
        if showxlabel:
            plt.xlabel('LR During Trial (*Train)')
        plt.ylabel(metric)
        
    def plotReplayScorePanel(self, meanpWake, meanpVisited, lrs):
            plt.plot(np.arange(len(lrs))+1,np.mean(meanpWake,axis=1),label='P[Time] in Wake Locs')
            plt.plot(np.arange(len(lrs))+1,np.mean(meanpVisited,axis=1),label='P[Wake Locs] visited')
            plt.legend()
        
        
        
    def plotTrajectoryPanel(self, traj, coverage, trajRange=None,
                            onsetTransient = 0, timeMarker=False, showOccupancy=True,
                            showRender=False):
        x = traj['x'].values
        y= traj['y'].values
        if trajRange is None:
            trajRange = (onsetTransient, np.size(x,0)-1)
            
        palette = copy.copy(plt.get_cmap('viridis'))
        palette.set_under('grey', 1.0)  # 1.0 represents not transparent
        if showOccupancy:
            plt.imshow(coverage['occupancy'].transpose(),
                       interpolation='nearest',alpha=self.decoder.mask.transpose(),
                      cmap=palette, vmin=0.001)
        elif showRender:
            plt.imshow(np.flip(coverage,axis=0),
              extent = (-0.5,17.5,-0.5,17.5))
        else:
            plt.imshow(np.ones_like(self.decoder.mask.transpose()),
                       interpolation='nearest',alpha=self.decoder.mask.transpose())
        plt.plot(x[trajRange[0]:trajRange[1]],
                 y[trajRange[0]:trajRange[1]],'r',
                 linewidth=1)
        if timeMarker:
            plt.scatter(x[trajRange[0]:trajRange[1]],
                        y[trajRange[0]:trajRange[1]],
                        c=range(trajRange[1]-trajRange[0]), marker='.',
                        cmap='cubehelix')
        else:
            plt.plot(x[trajRange[-1]], y[trajRange[-1]], 'r', marker='^')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.gca().invert_yaxis()
        
    #From OfflineTrajectoryAnalysis    
    def trajectoryPanel2(self,decoded,trajRange=(0,20)):
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