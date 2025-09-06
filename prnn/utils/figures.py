#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:15:54 2022

@author: dl2820
"""
import numpy as np
import matplotlib.pyplot as plt
from prnn.utils.general import saveFig
import prnn.analysis.trajectoryAnalysis as trajectoryAnalysis


def TrainingFigure(predictiveNet,
                   savename=None,savefolder=None,incDecode = True ):
    
    trainingfig = plt.figure(figsize=(15,10))
    (LCfig,PFfigs) = trainingfig.subfigures(2, 1, height_ratios=[0.66,0.33])
    PFfigs = PFfigs.subfigures(1,4,wspace=0.1)
    (LCfig,finalPFfig)=LCfig.subfigures(1,2, width_ratios=[0.66, 0.33])
    (LCax,noax)=LCfig.subplots(1,2,gridspec_kw={'width_ratios': [0.99, 0.01]})
    noax.axis("off")
    (SIhistfig,finalPFfig)=finalPFfig.subfigures(2,1, height_ratios=[0.25, 0.75])

    
    #TODO: Subset of spatial info for boxwhisker
    predictiveNet.plotLearningCurve(axis=LCax,maxBoxes=11,incDecode = incDecode)
    
    #Find which training steps have place fields
    numPFpanels = 4
    trials = ~predictiveNet.TrainingSaver['SI'].isna()
    index = predictiveNet.TrainingSaver['SI'].index[trials]  
    PFtrainsteps = np.round(np.linspace(1,len(index)-1,numPFpanels))
    #PFtrainsteps=np.delete(PFtrainsteps,0)

    #Final PF panel
    trainstep = index[PFtrainsteps[numPFpanels-1]]
    #PFfigs[PFpanel].suptitle(f'Step: {trainstep}')
    place_fields = predictiveNet.TrainingSaver.place_fields[trainstep]
    SI = predictiveNet.TrainingSaver.SI[trainstep]
    predictiveNet.plotTuningCurvePanel(fig=finalPFfig,
                                       place_fields=place_fields,SI=SI)
    
    #SI Histogram    
    SIhistax=SIhistfig.subplots(1,1)
    SIhistax.hist(SI,color='red')
    plt.xlabel('SI')
    plt.ylabel('# Units')
    
    for PFpanel in range(3):
        trainstep = index[PFtrainsteps[PFpanel]]
        PFfigs[PFpanel].suptitle(f'Step: {trainstep}')
        place_fields = predictiveNet.TrainingSaver.place_fields[trainstep]
        SI = predictiveNet.TrainingSaver.SI[trainstep]
        predictiveNet.plotTuningCurvePanel(fig=PFfigs[PFpanel],
                                           place_fields=place_fields,SI=None)
        
    
    if savename is not None: 
        saveFig(plt.gcf(),savename+'_TrainingFigure',savefolder,
                filetype='png')
    # plt.show()
    
    return


def SpontActivityFigure(predictiveNet, compareWAKEagent=None,
                        murange=0.5, maxstd=1, numpoints=11, timesteps=1000,
                        examples=((0,0.1),(0,0.25),(0,0.4)),
                        savename=None, savefolder=None, decoder=None):

    noisemags = np.linspace(-murange,murange,numpoints)
    noisestds = np.logspace(-2,np.log10(maxstd),numpoints)
    
    #Simulate all the spontaneous activities
    meanrate = np.ones((len(noisemags),len(noisestds)))*np.inf
    stdrate = np.zeros((len(noisemags),len(noisestds)))
    simcounts = np.zeros((len(noisemags),len(noisestds)))
    
    coverage = np.zeros((len(noisemags),len(noisestds)))
    continuity = np.zeros((len(noisemags),len(noisestds)))
    idx0 = int(numpoints/2)
    for i,noisemag in enumerate(noisemags):
        print(i)
        for j,noisestd in enumerate(noisestds):
            while meanrate[i,j] > 5 and simcounts[i,j]<5:
                obs_pred,h,noise_t = predictiveNet.spontaneous(timesteps,noisemag,noisestd)
                actStats = predictiveNet.calculateActivationStats(h)
                
                meanrate[i,j] = actStats['meanrate']
                stdrate[i,j] = actStats['meancellstds']
                
                if decoder:
                    onset = 100
                    decoded, p = predictiveNet.decode(h,decoder)
                    c = trajectoryAnalysis.calculateCoverage(decoded[onset:],
                                                                    [1,decoder.gridheight,
                                                                     1,decoder.gridwidth])
                    coverage[i,j] = c['nonuniformity']
                    c = trajectoryAnalysis.calculateContinuity(decoded[onset:],showFig=False)
                    continuity[i,j] = c['underthresh']
                if np.isnan(meanrate[i,j]): 
                    meanrate[i,j] = np.inf
                simcounts[i,j]+=1
                
    
    #Simulate the WAKE trajectory for comparison
    if compareWAKEagent:    
        env = predictiveNet.EnvLibrary[0]
        agent = compareWAKEagent
        obs,act,state, render  = predictiveNet.collectObservationSequence(env,agent,timesteps,
                                                                   includeRender = False)
        obs_pred, obs_next ,h_wake = predictiveNet.predict(obs,act)
        actStats_wake = predictiveNet.calculateActivationStats(h_wake)
        meanWAKE = actStats_wake['meanrate']
        stdWAKE = actStats_wake['meancellstds']
    
    h, actStats = {},{}    
    for s,sample in enumerate(examples):
            obs_pred,h[s],noise_t = predictiveNet.spontaneous(timesteps,sample[0],sample[1])
            actStats[s] = predictiveNet.calculateActivationStats(h[s])
        
    #The Figure
    plt.figure(figsize=(15,12))
    plt.subplot(3,3,1)
    plt.imshow(np.log10(meanrate.T),origin='lower',aspect='auto',
               extent=(-murange,murange,np.log10(noisestds[0]),np.log10(noisestds[-1])),
               cmap='twilight',vmin=np.log10(meanWAKE)-2.5, vmax=np.log10(meanWAKE)+2.5)
    plt.plot([0,0],np.log10(noisestds[[0,-1]]),'w--')
    plt.colorbar()
    plt.title('Mean Rate')
    plt.xlabel('Mean')
    plt.ylabel('log std')
    
    plt.subplot(3,3,2)
    plt.imshow(np.log10(stdrate.T),origin='lower',aspect='auto',
               extent=(-murange,murange,np.log10(noisestds[0]),np.log10(noisestds[-1])),
               cmap='twilight',vmin=np.log10(stdWAKE)-1.5, vmax=np.log10(stdWAKE)+1.5)
    plt.plot([0,0],np.log10(noisestds[[0,-1]]),'w--')
    plt.colorbar()
    plt.title('StDev Rates')
    plt.xlabel('Mean')
    plt.ylabel('log std')
    
    # plt.subplot(3,3,3)
    # plt.imshow(simcounts.T,origin='lower',aspect='auto',
    #            extent=(-murange,murange,np.log10(noisestds[0]),np.log10(noisestds[-1])))
    # plt.colorbar()
    # plt.title('counts')
    # plt.xlabel('Mean')
    # plt.ylabel('log std')
    
    plt.subplot(7,3,(11,14))
    plt.plot(np.log10(noisestds),meanWAKE*np.ones(np.shape(noisestds)),
             'r-',linewidth=1,label='WAKE')
    plt.plot(np.log10(noisestds),
             stdWAKE+meanWAKE*np.ones(np.shape(noisestds)),
             'r--',linewidth=1)
    plt.plot(np.log10(noisestds),meanrate[idx0,:],'k-',label='SLEEP',linewidth=2)
    plt.plot(np.log10(noisestds),meanrate[idx0,:]+stdrate[idx0,:],'k--',label='+std',linewidth=2)
    plt.xlabel('Log Std')
    plt.ylabel('Mean Rate')
    plt.xlim((np.log10(noisestds[0]),np.log10(noisestds[-1])))
    plt.ylim((0,1.5))
    plt.legend()
    
    
    if decoder:
        plt.subplot(3,3,3)
        plt.imshow(1-coverage.T,origin='lower',aspect='auto',
                   extent=(-murange,murange,np.log10(noisestds[0]),np.log10(noisestds[-1])))
        plt.plot([0,0],np.log10(noisestds[[0,-1]]),'w--')
        plt.colorbar()
        plt.title('Coverage')
        plt.xlabel('Mean')
        plt.ylabel('log std')
        
        plt.subplot(3,3,4)
        plt.imshow(continuity.T,origin='lower',aspect='auto',
                   extent=(-murange,murange,np.log10(noisestds[0]),np.log10(noisestds[-1])))
        plt.plot([0,0],np.log10(noisestds[[0,-1]]),'w--')
        plt.colorbar()
        plt.title('Continuity')
        plt.xlabel('Mean')
        plt.ylabel('log std')
    
    ax = plt.subplot(6,3,17)
    plt.hist((actStats_wake['meancellrates']),histtype=u'step',color='r',
             bins=np.linspace(0,1,21),label='WAKE')
    for s,sample in enumerate(examples):
        plt.hist((actStats[s]['meancellrates']),histtype=u'step',
                 bins=np.linspace(0,1,21),label=f'noise:{examples[s][1]}',
                 color=np.array([1,1,1])/(s+2),linewidth=1)
    plt.legend()
    plt.xlabel('Mean FR')
    plt.ylabel('# Units')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    
    ax = plt.subplot(5,3,13)
    predictiveNet.plotActivationTimeseries(h_wake)
    plt.ylim((0,1.5))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_color('red')
        ax.spines[axis].set_linewidth(3)
    
    for s,sample in enumerate(examples):
        ax = plt.subplot(5,3,9+(s*3))
        predictiveNet.plotActivationTimeseries(h[s])
        plt.ylim((0,1.5))
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_color(np.array([1,1,1])/(s+2))
            ax.spines[axis].set_linewidth(3)

    
    if savename is not None: 
        saveFig(plt.gcf(),savename+'_SpontActivityFigure',savefolder,
                filetype='png')

    # plt.show()
    

def SpontActivityExamples(predictiveNet, examples=((0,0.1),(0,0.25),(0,0.4)),
                          timesteps=4000,
                          savename=None, savefolder=None, decoder=None):
    
    h, actStats,coverage,continuity = {},{},{},{}    
    numex = len(examples)
    
    for s,sample in enumerate(examples):
        obs_pred,h[s],noise_t = predictiveNet.spontaneous(timesteps,sample[0],sample[1])
        actStats[s] = predictiveNet.calculateActivationStats(h[s])

        if decoder:
            onset = 100
            decoded, p = predictiveNet.decode(h[s],decoder)
            coverage[s] = trajectoryAnalysis.calculateCoverage(decoded[onset:],
                                                            [1,decoder.gridheight,
                                                             1,decoder.gridwidth])
            continuity[s] = trajectoryAnalysis.calculateContinuity(decoded[onset:],showFig=False)
            numdelays = continuity[s]['delays'][-1]
            maxdist = continuity[s]['dxbins'][-1]+1

    plt.figure(figsize=(15,15))
    for s,sample in enumerate(examples):
        ax = plt.subplot(numex,3,1+(s*3))
        predictiveNet.plotActivationTimeseries(h[s])
        #plt.ylim((0,1.5))
        # for axis in ['top','bottom','left','right']:
        #     ax.spines[axis].set_color(np.array([1,1,1])/(s+2))
        #     ax.spines[axis].set_linewidth(3)

        ax1 = plt.subplot(numex,6,4+(s*6))
        ax2 = ax1.twinx()
        ax1.imshow(continuity[s]['delay_dist'],origin='lower',
                   extent=(0.5,numdelays+0.5,-0.5,maxdist-0.5),aspect='auto')
        ax2.plot(continuity[s]['delays'],continuity[s]['delay_kl'],'.--r')
        ax1.text(numdelays-2,maxdist-2, f"{continuity[s]['underthresh']}", fontsize=10,color='r')
        ax1.set_xlabel('dt')
        ax1.set_ylabel('dx')
        ax2.set_ylabel('K-L Div from P[dx]')
    
    
        plt.subplot(numex,3,3+(s*3))
        plt.imshow(coverage[s]['occupancy'].transpose(),
                       interpolation='nearest',alpha = decoder.mask.transpose())
        plt.text(1, 1, f"{1-coverage[s]['nonuniformity']:0.1}", fontsize=10,color='r')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.title('Occupancy')
        
    if savename is not None:
        saveFig(plt.gcf(),savename+'_SpontaneousExamples',savefolder,
                filetype='png')

    # plt.show()


def SpontTrajectoryFigure(predictiveNet, decoder, noisemag=0, noisestd=0.25,  
                          timesteps=5000,
                          savename=None, savefolder=None):
    
    obs_pred,h,noise_t = predictiveNet.spontaneous(timesteps,noisemag,noisestd)
    #obs_pred = predictiveNet.pred2np(obs_pred)

    decoded, p = predictiveNet.decode(h,decoder)
    maxp = np.max(p,axis=(1,2))
    dx = np.sum(np.abs(decoded.values[:-1,:] - decoded.values[1:,:]), axis=1)

    continuity = trajectoryAnalysis.calculateContinuity(decoded,showFig=True)
    numdelays = continuity['delays'][-1]
    maxdist = continuity['dxbins'][-1]+1

    pdhist,pbinedges,dtbinedges = np.histogram2d(dx,np.log10(maxp[:-1]),
                            bins=[np.arange(0,15),np.linspace(-1,0,11)])
    pdhist = pdhist/np.sum(pdhist,axis=0)

    coverage = trajectoryAnalysis.calculateCoverage(decoded,
                                                    [1,decoder.gridheight,
                                                     1,decoder.gridwidth])

    plt.figure(figsize=(10,10))
    plt.subplot(3,3,9)
    #plt.plot(np.log10(maxp[:-1]),dx,'.')
    plt.imshow(pdhist[0:9,:],origin='lower',extent=(-1,-0,-0.5,9.5),aspect='auto')
    plt.xlabel('log p_x')
    plt.ylabel('dx')

    ax1 = plt.subplot(5,5,10)
    ax2 = ax1.twinx()
    ax1.imshow(continuity['delay_dist'],origin='lower',extent=(0.5,numdelays+0.5,-0.5,maxdist-0.5),aspect='auto')
    ax2.plot(continuity['delays'],continuity['delay_kl'],'.--r')
    ax1.text(numdelays-2,maxdist-2, f"{continuity['underthresh']}", fontsize=10,color='r')
    ax1.set_xlabel('dt')
    ax1.set_ylabel('dx')
    ax2.set_ylabel('K-L Div from P[dx]')


    plt.subplot(5,4,7)
    plt.imshow(coverage['occupancy'].transpose(),
               interpolation='nearest',alpha=decoder.mask.transpose())
    plt.text(1, 1, f"{coverage['nonuniformity']:0.1}", fontsize=10,color='r')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title('Occupancy')

    ax1 = plt.subplot(5,2,1)
    ax2 = ax1.twinx()
    ax2.plot(dx[range(timesteps-25,timesteps-1)],color='grey')
    ax1.plot(maxp[range(timesteps-25,timesteps-1)],color='k')
    plt.xlabel('t')
    ax2.set_ylabel('dx')
    ax1.set_ylabel('px')

    plt.subplot(5,2,3)
    predictiveNet.plotActivationTimeseries(h)

    predictiveNet.plotSequence(predictiveNet.env_shell.pred2np(obs_pred,
                                                               timesteps=range(timesteps-6,timesteps)), 
                                                               range(6),4,label='Predicted')
    predictiveNet.plotSequence(np.transpose(p,axes=[0,2,1]), range(timesteps-7,timesteps-1),5,
                      label='Decoded',mask=decoder.mask.transpose())


    if savename is not None:
        saveFig(plt.gcf(),savename+'_SpontaneousTrajectory',savefolder,
                filetype='png')

    # plt.show()



def OfflineTrajectoryProps(predictiveNet, decoder, 
                           timesteps=5000, noisemag=0, logstdrange=(-2,0.5),
                           numpoints=21,
                           savename=None, savefolder=None):

    noisestds = np.logspace(logstdrange[0],logstdrange[1],numpoints)
    
    spCoverage = np.zeros_like(noisestds)
    stContinuity = np.zeros_like(noisestds)
    coverage,continuity = {},{}
    for j,noisestd in enumerate(noisestds):
        #Get a trajectory
    
        #noisestd = 0.25
        meanrate = np.inf
        while meanrate > 5:
            obs_pred,h,noise_t = predictiveNet.spontaneous(timesteps,noisemag,noisestd)
            actStats = predictiveNet.calculateActivationStats(h)
            meanrate = actStats['meanrate']
            if np.isnan(meanrate): meanrate = np.inf
            
            decoded, p = predictiveNet.decode(h,decoder)
        
            coverage[j] = trajectoryAnalysis.calculateCoverage(decoded,[1,decoder.gridheight,
                                                                     1,decoder.gridwidth])
            continuity[j] = trajectoryAnalysis.calculateContinuity(decoded)
        
            spCoverage[j] = coverage[j]['nonuniformity']
            stContinuity[j] = continuity[j]['underthresh']
            
    
    examples = np.linspace(0,numpoints-1,5).astype(int)
    
    plt.figure()
    plt.subplot(4,3,8)
    plt.plot(np.log10(noisestds),1-spCoverage)
    plt.plot(np.log10(noisestds[examples]),np.ones_like(examples)+0.1,'r^')
    plt.ylabel('Coverage')
    plt.xticks([])
    plt.ylim([0,1.2])
    
    plt.subplot(4,3,11)
    plt.plot(np.log10(noisestds[examples]),0.5*np.ones_like(examples)+0.1,'w^')
    plt.plot(np.log10(noisestds),stContinuity)
    plt.ylim([0,0.5])
    plt.xlabel('Log Noise Std')
    plt.ylabel('Continuity')
    
    for e,eidx in enumerate(examples):
        plt.subplot(4,5,e+1)
        plt.imshow(coverage[eidx]['occupancy'].transpose(),
                   interpolation='nearest',alpha = decoder.mask.transpose())
        # plt.xticks([])
        # plt.yticks([])
        plt.axis('off')
        plt.subplot(4,5,e+6)
        plt.imshow(continuity[eidx]['delay_dist'],
                   origin='lower',extent=(0.5,10+0.5,-0.5,15-0.5),aspect='auto')
        if e>0:
            plt.xticks([])
            plt.yticks([])
        else:
            plt.xlabel('dt')
            plt.ylabel('dx')
    if savename is not None:
        saveFig(plt.gcf(),savename+'_OfflineTrajectoryProps',savefolder,
                filetype='png')
    
    # plt.show()
    
    
from sklearn import manifold
def IsoMapFigure(predictiveNet,env,agent, noisemag=0, noisestd=0.25, 
                 timesteps_wake=5000,timesteps_sleep=1000,
                 savename=None, savefolder=None,
                 usecells=None):

    #Collect h trajecotry
    
    obs,act,state,_ = predictiveNet.collectObservationSequence(env, agent, timesteps_wake)
    obs_pred, obs_next, h = predictiveNet.predict(obs,act)
    

    obs_pred,h_t,noise_t = predictiveNet.spontaneous(timesteps_sleep, noisemag, noisestd)
    
    actnp = act.detach().numpy()
    whichact = np.argmax(actnp,axis=2)
    
    inHD = state['agent_dir'][:-1]
    nextHD = state['agent_dir'][1:]
    
    
    n_neighbors = 50
    n_components = 3
    X = np.concatenate((np.squeeze(h.detach().numpy()),np.squeeze(h_t.detach().numpy())))
    method = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components,p=1)
    
    numcells = np.size(X,1)
    X = X[:,usecells]
    print(f"Using {np.size(X,1)} of {numcells} cells")
    
    Y = method.fit_transform(X)
    
    
    color = np.arctan(state['agent_pos'][:,0]/state['agent_pos'][:,1])
    #color = state['agent_dir']
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(3,3,1,projection='3d')
    ax.scatter(Y[:(timesteps_wake+1), 0], Y[:(timesteps_wake+1), 1],Y[:(timesteps_wake+1), 2], c=color,marker='.',s=5)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_title('Position')
    
    plt.subplot(4,4,3)
    plt.scatter(state['agent_pos'][:,0],state['agent_pos'][:,1], c=color,marker='.',s=5)
    plt.xticks([])
    plt.yticks([])
    
    SWcolor = np.concatenate((np.ones((timesteps_wake)),np.zeros((timesteps_sleep))))
    ax = fig.add_subplot(3,3,7,projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1],Y[:, 2], c=SWcolor,marker='.',s=5)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_title('Sleep-Wake')
    
    ACTcolor = whichact
    ax = fig.add_subplot(3,3,4,projection='3d')
    ax.scatter(Y[:(timesteps_wake), 0], Y[:(timesteps_wake), 1],Y[:(timesteps_wake), 2], c=ACTcolor,marker='.',s=5)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_title('Action')
    
    ax = fig.add_subplot(3,3,5,projection='3d')
    ax.scatter(Y[:(timesteps_wake), 0], Y[:(timesteps_wake), 1],Y[:(timesteps_wake), 2], c=inHD,marker='.',s=5)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_title('this HD')
    
    ax = fig.add_subplot(3,3,6,projection='3d')
    ax.scatter(Y[:(timesteps_wake), 0], Y[:(timesteps_wake), 1],Y[:(timesteps_wake), 2], c=nextHD,marker='.',s=5)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_title('next HD')
    
    
    if savename is not None:
        saveFig(plt.gcf(),savename+'_Isomap',savefolder,
                filetype='png')
    # plt.show()