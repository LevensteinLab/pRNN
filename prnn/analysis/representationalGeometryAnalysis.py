#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:35:12 2022

@author: dl2820
"""
import numpy as np
from prnn.utils.general import saveFig
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from prnn.utils.agent import RandomActionAgent
from sklearn import manifold
from scipy.stats.mstats import spearmanr as spearmanr_m
from scipy.stats import spearmanr
import copy
from scipy.linalg import toeplitz
from scipy.optimize import curve_fit
from prnn.utils.ActionEncodings import OneHot

defaultMetric = 'cosine'
maxNtimesteps = 4000


def randSubSample(h, maxN, axis=0):
    #pick random N of timesteps if bigger than maxN timesteps
    nT = np.size(h,axis)
    randIDX = np.arange(nT)
    if nT > maxN:
        randIDX = np.random.randint(0, high=nT, size=maxN)
        h = h[randIDX,:]
    return h, randIDX

class representationalGeometryAnalysis:
    def __init__(self, predictiveNet, timesteps_wake = 15000,
                 noisemag = 0, noisestd = 0.1, timesteps_sleep = 1000,
                 withIsomap = False, n_neighbors=150,
                 SIdependence=True, spacemetric = 'euclidean',
                 actRSA = True, obsRSA=True, HDRSA = True, theta = 'expand',
                 agent = False, start_pos = 1, mapcenter = [18,18],
                 discretize = False, inv_x = False, inv_y = False):
        self.pN = predictiveNet
        
        env = predictiveNet.EnvLibrary[0]
        self.cont = env.continuous
        self.max_dist = env.max_dist
        self.start_pos = start_pos # the numbering of occupiable locations starts from this
        self.mapcenter = mapcenter
        
        # this is for backward compatibility, better provide the agent as an arg
        if not agent:
            action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
            agent = RandomActionAgent(env.action_space, action_probability)
        
        self.WAKEactivity = self.runWAKE(env, agent, timesteps_wake, theta=theta,
                                         discretize=discretize, inv_x=inv_x, inv_y=inv_y)
        
        self.RSA_cb = []
        self.RSA_cs = []
        if type(spacemetric) is not list:
            spacemetric = [spacemetric]
        for sidx,metric in enumerate(spacemetric):
            self.RSA_cb.append(self.calculateRSA_space(self.WAKEactivity, 'cityblock', spacemetric =metric,
                                                       cont=self.cont, max_dist=self.max_dist))
            self.RSA_cs.append(self.calculateRSA_space(self.WAKEactivity, 'cosine', spacemetric=metric,
                                                       cont=self.cont, max_dist=self.max_dist))
        if len(self.RSA_cb) == 1:
            self.RSA_cb = self.RSA_cb[0]
            self.RSA_cs = self.RSA_cs[0]

        self.hill_fit = self.calculateHillFit(self.WAKEactivity)
        
        self.actEncode_cs = False
        if actRSA:
            self.actEncode_cs = self.calculateRSA_action(self.WAKEactivity, 'cosine')
            
        self.RSA_obs = False
        if obsRSA:
            self.RSA_obs = self.calculateRSA_obs(self.WAKEactivity, 'cosine')
            
        self.RSA_HD = False
        if HDRSA:
            self.RSA_HD = self.calculateRSA_HD(self.WAKEactivity, 'cosine')
        
        self.SLEEPactivity = None
        if timesteps_sleep>0:
            self.SLEEPactivity = self.runSLEEP(noisemag, noisestd, timesteps_sleep)
            self.SWdist_cb = self.calculateSleepWakeDist(self.WAKEactivity['h'], 
                                                         self.SLEEPactivity['h'],
                                                         metric='cityblock')
            self.SWdist_cs = self.calculateSleepWakeDist(self.WAKEactivity['h'], 
                                                         self.SLEEPactivity['h'],
                                                         metric='cosine')
        
        
        self.Isomap = withIsomap
        if self.Isomap:
            self.Isomap = self.fitIsomap(self.WAKEactivity, self.SLEEPactivity,
                                         n_neighbors = n_neighbors)

            
    def calculateRSA_action(self,WAKEactivity, metric=defaultMetric):
        dists,keepIDX = self.calculateNeuralDistWAKE(WAKEactivity['h'], metric)
        actID, actdist = self.getActionIDs(keepIDX)
        actsort = np.argsort(actID).flatten()
        numacts,_ = np.histogram(actID,
                                 bins=np.linspace(-0.5,
                                                  np.max(actID)+0.5,
                                                  np.max(actID)+2))
        
        if metric == 'cityblock' :  goodmax = 0.4
        elif metric == 'cosine'  :  goodmax = 1
        
        (hist2,sbins,rbins) = np.histogram2d(dists, actdist, 
                                             bins=[np.linspace(0,goodmax,25),
                                                   np.arange(-0.5,2.5,1)])
        hist2 = hist2/np.sum(hist2,axis=0)
        #NOTE: DOES THIS NEED TO BE FLATTENED?!?!?!?!? (check that it's the same)
        RSA = spearmanr(dists,actdist)

        return RSA, hist2, rbins, sbins, dists, actsort, numacts
    
    def actionDistancePanel(self,RSA):
        RSA, hist2, rbins, sbins, dists, actsort, numacts = RSA
        actionlabels = ['Rotate L','Rotate R','Move Forward','Hold']
        vmin = np.mean(dists)-1*np.std(dists)
        vmax = np.mean(dists)+1*np.std(dists)
        
        ndists = squareform(dists)
        plt.imshow(ndists[np.ix_(actsort,actsort)],
                   vmin=vmin, vmax=vmax)

        drawGroupLines(numacts,actionlabels)
        clb = plt.colorbar()
        clb.ax.set_ylabel('Neural Distance')
        #plt.show()
    
    def samediffActionPanel(self,RSA):
        RSA, hist2, sbins, rbins, dists, actsort, numacts = RSA
        
        #plt.boxplot(hist2,labels=['same','diff'])
        plt.plot(rbins[:-1]+0.5*np.diff(rbins[0:2]),hist2)
    
    def ActionRSAFigure(self,netname,savefolder):
        plt.figure()
        plt.subplot(2,2,1)
        self.actionDistancePanel(self.actEncode_cs)

        plt.subplot(2,2,2)
        self.samediffActionPanel(self.actEncode_cs)
        #plt.subplot(2,2,3)
        #self.spatialRSApanel(self.RSA_cs,'cosine')
        #plt.xlabel('Spatial Dist')
        
        if self.Isomap:
            plt.subplot(3,3,7)
            self.isomapPanel('action')
            
            plt.subplot(3,3,8)
            self.isomapPanel('position')

        saveFig(plt.gcf(),'ActionRSA_'+netname,savefolder,
                filetype='png')
        plt.show()
    
    def getActionIDs(self,keepIDX=None):
        #actnp = self.WAKEactivity['act'].detach().numpy()
        act = OneHot(self.WAKEactivity['act_env'],None)
        actnp = act.detach().numpy()
        actID = np.argmax(actnp,axis=2)
        if keepIDX is not None:
            actID = actID[:,keepIDX]
            actnp = actnp[:,keepIDX,:]
        
        dists = pdist(np.squeeze(actnp[:,:,:np.max(actID)+1]),'cosine')
        return actID, dists
        
    def runWAKE(self, env, agent, timesteps_wake, theta='expand',
                discretize=False, inv_x=False, inv_y=False):
        print('Running WAKE')
        a = {}
        a['obs_env'],a['act_env'],a['state'],_ = self.pN.collectObservationSequence(env,
                                                            agent,
                                                            timesteps_wake,
                                                            obs_format=None,
                                                            discretize=discretize,
                                                            inv_x=inv_x, inv_y=inv_y)
        a['obs'],a['act'] = env.env2pred(a['obs_env'],a['act_env'])
        a['obs_pred'], a['obs_next'], h = self.pN.predict(a['obs'],a['act'])
        
        if theta == 'mean':
            h = h.mean(axis=0,keepdims=True)
            a['act'] = a['act'][:,:h.size(dim=1),:]
            a['state']['agent_pos'] = a['state']['agent_pos'][:h.size(dim=1)+1,:]
        if theta == 'expand':
            k = h.size(dim=0)
            h = h.transpose(0,1).reshape((-1,1,h.size(dim=2))).swapaxes(0,1)
            a['state']['agent_pos'] = np.repeat( a['state']['agent_pos'], k, axis=0)
            a['state']['agent_pos'] = a['state']['agent_pos'][:h.size(dim=1)+1,:]
            
            
        a['h'] = np.squeeze(h.detach().numpy())
        return a

    def runSLEEP(self, noisemag, noisestd, timesteps_sleep):
        print('Running SLEEP')
        a = {}
        a['obs_pred'],h_t,a['noise_t'] = self.pN.spontaneous(timesteps_sleep,
                                                   noisemag,
                                                   noisestd)
        a['h'] = np.squeeze(h_t.detach().numpy())
        return a
        
    def fitIsomap(self,WAKEactivity, SLEEPactivity, usecells=None, n_neighbors=150):
        print('Fitting Isomap')
        #X = self.WAKEactivity['h']
        h_wake = WAKEactivity['h']
        h_sleep = SLEEPactivity['h']
        
        h_wake, keepIDX = randSubSample(h_wake, maxNtimesteps, axis=0)
        
        if usecells is not None: 
            h_wake = h_wake[:,usecells]
            h_sleep = h_sleep[:,usecells]
            
        X = np.concatenate((h_wake,h_sleep))
        #method = manifold.Isomap(n_neighbors=50, n_components=2,p=1)
        method = manifold.Isomap(n_neighbors=n_neighbors, n_components=2, metric='cosine')
        method.fit(X)
        return method
    
    @staticmethod
    def calculateNeuralDistWAKE(WAKEactivity_h, metric=defaultMetric,
                                usecells=None):
        
        h_np, keepIDX = randSubSample(WAKEactivity_h, maxNtimesteps, axis=0)
        
        
        if usecells is not None: 
            h_np = h_np[:,usecells]
        
        if metric == 'cityblock':
            N = np.size(h_np,1)
            dists = pdist(h_np,'cityblock')
            dists = dists/N
        elif metric == 'cosine':
            dists = pdist(h_np,'cosine')
        return dists, keepIDX
    
    @staticmethod
    def calculateSpatialDist(WAKEactivity_state, keepIDX=None, metric='cityblock'):
        position = WAKEactivity_state['agent_pos'][:-1,:]
        if keepIDX is not None:
            position = position[keepIDX,:]
            
        sp_dists = pdist(position,metric)
        return sp_dists
    
    def calculateRSA_space(self, WAKEactivity, metric=defaultMetric,
                          usecells = None, spacemetric='euclidean',
                          cont=False, max_dist=False):
        dists,keepIDX = self.calculateNeuralDistWAKE(WAKEactivity['h'], 
                                                     metric, usecells)
        sp_dists = self.calculateSpatialDist(WAKEactivity['state'],keepIDX,
                                            metric=spacemetric)
        
        if metric == 'cityblock':
            goodmax = 0.9
        elif metric == 'cosine':
            goodmax = 1
        
        if cont:
            (hist2,sbins,rbins) = np.histogram2d(dists, sp_dists, 
                                             bins=[np.linspace(0,goodmax,50),
                                                   np.arange(0,max_dist*0.8,0.1)])
        else:
            (hist2,sbins,rbins) = np.histogram2d(dists, sp_dists, 
                                                bins=[np.linspace(0,goodmax,50),
                                                    np.arange(-0.5,16.5,1)])
        hist2 = hist2/np.sum(hist2,axis=0)
        #RSA = np.corrcoef(sp_dists,dists)[0,1]
        RSA = spearmanr(dists,sp_dists)

        return RSA, hist2, sbins, rbins

    def calculateHillFit(self, WAKEactivity, metric='cosine',
                          usecells = None, spacemetric='euclidean'):

        dists,keepIDX = self.calculateNeuralDistWAKE(WAKEactivity['h'], 
                                                     metric, usecells)
        sp_dists = self.calculateSpatialDist(WAKEactivity['state'],keepIDX,
                                            metric=spacemetric)

        x0_guess = np.mean(dists[sp_dists < 1])
        t_half_guess = np.mean(sp_dists)
        xinf_guess = np.mean(dists[sp_dists > t_half_guess])

        initial_guess = [x0_guess, xinf_guess, t_half_guess, 2.0]  # Adjust as needed
        bounds = ([0, 0, 0, 0.1], [1, 1, np.inf, 10])  # Optional

        popt, pcov = curve_fit(fit_func, sp_dists, dists, p0=initial_guess, bounds=bounds)
        fitted_x0, fitted_x_inf, fitted_t_half, fitted_n = popt
     
        #For Plotting
        sp_bins = np.arange(-0.5,21.5,1)
        sp_centers = (sp_bins[:-1] + sp_bins[1:]) / 2
        bin_indices = np.digitize(sp_dists, sp_bins)
        hdist_means = [dists[bin_indices == i].mean() for i in range(1, len(sp_bins))]
        hdist_stds = [dists[bin_indices == i].std() for i in range(1, len(sp_bins))]

        hill_fit = {
            'x0': fitted_x0,
            'x_inf': fitted_x_inf,
            't_half': fitted_t_half,
            'n': fitted_n,
            'sp_centers' : sp_centers,
            'hdist_means' : hdist_means,
            'hdist_stds' : hdist_stds
        }

        return hill_fit
    



    #@staticmethod
    def calculateObsDist(self, WAKEactivity_obs, keepIDX=None):
        obs = WAKEactivity_obs.squeeze().detach().numpy()
        numobs = obs.shape[1]
        if keepIDX is not None:
            obs = obs[keepIDX,:]
            
        sp_dists = pdist(obs,'cityblock')/numobs
        return sp_dists
    
    def calculateHDDist(self, WAKEactivity_state, keepIDX=None):
        hd = np.expand_dims(WAKEactivity_state['agent_dir'],1)
        if keepIDX is not None:
            hd = hd[keepIDX,:]
        if self.cont:
            hd_dists = pdist(hd)
            hd_dists[hd_dists>np.pi] = 2*np.pi - hd_dists[hd_dists>np.pi] #Circular
        else:
            hd_dists = pdist(hd,'cityblock')
            hd_dists[hd_dists==3] = 1 #Circular
        return hd_dists
    
    def getLocationGroups(self, WAKEactivity_state, keepIDX=None):
        sp_dists = self.calculateSpatialDist(WAKEactivity_state, keepIDX=keepIDX)
        hd_dists = self.calculateHDDist(WAKEactivity_state, keepIDX=keepIDX)
        
        locGroups = {
            'farLocation' : sp_dists>5,
            'diffLocation' : sp_dists>0,
            'diffHD'       : np.logical_and(sp_dists==0, hd_dists>0),
            'sameView'     : np.logical_and(sp_dists==0, hd_dists==0)
        }
        
        return locGroups
    
    def calculateRSA_obs(self, WAKEactivity, metric=defaultMetric,
                        usecells = None):
        dists,keepIDX = self.calculateNeuralDistWAKE(WAKEactivity['h'], 
                                                     metric, usecells)
        obs_dists = self.calculateObsDist(WAKEactivity['obs'],keepIDX)
        
        locationGroups = self.getLocationGroups(WAKEactivity['state'],keepIDX)
        
        if metric == 'cityblock':
            goodmax = 0.9
        elif metric == 'cosine':
            goodmax = 1
        
        RSA_obs = {}
        for key, value in locationGroups.items():
            (hist2,sbins,rbins) = np.histogram2d(dists[value], 
                                                 obs_dists[value], 
                                                 bins=[np.linspace(0,goodmax,50),
                                                       np.arange(-0.01,0.21,0.02)])
            hist2 = hist2/np.sum(hist2,axis=0)
            RSA = spearmanr(dists[value],obs_dists[value])
            RSA_obs[key] = (RSA, hist2, sbins, rbins)

        return RSA_obs
    
    
    def calculateRSA_HD(self, WAKEactivity, metric=defaultMetric,
                        usecells = None):
        dists,keepIDX = self.calculateNeuralDistWAKE(WAKEactivity['h'], 
                                                     metric, usecells)
        HD_dists = self.calculateHDDist(WAKEactivity['state'],keepIDX)
        
        
        if metric == 'cityblock':
            goodmax = 0.9
        elif metric == 'cosine':
            goodmax = 1
        

        hist2,sbins,rbins = np.histogram2d(dists, HD_dists, 
                                            bins=[np.linspace(0,goodmax,50),
                                                  np.arange(-0.5,3.5,1)])
        hist2 = hist2/np.sum(hist2,axis=0)
        RSA = spearmanr(dists,HD_dists)

        return RSA, hist2, sbins, rbins
    
    
    def calculateSIdependence(self, WAKEactivity, 
                              metric=defaultMetric, maxSI=1, numSI=20,
                              exampleSI=0.15, cont=False, max_dist=False):
        
        SI = np.squeeze(self.pN.TrainingSaver['SI'].values[-1])

        SIvals = np.linspace(0,maxSI,numSI)
        RSA_SIthresh = np.zeros_like(SIvals)
        for sIDX,SIthreshold in enumerate(SIvals):
            overthresh = SI>=SIthreshold
            (RSA_SIthresh[sIDX],_),_,_,_ = self.calculateRSA_space(WAKEactivity,
                                               metric=metric,usecells=overthresh,
                                               cont=cont, max_dist=max_dist)
            
        #Run the example   
        excells = SI>=exampleSI
        exampleSI_RSA = self.calculateRSA_space(self.WAKEactivity,
                                               metric=metric,usecells=excells,
                                               cont=cont, max_dist=max_dist)
        exIsomap = None
        #if self.Isomap:
        #    exIsomap = self.fitIsomap(self.WAKEactivity,self.SLEEPactivity,
        #                                 usecells=excells)
            
        return RSA_SIthresh, SIvals, exampleSI_RSA, exIsomap
    
    def calculateTuningCurveControl(self, WAKEactivity, metric=defaultMetric,
                                    cont=False, max_dist=False):
        #FAKEactivity = copy.deepcopy(WAKEactivity)
        FAKEactivity = {'state':WAKEactivity['state']}
        FAKEactivity = self.makeFAKEdata(WAKEactivity)
        
        RSA_fake = self.calculateRSA_space(FAKEactivity, metric,
                                           cont=cont, max_dist=max_dist)
        return RSA_fake
    
    def makeFAKEdata(self,WAKEactivity, useMstats=False):
        FAKEactivity = {'state':WAKEactivity['state']}
        position = WAKEactivity['state']['agent_pos']
        tuning_curves = self.pN.TrainingSaver['place_fields'].values[-1]
        FAKEactivity['h'] = np.zeros_like(WAKEactivity['h'])
        for cell,(k,tuning_curve) in enumerate(tuning_curves.items()):
            FAKEactivity['h'][:,cell] = tuning_curve[position[:-1,0]-self.start_pos,
                                                     position[:-1,1]-self.start_pos]
        
        #Calculate the tuning-curve reliability 
        
        FAKEcorr = spearmanr(WAKEactivity['h'],FAKEactivity['h'],
                            axis=0)
        if useMstats and FAKEcorr.correlation.size==1 and np.isnan(FAKEcorr.correlation).all():
            print(FAKEcorr.correlation)
            print('correlation nan, using mstats (slower)')
            FAKEcorr = spearmanr_m(WAKEactivity['h'],FAKEactivity['h'],
                                axis=0)
            print(FAKEcorr.correlation)
        Nneurons = np.size(WAKEactivity['h'],1)
        FAKEcorr = np.diagonal(FAKEcorr.correlation,offset = Nneurons)
        FAKEactivity['TCcorr'] = FAKEcorr
            
        return FAKEactivity
        
    
    def spatialRSApanel(self, RSA, unitlabel, 
                        colorbar=True, vmin=None, vmax=None):
        sRSA, hist2, sbins, rbins = RSA
        if vmax == 'auto':
            vmax = np.median(hist2.max(0))
        
        plt.imshow((hist2), origin='lower', aspect='auto',
                  extent=(rbins[0],rbins[-1],sbins[0],sbins[-1]),
                  cmap='binary', vmin=vmin, vmax=vmax,
                  interpolation='none')
        #ax
        plt.text(rbins[1], sbins[-6], f"sRSA: {sRSA[0]:0.01}", fontsize=10,color='r')
        #plt.xlabel('Spatial Dist')
        plt.ylabel('Neural Dist ('+unitlabel+')')
        if colorbar:
            clb = plt.colorbar()
            clb.ax.set_ylabel('P[neural|space]')
        
    def siPanel(self, SIdependence, FAKE_SIdep=None):
        RSA_SIthresh, SIvals, exampleSI_RSA, exIsomap = SIdependence
        SI = np.squeeze(self.pN.TrainingSaver['SI'].values[-1])
        
        
        plt.ylabel('sRSA')
        plt.xlabel('SI Threshold')
        ax1 = plt.gca()
        ax2 = ax1.twinx()    
        ax2.hist(SI,density=True,color='r', alpha=0.5)
        ax1.plot(SIvals,RSA_SIthresh,'k',linewidth=2,
                label='Real')
        if FAKE_SIdep is not None:
            FRSA_SIthresh, _, _, _ = FAKE_SIdep
            ax1.plot(SIvals,FRSA_SIthresh,'k--',label='TC Control')
            ax1.legend(loc='center right')
        
    
    def runControlAnalyses(self, metric = defaultMetric, maxSI=0.25,
                          exampleSI=0.1, numSI=10, cont=False, max_dist=False):
        self.SIdep = self.calculateSIdependence(self.WAKEactivity,
                                          metric=metric,maxSI=maxSI,
                                          exampleSI=exampleSI,numSI=numSI,
                                          cont=cont, max_dist=max_dist)
        self.FAKEactivity = self.makeFAKEdata(self.WAKEactivity)
        self.SIdep_fake = self.calculateSIdependence(self.FAKEactivity,
                                          metric=metric,maxSI=maxSI,
                                          exampleSI=exampleSI,numSI=numSI,
                                          cont=cont, max_dist=max_dist)
        self.RSA_fake = self.calculateTuningCurveControl(self.WAKEactivity,
                                                         metric='cityblock',
                                                         cont=cont, max_dist=max_dist)
        
        print('Success!')
        return
    
    def TCSIpanel(self,excells=None):
        SI = np.squeeze(self.pN.TrainingSaver['SI'].values[-1])
        TCreliability = self.FAKEactivity['TCcorr']
        plt.plot(SI,TCreliability,'.')
        for excell in excells:
            plt.plot(SI[excell],TCreliability[excell],'kx')
        plt.xlabel('SI')
        plt.ylabel('Tuning Curve Variablity')
        
    def TCreliabilitypanel(self,excell):
        plt.plot(self.WAKEactivity['h'][:,excell],
                 self.FAKEactivity['h'][:,excell],'.',
                markersize=1)
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('real rate')
        plt.ylabel('Fake Rate')
        
    def tuningCurvepanel(self,excell):
        place_fields=self.pN.TrainingSaver['place_fields'].iloc[-1]
        totalPF = np.array(list(place_fields.values())).sum(axis=0)
        mask = np.array((totalPF>0)*1.)
        
        plt.imshow(place_fields[excell].transpose(),
                               interpolation='nearest',
                   alpha=mask.transpose())
        plt.axis('off')
        
    
    def SIDependenceFigure(self, netname=None, savefolder=None, excells=[0,1]):
        #Calculate RGA.SIdep = RGA.calculateSIdependence() first... This is sloppy
        SIdep = self.SIdep
        SIdep_fake = self.SIdep_fake
        RSA_fake = self.RSA_fake  

        plt.figure(figsize=(12, 8))
        plt.subplot(3,2,5)
        self.siPanel(SIdep,SIdep_fake)

        plt.subplot(4,4,1)
        self.spatialRSApanel(self.RSA_cb,'cityblock/N')

        plt.subplot(4,4,2)
        self.spatialRSApanel(SIdep[2],'cityblock/N')
        plt.ylabel('')
        
        plt.subplot(4,4,5)
        self.spatialRSApanel(RSA_fake,'cityblock/N')
        
        plt.subplot(4,4,6)
        self.spatialRSApanel(SIdep_fake[2],'cityblock/N')
        plt.ylabel('')
         
        
        plt.subplot(3,3,9)
        self.TCSIpanel(excells)
        
        for eidx,excell in enumerate(excells):
            plt.subplot(5,5,9+eidx)
            self.tuningCurvepanel(excell)

            plt.subplot(5,5,14+eidx)
            self.TCreliabilitypanel(excell)
        plt.ylabel(None)
        
        if netname is not None:
            saveFig(plt.gcf(),'SIDependence_'+netname,savefolder,
                    filetype='png')
        plt.show()
        
    def isomapPanel3d(self, colorvar='position', rotate=(0,0)):
        X = self.WAKEactivity['h']
        if colorvar == 'position':
            state = self.WAKEactivity['state']
            color = np.arctan(state['agent_pos'][:-1,0]/state['agent_pos'][:-1,1])
        elif colorvar == 'SleepWake':
            h_sleep = self.SLEEPactivity['h']
            color = np.concatenate((np.ones((X.shape[0])),np.zeros((h_sleep.shape[0]))))
            X = np.concatenate((X,h_sleep))
        elif colorvar == 'Sleep':
            h_sleep = self.SLEEPactivity['h']
            color = [0.5,0.5,0.5]
            X = h_sleep
        elif colorvar == 'HD':
            state = self.WAKEactivity['state']
            color = state['agent_dir'][:-1]
        elif colorvar == 'action':
            if not self.cont:
                color,_ = self.getActionIDs()
            else:
                color = self.WAKEactivity['act']
            
        Y = self.Isomap.transform(X)
        
        ax = plt.gca()
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], 
                   c=color, marker='.', s=4)
        ax.view_init(rotate[0]-140, rotate[1]+60)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.set_title(colorvar)
        return
    
    def isomapPanel(self, colorvar='position', onsetTransient=5):
        X = self.WAKEactivity['h']
        if colorvar == 'position':
            state = self.WAKEactivity['state']
            color = np.arctan((state['agent_pos'][:-1,0]-self.mapcenter[0])/
                              (state['agent_pos'][:-1,1]-self.mapcenter[1]))
        elif colorvar == 'SleepWake':
            h_sleep = self.SLEEPactivity['h'][onsetTransient:,:]
            color = np.concatenate((np.ones((X.shape[0])),np.zeros((h_sleep.shape[0]))))
            X = np.concatenate((X,h_sleep))
        elif colorvar == 'Sleep':
            h_sleep = self.SLEEPactivity['h'][onsetTransient:,:]
            color = np.tile([0.7,0,0],(np.size(h_sleep,axis=0),1))
            X = h_sleep
        elif colorvar == 'HD':
            state = self.WAKEactivity['state']
            color = state['agent_dir'][:-1]
        elif colorvar == 'action':
            if not self.cont:
                color,_ = self.getActionIDs()
            else:
                color = self.WAKEactivity['act']
        else:
            color= colorvar
            
        maxplotpoints = 10000    
        X, keepIDX = randSubSample(X, maxplotpoints, axis=0)
        color = color[keepIDX]
            
        
            
        Y = self.Isomap.transform(X)
        
        ax = plt.gca()
        ax.scatter(Y[:, 0], Y[:, 1], 
                   c=color, marker='.', s=4)
        #ax.view_init(rotate[0]-140, rotate[1]+60)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.axis('off')
        #ax.zaxis.set_ticklabels([])
        #ax.set_title(colorvar)
        return
    
    def keyPanel(self):
        pos = np.unique(self.WAKEactivity['state']['agent_pos'],axis=0)
        color = np.arctan((pos[:,0]-self.mapcenter[0])/
                          (pos[:,1]-self.mapcenter[1]))
        
        ax = plt.gca()
        ax.scatter(pos[:, 0], pos[:, 1], 
                   c=color, marker='.', s=4)
        #ax.view_init(rotate[0]-140, rotate[1]+60)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.axis('off')
        return    
    
    
    def SpatialRSAFigure(self, netname=None, savefolder=None):
                
        plt.figure()
        plt.subplot(2,2,1)
        self.spatialRSApanel(self.RSA_cb,'cityblock/N')

        plt.subplot(2,2,3)
        self.spatialRSApanel(self.RSA_cs,'cosine')
        plt.xlabel('Spatial Dist')
        
        if self.Isomap:
            #plt.subplot(2,2,2,projection='3d')
            #self.isomapPanel3d()
            plt.subplot(2,2,2)
            self.isomapPanel()
        if netname is not None:
            saveFig(plt.gcf(),'SpatialRSA_'+netname,savefolder,
                    filetype='png')
        plt.show()
    
    @staticmethod
    def calculateSleepWakeDist(h_wake, h_sleep, metric=defaultMetric):
        h_wake, keepIDX = randSubSample(h_wake, maxNtimesteps, axis=0)
        
        X = np.concatenate((h_wake,h_sleep))
        if metric == 'cityblock':
            N = X.shape[1]
            ndists = squareform(pdist(X,'cityblock'))
            ndists = ndists/N
        elif metric == 'cosine':
            ndists = squareform(pdist(X,'cosine'))

        SWdist = ndists[:h_wake.shape[0],h_wake.shape[0]:]
        dist_closest = np.min(SWdist,axis=0)
        SleepSimilarity = np.median(dist_closest)
        
        return SleepSimilarity, SWdist, dist_closest
    
    
    def sleepdistPanel(self,SWdist,rbins,colorbar=True): 
        SleepSimilarity, SWdist, dist_closest = SWdist
        
        
        #plt.imshow(SWdist)
        #plt.xlabel('Sleep Timestep')
        #plt.ylabel('Wake Timestep')
        #clb = plt.colorbar()
        #clb.ax.set_ylabel('Distance')

        #plt.subplot(2,2,1)
        n,bins = np.histogram(dist_closest,bins=rbins)
        n = n/np.sum(n)
        plt.imshow(np.expand_dims(n,axis=1),
                  origin='lower', 
                  extent=(0,0.1,rbins[0],rbins[-1]),
                  cmap='binary')
        #plt.axis('off')
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.xlabel('S-W')
        if colorbar:
            clb = plt.colorbar()
            clb.ax.set_ylabel('P[neural|space]')
        
    def WakeSleepFigure(self, netname, savefolder=None, 
                        isomapRotation=(0,0), withKey =True, withSleep=True):
        
        plt.figure()
        # plt.subplot(3,8,3)
        # self.sleepdistPanel(self.SWdist_cb, self.RSA_cb[2])
        #plt.xlabel('Sleep-Wake Dist')

        plt.subplot(3,8,7)
        self.sleepdistPanel(self.SWdist_cs, self.RSA_cs[2])
        
        # plt.subplot(3,4,1)
        # self.spatialRSApanel(self.RSA_cb, 'cityblock/N', colorbar=False)
        # plt.xlabel('Spatial Dist')

        plt.subplot(3,4,3)
        self.spatialRSApanel(self.RSA_cs, 'cosine', colorbar=False, vmax='auto')
        plt.xlabel('Spatial Dist')

        if self.Isomap:
            #plt.subplot(2,2,2,projection='3d')
            #self.isomapPanel3d('position')
            #self.isomapPanel3d('Sleep', rotate=isomapRotation)
            plt.subplot(3,3,1)
            self.isomapPanel('position')
            if withSleep:
                self.isomapPanel('Sleep')
            if withKey:
                plt.subplot(6,6,13)
                self.keyPanel()
        
        #plt.tight_layout()
        if savefolder is not None:
            saveFig(plt.gcf(), 'WakeSleepDistance_'+netname, savefolder,
                    filetype='png')

        plt.show()
        
        
    def AllRSAFigure(self, netname, savefolder):
        plt.subplot(3,3,1)
        self.spatialRSApanel(self.RSA_obs['farLocation'],'cos')

        plt.subplot(3,3,2)
        self.spatialRSApanel(self.RSA_obs['diffHD'],'cos')

        plt.subplot(3,3,3)
        self.spatialRSApanel(self.RSA_obs['sameView'],'cos')

        plt.subplot(3,3,4)
        self.spatialRSApanel(self.RSA_cs,'cos')

        plt.subplot(3,3,6)
        self.spatialRSApanel(self.RSA_HD,'cos')


        plt.subplot(3,3,7)
        self.samediffActionPanel(self.actEncode_cs)

        plt.subplot(3,3,8)
        self.actionDistancePanel(self.actEncode_cs)
        
        if self.Isomap:
            plt.subplot(3,3,9)
            self.isomapPanel('position')

        plt.tight_layout()
        saveFig(plt.gcf(),'AllRSA_'+netname,savefolder,
                filetype='png')
        plt.show()

    def hillFitPanel(self, hill_fit, color='k', datalabel='Data'):
        bin_centers = hill_fit['sp_centers']
        bin_means = hill_fit['hdist_means']
        bin_stds = hill_fit['hdist_stds']

        fitted_x0 = hill_fit['x0']
        fitted_x_inf = hill_fit['x_inf']
        fitted_t_half = hill_fit['t_half']
        fitted_n = hill_fit['n']

        t_plot = np.linspace(min(bin_centers), max(bin_centers), 100)
        # --- Generate fitted curve ---
        y_fitted = hill_saturating(t_plot, fitted_x0, fitted_x_inf, fitted_t_half, fitted_n)


        plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='none', capsize=5, 
                    alpha=0.5, color=color, label=datalabel)
        plt.plot(t_plot, y_fitted, 'r-', linewidth=2,
                color=color)
        plt.plot(0-0.5,fitted_x0,'<',color=color, label=f'dh_0: {fitted_x0:.2f}')
        plt.plot(t_plot[-1]+1,fitted_x_inf,'<',color=color, label=f'dh_inf: {fitted_x_inf:.2f}')
        plt.plot(fitted_t_half,1.1,'v',color=color, label=f'dx_1/2: {fitted_t_half:.2f}')
        plt.xlabel('Spatial Distance')
        plt.ylabel('Neural Distance (cos)')
        plt.legend()

    def saveAnalysis(self, savefolder):
        return
        
        
        
def drawGroupLines(numelements, labels=None):
    xlim = plt.xlim()
    ylim = plt.ylim()
    for a,_ in enumerate(numelements[:-1]):
        plt.plot(xlim,np.cumsum(numelements)[a]*np.ones(2,),'r')
        plt.plot(np.cumsum(numelements)[a]*np.ones(2,),ylim,'r')
    if labels is not None:
        ticks = np.cumsum(numelements)-0.5*numelements
        plt.xticks(ticks=ticks, labels=labels)
        plt.yticks(ticks=ticks, labels=labels)


# --- Logistic function with steepness parameter n ---
def hill_saturating(t, x0, x_inf, t_half, n):
    return x0 + (x_inf - x0) * (t**n) / (t**n + t_half**n)

# --- Fitting function ---
def fit_func(t, x0, x_inf, t_half, n):
    return hill_saturating(t, x0, x_inf, t_half, n)
        