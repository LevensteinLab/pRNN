import numpy as np
from scipy.signal import correlate2d
from scipy.stats import gmean
import pandas as pd
from utils.general import saveFig
import matplotlib.pyplot as plt
from analysis.SpatialTuningAnalysis import SpatialTuningAnalysis
import time

class RepresentationalConnectivityAnalysis:
    def __init__(self, predictiveNet, reliability_thresh = 0.5):
        
        self.pN = predictiveNet
        self.SIthresh = reliability_thresh
        self.PFmask = self.getPFmask(self.pN)
        
        self.pairData, self.distmat = self.buildPairData(self.pN, reliability_thresh)
        self.adjustThreshold(self.SIthresh,thtype='reliability')
        
        self.examples = self.getExamples(self.goodPairData)
        

        
        
    def adjustThreshold(self,newthresh, thtype='reliability'):
        if thtype == 'SIpercent':
            thtype = 'SI'
            newthresh = self.pairData['meanSI'].quantile(newthresh)
        elif thtype == 'reliabilitypercent':
            thtype = 'reliability'
            newthresh = self.pairData['meanReliability'].quantile(newthresh)
            
            
        self.SIthresh = newthresh
        self.thtype = thtype
        
        if self.thtype == 'reliability':
            overthresh = (self.pairData['reliability1']>self.SIthresh) & (self.pairData['reliability2']>self.SIthresh)
            #overthresh = self.pairData['meanReliability']>self.SIthresh
        
        elif self.thtype == 'SI':
            overthresh = (self.pairData['SI 1']>self.SIthresh) & (self.pairData['SI 2']>self.SIthresh)
            #overthresh = self.pairData['meanSI']>self.SIthresh
            
        self.goodPairData = self.pairData[overthresh]
        self.examples = self.getExamples(self.goodPairData)
        self.weightdist = self.calculateWeightByPercentile(self.pairData,
                                                          metric=self.thtype)
        self.weightdist_value = self.calculateWeightByValue(self.goodPairData,
                                                          metric='SI')
        self.weightcorr = self.calculateWeightDistRelationship(self.goodPairData,
                                                                   metric='SI',
                                                                  numquantiles=4)

        
    def getPFmask(self,pN):
        try: #Bug with nets that don't save whole training data
            tuningCurves = pN.TrainingSaver['place_fields'].values[-1]
        except:
            tuningCurves = pN.TrainingSaver['place_fields']
            
        totalPF = np.array(list(tuningCurves.values())).sum(axis=0)
        mask = np.array((totalPF>0)*1.)
        return mask
        
        
    def getExamples(self, goodPairData, farThresh=8, closeThresh=3):
        excell = goodPairData.iloc[19]['cellIDX2']
        #Find a pair with this cell with high spatial distance and one with low spatial distance
        farPair = goodPairData[(goodPairData['cellIDX2']==excell) & (goodPairData['PeakDist']>farThresh)].iloc[0]
        closePair = goodPairData[(goodPairData['cellIDX2']==excell) & (goodPairData['PeakDist']<closeThresh) & (goodPairData['PeakDist']>0)].iloc[0]
        
        examples = {'close': closePair, 'far': farPair}
        return examples
        
        
    def buildPairData(self, pN, rel_thresh = 0.5):
        
        STA = SpatialTuningAnalysis(pN)
        tuningCurves = STA.tuning_curves
        SI = STA.SI
        TCreliability = STA.TCreliability
        
        #tuningCurves = pN.TrainingSaver['place_fields'].values[-1]
        #SI = pN.TrainingSaver['SI'].values[-1]
        weights = pN.pRNN.W.detach().numpy()
        distmat = np.zeros_like(weights)
        
        
        pairData = pd.DataFrame()
        
        #Speed up options: 1) below EV threshold, don't calculate distance
        #                  2) use symmetry to you don't have to double calculate
        for tcIDX1,tc1 in enumerate(tuningCurves.values()):
            for tcIDX2,tc2 in enumerate(tuningCurves.values()):
                if tcIDX1 >= tcIDX2:
                    continue
                    
                
                if (TCreliability[tcIDX1]>=rel_thresh) & (TCreliability[tcIDX2]>=rel_thresh):
                    (peakdist, peakloc, peakval, xcorr) = self.getRepresentationalDistance(tc1,tc2)
                else:
                    (peakdist, peakloc, peakval, xcorr) = np.nan,np.array([np.nan,np.nan]),np.nan,np.nan
                    
                distmat[tcIDX1,tcIDX2]=peakdist
                df1 = pd.DataFrame({
                    "PeakDist": [peakdist],
                    "PeakLoc": [peakloc],
                    "PeakVal": [peakval],
                    "XCorr": [xcorr],
                    "SI 1": SI[tcIDX1],
                    "SI 2": SI[tcIDX2],
                    "meanSI" : gmean([SI[tcIDX1],SI[tcIDX2]]),
                    "reliability1": TCreliability[tcIDX1],
                    "reliability2": TCreliability[tcIDX2],
                    "meanReliability": np.mean([TCreliability[tcIDX1],TCreliability[tcIDX2]]),
                    "TC 1": [tc1],
                    "TC 2": [tc2],
                    "cellIDX1": [tcIDX1],
                    "cellIDX2": [tcIDX2],
                    "weight" : weights[tcIDX1,tcIDX2]
                })
                
                df2 = pd.DataFrame({
                    "PeakDist": [peakdist],
                    "PeakLoc": [[-peakloc[0],-peakloc[1]]],
                    "PeakVal": [peakval],
                    "XCorr": [np.flip(xcorr)],
                    "SI 1": SI[tcIDX2],
                    "SI 2": SI[tcIDX1],
                    "meanSI" : gmean([SI[tcIDX2],SI[tcIDX1]]),
                    "reliability1": TCreliability[tcIDX2],
                    "reliability2": TCreliability[tcIDX1],
                    "meanReliability": np.mean([TCreliability[tcIDX2],TCreliability[tcIDX1]]),
                    "TC 1": [tc2],
                    "TC 2": [tc1],
                    "cellIDX1": [tcIDX2],
                    "cellIDX2": [tcIDX1],
                    "weight" : weights[tcIDX2,tcIDX1]
                })

                pairData = pd.concat([pairData, df1])
                pairData = pd.concat([pairData, df2])
                
        return pairData, distmat
        
    def getRepresentationalDistance(self, tc1,tc2):
        #NOTE TODO: un-hard-code value 15
        xcorr = correlate2d(tc1,tc2)
        peakval = np.max(xcorr)
        peakloc = np.unravel_index(np.argmax(xcorr, axis=None),
                                   xcorr.shape)
        peakloc = [peakloc[0]-15,peakloc[1]-15]

        #Euclidian
        #peakdist = np.sqrt(peakloc[0]**2 + peakloc[1]**2)
        #City block
        peakdist = np.abs(peakloc[0]) + np.abs(peakloc[1])

        return peakdist, peakloc, peakval, xcorr
    
    
    
    def RepConnectFigure(self, netname = None, savefolder = None, showPercent=False):
        
        plt.figure()
        plt.subplot(3,3,3)
        #self.SIDistPanel(self.pairData)
        self.SIEVPanel(self.pairData)
        
        plt.subplot(2,2,3)
        _ = self.RepDistanceConnectivityPanel(self.goodPairData)
        
        plt.subplot(6,6,1)
        self.TuningCurvePanel(self.examples['close']['TC 1'])
        plt.title('Cell 1')
        
        plt.subplot(6,6,2)
        self.TuningCurvePanel(self.examples['close']['TC 2'])
        plt.title('Cell 2')
        
        plt.subplot(6,6,3)
        self.TuningCurvePanel(self.examples['far']['TC 1'])
        plt.title('Cell 3')
        
        plt.subplot(4,4,5)
        self.XCorrPanel(self.examples['close'])
        plt.title(f"1-2 dist: {self.examples['close']['PeakDist']}")
        
        plt.subplot(4,4,6)
        self.XCorrPanel(self.examples['far'])  
        plt.title(f"2-3 dist: {self.examples['far']['PeakDist']}")
        #plt.xlabel(f"Dist: {self.examples['far']['PeakDist']}")
        
        plt.subplot(3,3,6)
        if showPercent:
            self.WeightPercentilePanel(self.weightdist)
            plt.ylabel(self.thtype+' Percentile')
        else:
            self.WeightPercentilePanel(self.weightdist_value)
            plt.ylabel('mean SI')
            
        plt.subplot(3,3,9)
        self.WeightDistCorrPanel(self.weightcorr)
        
        #plt.tight_layout()
        if netname is not None:
            saveFig(plt.gcf(),netname+'_RepConnectivity',savefolder,
                    filetype='pdf')
        plt.show()
        
        
        
    def SIDistPanel(self,pairData):
        sithresh = self.SIthresh
        if self.thtype == 'reliability':
            plt.hist(pairData['reliability1'])
        elif self.thtype == 'SI':
            plt.hist(pairData['SI 1'])
        plt.plot([sithresh,sithresh],plt.ylim(),'r--',label='threshold')
        plt.xlabel(self.thtype)
        plt.ylabel('# Cells')
        plt.yticks([])
        plt.legend()
        
    def SIEVPanel(self,pairData):
        sithresh = self.SIthresh
        plt.plot(pairData['SI 1'],pairData['reliability1'],'ko',markersize=2)
        if self.thtype == 'reliability':
            plt.plot(plt.xlim(),[sithresh,sithresh],'k--')
        elif self.thtype == 'SI':
            plt.hist(pairData['SI 1'])
            plt.plot([sithresh,sithresh],plt.ylim(),'k--')
        plt.xlabel('SI')
        plt.ylabel('EV_s')
        
    def RepDistanceConnectivityPanel(self,pairData,maxdist=10):
        meandf = pairData.groupby('PeakDist')['weight'].mean()

        #plt.plot(pairData['PeakDist'],pairData['weight'],'k.')
        pairData.boxplot(column='weight',by='PeakDist', 
                         grid = False,showfliers=False,
                        ax=plt.gca(),positions=meandf.index,
                        color='k')
        plt.title('')
        plt.suptitle('')
        meandf.plot(color='r',linewidth=2)
        plt.plot(plt.xlim(),[0,0],'k--')
        plt.ylim([-0.12,0.18])
        plt.xlim([-0.5,maxdist+0.5])
        plt.xlabel('Representational Distance')
        plt.ylabel('Weight')
        return meandf
        
    
    def TuningCurvePanel(self,tc):
        plt.imshow(tc.transpose(),
                   interpolation='nearest',alpha=self.PFmask.transpose())
        plt.axis('off')
        
    def XCorrPanel(self,singlePairData):
        plt.imshow(singlePairData['XCorr'].transpose(),
                  extent=(-15,15,-15,15))
        plt.plot([0,0],plt.ylim(),'r--',linewidth=1)
        plt.plot(singlePairData['PeakLoc'][0], 
                 -singlePairData['PeakLoc'][1], 
                 'r+')
        plt.plot(plt.ylim(),[0,0],'r--',linewidth=1)
        plt.axis('off')
        
    
    def AllWeightspanel(self, pairData):
        
        randoffset = 0.07*random.randn(*pairData['PeakDist'].shape)

        plt.scatter(pairData['PeakDist']+randoffset,pairData['weight'],
               c=np.log10(pairData[cmap]),s=1)
        plt.colorbar(label='log'+cmap)
        plt.xlabel('Distance')
        plt.ylabel('Weight')
        plt.xlim([-1,20.5])
        
        
        
    def calculateWeightByPercentile(self, pairData, metric = 'reliability',
                                    numquantiles=10, maxdist=11):
        if metric == 'reliability':
            idxdata = pairData['meanReliability']
        elif metric == 'SI':
            idxdata = pairData['meanSI']
        quantiles = np.linspace(0,1,numquantiles+1)
        weightdist = np.zeros((numquantiles,maxdist))
        for p in range(numquantiles):
            quantdata = pairData[idxdata.between(*idxdata.quantile([quantiles[p],quantiles[p+1]]).values,
                                                 inclusive='both')]

            meandf = quantdata.groupby('PeakDist')['weight'].mean()
            meandf = meandf.reindex(range(maxdist+1), fill_value= np.nan)
            weightdist[p,:] = meandf[:maxdist]
            
        return weightdist
    
    
    def calculateWeightByValue(self, pairData, metric = 'reliability',
                                    numquantiles=5, maxdist=13):
        if metric == 'reliability':
            idxdata = pairData['meanReliability']
            bounds = [0,1]
        elif metric == 'SI':
            idxdata = pairData['meanSI']
            bounds = [0,1]
            
        quantiles = np.linspace(bounds[0],bounds[1],numquantiles+1)
        quantiles[0]=-np.inf
        quantiles[-1]=np.inf
        weightdist = np.zeros((numquantiles,maxdist))
        for p in range(numquantiles):
            quantdata = pairData[idxdata.between(quantiles[p],quantiles[p+1],
                                                 inclusive='both')]

            meandf = quantdata.groupby('PeakDist')['weight'].mean()
            meandf = meandf.reindex(range(maxdist+1), fill_value= np.nan)
            weightdist[p,:] = meandf[:maxdist]
            
        return weightdist
        
    def WeightPercentilePanel(self,weightdist):
        plt.imshow(weightdist,vmin=-0.08, vmax = 0.08,
                  cmap='coolwarm', origin='lower')
        plt.colorbar(label='mean weight',location='right')
        plt.xlabel('Distance')
        plt.ylabel('Percentile')
        #plt.title('Next Step (LN)')
        
        
    def calculateWeightDistRelationship(self, pairData, numquantiles=5, metric = 'reliability', bounds = [0,1]):
        
        if metric == 'reliability':
            idxdata1 = pairData['reliability1']
            idxdata2 = pairData['reliability2']
        elif metric == 'SI':
            idxdata1 = pairData['SI 1']
            idxdata2 = pairData['SI 2']
            
        quantiles = np.linspace(bounds[0],bounds[1],numquantiles+1)
        quantiles[0]=-np.inf
        quantiles[-1]=np.inf
        weightdistcorr = np.zeros((numquantiles,numquantiles))
        for p1 in range(numquantiles):
            for p2 in range(numquantiles):
                inidx = (idxdata1.between(quantiles[p1],quantiles[p1+1], inclusive='both')) & (idxdata2.between(quantiles[p2],quantiles[p2+1], inclusive='both'))

                weightdistcorr[p1,p2] = pairData.PeakDist[inidx].corr(pairData.weight[inidx],method='spearman')
            
        return weightdistcorr, quantiles
    
    def WeightDistCorrPanel(self,weightdistcorr):
        plt.imshow(weightdistcorr[0],extent=(0,1,0,1), vmin=-0.6,vmax=0.6,
                  cmap='coolwarm', origin='lower')
        plt.xlabel('SI Cell 1')
        plt.ylabel('SI Cell 2')
        plt.colorbar(label='W-Dist Corr')