# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:46:37 2021

@title : plot results

@author: Uri Kapustin
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
from PhysicalModels.TrajectoryCreation import EstimateTrajParams, CreateTrajectory
from PhysicalModels.PartialTrajectories import CalcPassivePartialEntropyProdRate,CalcInformedPartialEntropyProdRate,CalcStallingData,CalcW4DrivingForce,CreateCoarseGrainedTraj
from PhysicalModels.UtilityTraj import CgRateMatrix,EntropyRateCalculation
import Utility.FindPluginInfEPR as infEPR
from Utility.Params import BaseSystem, ExtForcesGrid


# %% Init
mW, nDim, vHiddenStates, timeRes = BaseSystem()
vPiSt,xSt,r01,r10  = CalcStallingData(mW)
nCgDim = nDim - (np.size(vHiddenStates)-1)
gridInterpRes = 5e-3

vPiStCg = np.zeros(nCgDim)
vPiStCg[0:2] = vPiSt[0:2]
vPiStCg[2] = vPiSt[2]+vPiSt[3]
mWCg = CgRateMatrix(mW, vHiddenStates)

# dbName = 'RneepDbCoarse'#'RneepDbCoarse'
# dbPath = '..'+os.sep+'StoredDataSets'+os.sep+dbName#'C:\\Uri\\MSc\\Thesis\\Datasets'+os.sep+dbName #'..'+os.sep+'StoredDataSets'+os.sep+dbName
# dbFileName = 'InitRateMatAsGilis'

# %% UI
addPlugin = True
addSemiCG = True
addGilisRes = True

pathWoTime = 'Analysis_RNEEP_paper_22_03_07'#'Analysis_RNEEP_22_01_11'#'Analysis_RNEEP_collection'#'Analysis_RNEEP_21_08_09'# 'RNEEP_21_05_27' # Example, you should the wanted recording for plot
pathWoTimekld = pathWoTime#'Analysis_RNEEP_21_08_09'

nRuns = 10 # number of runs for collectiong statistics
nSeqs = 5 # number of different seq size input - 3,16,32,64,128

# -----------Grid---------------- 
vGrid, vGridInterp, subFolder = ExtForcesGrid('converege',interpRes=1e-3)
nLast = np.size(vGrid)-1
subFolderkld = subFolder
# -------------------------------

# Define semi coarse grain results
subFolderSemi = 'AnalysisPaperSemi_'#'AnalysisFullSemi_'
vGridSemi, _, _ = ExtForcesGrid('converege', interpRes=1e-3)
nRunsSemi = 10 # number of runs for collectiong statistics
nSeqsSemi = nSeqs # number of different seq size input - 3,16,32,64,128
nLastSemi = np.size(vGridSemi)-1

# %% Calculate analytic boundries
vFull = np.zeros(np.size(vGridInterp))
vInformed = np.zeros(np.size(vGridInterp))
vPassive = np.zeros(np.size(vGridInterp))
vPluginInf = np.zeros(np.size(vGrid))
vPluginInfSemi = np.zeros(np.size(vGridSemi))
vTfactor = np.zeros(np.size(vGrid))

# Calculate full entropy rate
i = 0
for x in vGridInterp: 
    mWx = CalcW4DrivingForce(mW, x) # Calculate W matrix after applying force
    # Passive partial entropy production rate
    vP0 = np.random.uniform(size=(nDim))
    vP0 = vP0/sum(vP0)
    n, vPiX, mWx, vWPn = MESolver(nDim, vP0, mWx, timeRes)
    mWCgx = CgRateMatrix(mWx, vHiddenStates)
    vPiXCg = np.zeros(nCgDim)
    vPiXCg[0:2] = vPiX[0:2]
    vPiXCg[2] = vPiX[2]+vPiX[3]
    # The full entropy rate
    vFull[i] = EntropyRateCalculation(nDim,mWx,vPiX)
    vInformed[i] = CalcInformedPartialEntropyProdRate(mWCgx,vPiXCg,vPiStCg)
    vPassive[i] = CalcPassivePartialEntropyProdRate(mWCgx,vPiXCg)
    # vKinetic[i] = CalcKineticBoundEntProdRate(mWCgx,vPiXCg)
    i+=1

if addPlugin:
    for ix, x in enumerate(vGrid):
        mWx = CalcW4DrivingForce(mW, x)
        mCgTraj, nCgDim, vHiddenStates = CreateCoarseGrainedTraj(nDim, int(2e7), mWx, vHiddenStates, timeRes)
        mIndStates, mWaitTimes, _, _, _ = EstimateTrajParams(mCgTraj)
        # Calculate common paramerters
        vTau = np.zeros(nCgDim)
        vTau[0] = np.sum(mWaitTimes[0]) / np.size(mWaitTimes[0], 0)
        vTau[1] = np.sum(mWaitTimes[1]) / np.size(mWaitTimes[1], 0)
        vTau[2] = np.sum(mWaitTimes[2]) / np.size(mWaitTimes[2], 0)


        vR = np.zeros(nCgDim)
        nTot = np.size(mIndStates[0], 0) + np.size(mIndStates[1], 0) + np.size(mIndStates[2], 0)
        vR[0] = np.size(mIndStates[0], 0) / nTot
        vR[1] = np.size(mIndStates[1], 0) / nTot
        vR[2] = np.size(mIndStates[2], 0) / nTot


        T = np.dot(vTau, vR)
        vTfactor[ix] = T
        vPluginInf[ix] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=1e-15)/T

    if addSemiCG:
        for ix, x in enumerate(vGridSemi):
            mWx = CalcW4DrivingForce(mW, x)
            mCgTraj, nCgDim, vHiddenStates = CreateCoarseGrainedTraj(nDim, int(1e7), mWx, vHiddenStates, timeRes, semiCG=True)
            mIndStates, mWaitTimes, _, _, _ = EstimateTrajParams(mCgTraj)
            # Calculate common paramerters
            vTau = np.zeros(nCgDim)
            vTau[0] = np.sum(mWaitTimes[0]) / np.size(mWaitTimes[0], 0)
            vTau[1] = np.sum(mWaitTimes[1]) / np.size(mWaitTimes[1], 0)
            vTau[2] = np.sum(mWaitTimes[2]) / np.size(mWaitTimes[2], 0)
        
            vR = np.zeros(nCgDim)
            nTot = np.size(mIndStates[0], 0) + np.size(mIndStates[1], 0) + np.size(mIndStates[2], 0)
            vR[0] = np.size(mIndStates[0], 0) / nTot
            vR[1] = np.size(mIndStates[1], 0) / nTot
            vR[2] = np.size(mIndStates[2], 0) / nTot

            Tsemi = np.dot(vTau, vR)
            vPluginInfSemi[ix] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=1e-15) / Tsemi


# %% Read data
# NOTE : data is recorded after each 'x' - force, thus after each iteration data is incremented and not overriden so its sufficient to read the data for the last applied force
# RNEEP w/o time data
print("Plotting RNEEP results without time data")
specialPath = '_x_'+str(nLast,)
mNeepRaw = np.zeros([nSeqs,nLast+1,nRuns])
mNeep = np.zeros([nSeqs,nLast+1,nRuns])
vKld = np.ones([nLast+1,nRuns])
for iRun in range(1+10,nRuns+1+10):
    with open(pathWoTimekld+os.sep+subFolderkld+str(iRun)+os.sep+'vKld'+specialPath+'.pickle', 'rb') as handle:
        vKld[:,iRun-1 - 10] = pickle.load(handle)
    with open(pathWoTime+os.sep+subFolder+str(iRun)+os.sep+'mNeep'+specialPath+'.pickle', 'rb') as handle:
        mNeepRaw[:,:,iRun-1 - 10] = pickle.load(handle)

# %% continue
for i in range(mNeep.shape[1]):
    mNeep[:,i,:] = mNeepRaw[:,i,:]#/vTfactor[i]
mNeepMean = np.mean(mNeep,axis=2)
mNeepStd = np.std(mNeep,axis=2)

# %% Add Gilis Results
if addGilisRes:
    gilisRes = pd.read_excel('2021-03-08Data.xls')
    vGiliGrid = gilisRes["Force.1"]
    vGiliMask = vGiliGrid >= np.min(-vGrid)
    vGiliMask &= vGiliGrid <= np.max(-vGrid)

# %% Add Semi Coarse Grain
if addSemiCG:
    print("Plotting RNEEP results without time data - Semi coarse grained")
    specialPath = '_x_' + str(nLastSemi, )
    mNeepRawSemi = np.zeros([nSeqsSemi, nLastSemi + 1, nRunsSemi])
    mNeepSemi = np.zeros([nSeqsSemi, nLastSemi + 1, nRunsSemi])
    vKldSemi = np.ones([nLastSemi + 1, nRunsSemi])
    for iRun in range(1+10, nRunsSemi + 1+10):
        with open(pathWoTimekld + os.sep + subFolderSemi + str(iRun) + os.sep + 'vKld' + specialPath + '.pickle',
                  'rb') as handle:
            vKldSemi[:, iRun - 1 - 10] = pickle.load(handle)
        with open(pathWoTime + os.sep + subFolderSemi + str(iRun) + os.sep + 'mNeep' + specialPath + '.pickle',
                  'rb') as handle:
            mNeepRawSemi[:, :, iRun - 1 - 10] = pickle.load(handle)

    for i in range(mNeepSemi.shape[1]):
        mNeepSemi[:,i,:] = mNeepRawSemi[:,i,:]#/vTfactor[i]
    mNeepMeanSemi = np.mean(mNeepSemi,axis=2)
    mNeepStdSemi = np.std(mNeepSemi,axis=2)

# %% Plot 
# TMP! TODO: delete
#vInformed[int(np.size(vGridInterp)/2)-1]=2*vPassive[int(np.size(vGridInterp)/2)-1]
# vInformed[5]=2*vPassive[5] # only for record from 19_06_21
#
resFig = plt.figure(0)
plt.plot(np.flip(-vGridInterp), np.flip(vFull), 'r', label='Full')
plt.plot(np.flip(-vGridInterp), np.flip(vInformed), ':k', label='Informed')
plt.plot(np.flip(-vGridInterp), np.flip(vPassive), ':g', label='Passive')

plt.plot(np.flip(-vGrid),np.flip(vKld[:,0]), 'o:b', label='KLD')

plt.errorbar(np.flip(-vGrid),np.flip(mNeepMean[0,:]),yerr=np.flip(mNeepStd[0,:]), fmt='xm', label='RNEEP-seq2')
plt.errorbar(np.flip(-vGrid),np.flip(mNeepMean[1,:]),yerr=np.flip(mNeepStd[1,:]), fmt='xc', label='RNEEP-seq3')
plt.errorbar(np.flip(-vGrid),np.flip(mNeepMean[2,:]),yerr=np.flip(mNeepStd[2,:]), fmt='xr', label='RNEEP-seq6')
plt.errorbar(np.flip(-vGrid),np.flip(mNeepMean[3,:]),yerr=np.flip(mNeepStd[3,:]), fmt='xy', label='RNEEP-seq11')
plt.errorbar(np.flip(-vGrid),np.flip(mNeepMean[4,:]),yerr=np.flip(mNeepStd[4,:]), fmt='xk', label='RNEEP-seq12')

if addSemiCG:
    plt.errorbar(np.flip(-vGridSemi), np.flip(mNeepMeanSemi[0, :]), yerr=np.flip(mNeepStdSemi[0, :]), fmt='om', label='semiCG-seq2')
    plt.errorbar(np.flip(-vGridSemi), np.flip(mNeepMeanSemi[1, :]), yerr=np.flip(mNeepStdSemi[1, :]), fmt='oc', label='semiCG-seq3')
    plt.errorbar(np.flip(-vGridSemi), np.flip(mNeepMeanSemi[2, :]), yerr=np.flip(mNeepStdSemi[2, :]), fmt='or', label='SemiCG-seq6')
    plt.errorbar(np.flip(-vGridSemi), np.flip(mNeepMeanSemi[3, :]), yerr=np.flip(mNeepStdSemi[3, :]), fmt='oy', label='SemiCG-seq11')
    plt.errorbar(np.flip(-vGridSemi), np.flip(mNeepMeanSemi[4, :]), yerr=np.flip(mNeepStdSemi[4, :]), fmt='ok', label='SemiCG-seq12')

if addGilisRes:
    plt.plot(gilisRes["Force.1"][:7], gilisRes.KLD_Exp[:7], ':c', label='KLDgili')

if addPlugin:
    plt.plot(np.flip(-vGrid), np.flip(vPluginInf), ':m', label='PluginKLD')
    plt.plot(np.flip(-vGridSemi), np.flip(vPluginInfSemi), '--m', label='PluginKLDSemi')

plt.yscale('log')
plt.xlabel('x - Driving Force')
plt.ylabel('Entropy Production rate[per time unit]')
plt.ylim(bottom=1e-3)
plt.legend(prop={'size': 6})
plt.show()
resFig.set_size_inches((8, 8))
resFig.savefig(
        os.path.join(pathWoTimekld,
                     f'Plot_{subFolder}Analysis.png'))


# # %% Plot by sequence
# for k in np.arange(len(vGrid)):
#     plt.figure(k+1)
#     if plotRneepFlag:
#         plt.plot(np.array([3,16,32,64,128]),(mNeep[:,k])) 
#     if plotRneepTFlag:
#         plt.plot(np.array([3,16,32,64,128]),(mNeepT[:,k]))   

#     plt.yscale('log')
#     plt.xlabel('input sequence size')
#     plt.ylabel('Entropy Production rate')
#     plt.title(['External force x-',str(vGrid[k])])
#     plt.show()
