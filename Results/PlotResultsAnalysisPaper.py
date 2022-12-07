# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:46:37 2021

@title : Plot results for base 4-states system

@author: Uri Kapustin
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid.inset_locator import inset_axes, InsetPosition, mark_inset
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
from PhysicalModels.TrajectoryCreation import EstimateTrajParams, CreateTrajectory
from PhysicalModels.PartialTrajectories import CalcPassivePartialEntropyProdRate, CalcInformedPartialEntropyProdRate, \
    CalcStallingData, CalcW4DrivingForce, CreateCoarseGrainedTraj, CalcKLDPartialEntropyProdRate
from PhysicalModels.UtilityTraj import CgRateMatrix, EntropyRateCalculation
import Utility.FindPluginInfEPR as infEPR
from Utility.Params import BaseSystem, ExtForcesGrid

# %% Init
mW, nDim, vHiddenStates, timeRes = BaseSystem()
vPiSt, xSt, r01, r10 = CalcStallingData(mW)
nCgDim = nDim - (np.size(vHiddenStates) - 1)
gridInterpRes = 5e-3

vPiStCg = np.zeros(nCgDim)
vPiStCg[0:2] = vPiSt[0:2]
vPiStCg[2] = vPiSt[2] + vPiSt[3]
mWCg = CgRateMatrix(mW, vHiddenStates)

# dbName = 'RneepDbCoarse'#'RneepDbCoarse'
# dbPath = '..'+os.sep+'StoredDataSets'+os.sep+dbName#'C:\\Uri\\MSc\\Thesis\\Datasets'+os.sep+dbName #'..'+os.sep+'StoredDataSets'+os.sep+dbName
# dbFileName = 'InitRateMatAsGilis'

# %% UI
addPlugin = True
addSemiCG = True
addGilisRes = False

pathWoTime = 'Analysis_RNEEP_paper_22_04_06'  #'Analysis_RNEEP_paper_22_03_07' # 'Analysis_RNEEP_22_01_11'#'Analysis_RNEEP_collection'#'Analysis_RNEEP_21_08_09'# 'RNEEP_21_05_27' # Example, you should the wanted recording for plot
pathWoTimekld = pathWoTime  # 'Analysis_RNEEP_21_08_09'

nRuns = 10  # number of runs for collecting statistics (from recorded data
nSeqs = 6#6  # number of different seq size input - 3,16,32,64,128
nRunsC = 5  # number of runs for collecting statistics of plugin/KLD

# -----------Grid----------------
vGrid, vGridInterp, subFolder = ExtForcesGrid('converege', interpRes=1e-2)
nLast = np.size(vGrid) - 1
subFolderkld = subFolder
# -------------------------------

# Define semi coarse grain results
subFolderSemi = 'AnalysisPaperSemi_'  # 'AnalysisFullSemi_'
vGridSemi, vGridInterpCoarse, _ = ExtForcesGrid('converege', interpRes=1e-1)
nRunsSemi = 10  # number of runs for collectiong statistics
nSeqsSemi = nSeqs  # number of different seq size input - 3,16,32,64,128
nLastSemi = np.size(vGridSemi) - 1

# %% Calculate analytic boundries
vFull = np.zeros(np.size(vGridInterp))
vAffinity = np.zeros(np.size(vGridInterp))
vPassive = np.zeros(np.size(vGridInterp))
vPluginInf = np.zeros([np.size(vGridInterpCoarse), nRunsC])
vPluginInfSemi = np.zeros([np.size(vGridInterpCoarse), nRunsC])
vTfactor = np.zeros(np.size(vGridInterp))
vKld2 = np.zeros([np.size(vGridInterpCoarse), nRunsC])
vKldSemi2 = np.zeros([np.size(vGridInterpCoarse), nRunsC])
# Calculate full entropy rate
i = 0
for x in vGridInterp:
    mWx = CalcW4DrivingForce(mW, x)  # Calculate W matrix after applying force
    # Passive partial entropy production rate
    vP0 = np.random.uniform(size=(nDim))
    vP0 = vP0 / sum(vP0)
    n, vPiX, mWx, vWPn = MESolver(nDim, vP0, mWx, timeRes)
    mWCgx = CgRateMatrix(mWx, vHiddenStates)
    vPiXCg = np.zeros(nCgDim)
    vPiXCg[0:2] = vPiX[0:2]
    vPiXCg[2] = vPiX[2] + vPiX[3]
    # The full entropy rate
    vFull[i] = EntropyRateCalculation(nDim, mWx, vPiX)
    vAffinity[i] = CalcInformedPartialEntropyProdRate(mWCgx, vPiXCg, vPiStCg)
    vPassive[i] = CalcPassivePartialEntropyProdRate(mWCgx, vPiXCg)
    # vKinetic[i] = CalcKineticBoundEntProdRate(mWCgx,vPiXCg)
    i += 1

# %% Calculate KLD and Plugin estimators

for ix, x in enumerate(vGridInterpCoarse):
    mWx = CalcW4DrivingForce(mW, x)
    for iRun in range(nRunsC):
        mCgTraj, nCgDim, vHiddenStatesF = CreateCoarseGrainedTraj(nDim, int(1e7), mWx, vHiddenStates, timeRes)
        vKld2[ix, iRun], T, _, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vHiddenStatesF)
        if addPlugin:
            vPluginInf[ix, iRun] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=1e-4) / T

    # Add the SCG with times
    if addSemiCG:
        for iRun in range(nRunsC):
            mCgTraj, nCgDim, vHiddenStatesS2 = CreateCoarseGrainedTraj(nDim, int(1e7), mWx, vHiddenStates, timeRes, semiCG=True, remap=True)
            vStates = np.unique(mCgTraj[:, 0])
            states2Omit = []  # np.array([1007, 1009])  # vStates[vStates > 1006]  #
            vKldSemi2[ix, iRun], Tsemi2, _, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vHiddenStatesS2,
                                                                    states2Omit=states2Omit)
            if addPlugin:
                mCgTraj, _, vHiddenStatesS = CreateCoarseGrainedTraj(nDim, int(1e7), mWx, vHiddenStates, timeRes, semiCG=True)
                vPluginInfSemi[ix, iRun] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=1e-4) / Tsemi2

# %% Read data
# NOTE : data is recorded after each 'x' - force, thus after each iteration data is incremented and not overriden so its sufficient to read the data for the last applied force
# RNEEP w/o time data
print("Plotting RNEEP results without time data")
specialPath = '_x_' + str(nLast, )
mNeepRaw = np.zeros([nSeqs, nLast + 1, nRuns])
mNeep = np.zeros([nSeqs, nLast + 1, nRuns])
vKld = np.ones([nLast + 1, nRuns])
for iRun in range(1 + 10, nRuns + 1 + 10):
    with open(pathWoTimekld + os.sep + subFolderkld + str(iRun) + os.sep + 'vKld' + specialPath + '.pickle',
              'rb') as handle:
        vKld[:, iRun - 1 - 10] = pickle.load(handle)
    with open(pathWoTime + os.sep + subFolder + str(iRun) + os.sep + 'mNeep' + specialPath + '.pickle', 'rb') as handle:
        mNeepRaw[:, :, iRun - 1 - 10] = pickle.load(handle)

# %% continue
for i in range(mNeep.shape[1]):
    mNeep[:, i, :] = mNeepRaw[:, i, :]
mNeepMean = np.mean(mNeep, axis=2)
mNeepStd = np.std(mNeep, axis=2)

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
    for iRun in range(1 + 10, nRunsSemi + 1 + 10):
        with open(pathWoTimekld + os.sep + subFolderSemi + str(iRun) + os.sep + 'vKld' + specialPath + '.pickle',
                  'rb') as handle:
            vKldSemi[:, iRun - 1 - 10] = pickle.load(handle)
        with open(pathWoTime + os.sep + subFolderSemi + str(iRun) + os.sep + 'mNeep' + specialPath + '.pickle',
                  'rb') as handle:
            mNeepRawSemi[:, :, iRun - 1 - 10] = pickle.load(handle)

    for i in range(mNeepSemi.shape[1]):
        mNeepSemi[:, i, :] = mNeepRawSemi[:, i, :]
    mNeepMeanSemi = np.mean(mNeepSemi, axis=2)
    mNeepStdSemi = np.std(mNeepSemi, axis=2)

# %% Plot
# TODO : grid axes are flipped due to convention discrepancy with 2017 Gili's paper. need to first CalcW4DrivingForce, run all the estimators again and than we can get rid of the flipping od the axes
#resFig = plt.figure(0)
resFig, ax1 = plt.subplots()
# Plot full trajectory EPR
ax1.plot(np.flip(-vGridInterp), np.flip(vFull), color=(0.6350, 0.0780, 0.1840), label='$\sigma_{tot}$')

# Plot S-CG estimators
ax1.errorbar(np.flip(-vGridInterpCoarse), np.flip(vKldSemi2.mean(axis=1)), yerr=np.flip(vKldSemi2.std(axis=1)), fmt='-', lw=0.5, color=(0, 0.4470, 0.7410), label='$\sigma_{KLD}$(Semi-CG)')  # add to vKld -> vKld[:, 0]

if addPlugin:
    ax1.errorbar(np.flip(-vGridInterpCoarse), np.flip(vPluginInfSemi.mean(axis=1)), yerr=np.flip(vPluginInfSemi.std(axis=1)), fmt='-', lw=0.5,
                 color=(0.8500, 0.3250, 0.0980), label='$\sigma_{plug}$(Semi-CG)')
if addSemiCG:
    ax1.errorbar(np.flip(-vGridSemi), np.flip(mNeepMeanSemi[5, :]), yerr=np.flip(mNeepStdSemi[5, :]), fmt='d',
                 color=(0.2940, 0.1140, 0.3560), markersize=5, label='$\sigma_{RNEEP,128}$(semi-CG)')
    # ax1.errorbar(np.flip(-vGridSemi), np.flip(mNeepMeanSemi[3, :]), yerr=np.flip(mNeepStdSemi[3, :]), fmt='Xy',
    #              label='SemiCG-seq11')
    # ax1.errorbar(np.flip(-vGridSemi), np.flip(mNeepMeanSemi[2, :]), yerr=np.flip(mNeepStdSemi[2, :]), fmt='Xr',
    #              label='SemiCG-seq6')
    ax1.errorbar(np.flip(-vGridSemi), np.flip(mNeepMeanSemi[1, :]), yerr=np.flip(mNeepStdSemi[1, :]), fmt='d',
                 color=(0.4940, 0.2840, 0.5560), markersize=4, label='$\sigma_{RNEEP,8}$(semi-CG)')
    ax1.errorbar(np.flip(-vGridSemi), np.flip(mNeepMeanSemi[0, :]), yerr=np.flip(mNeepStdSemi[0, :]), fmt='d',
                 color=(0.7940, 0.4840, 0.8560), markersize=2, label='$\sigma_{RNEEP,4}$(semi-CG)')


# Plot FCG estimators
ax1.errorbar(np.flip(-vGridInterpCoarse), np.flip(vKld2.mean(axis=1)), yerr=np.flip(vKld2.std(axis=1)), fmt='-', lw=0.5, color=(0.3010, 0.7450, 0.9330), label='$\sigma_{KLD}$(full-CG)')  # add to vKld -> vKld[:, 0]

if addPlugin:
    ax1.errorbar(np.flip(-vGridInterpCoarse), np.flip(vPluginInf.mean(axis=1)), yerr=np.flip(vPluginInf.std(axis=1)), fmt='-', lw=0.5,
                 color=(0.9290, 0.6940, 0.1250), label='$\sigma_{plug}$(full-CG)')

ax1.errorbar(np.flip(-vGrid), np.flip(mNeepMean[4, :]), yerr=np.flip(mNeepStd[5, :]), fmt='d', color=(0.1660, 0.3740, 0.0880), markersize=5, label='$\sigma_{RNEEP,128}$(full-CG)')
# ax1.errorbar(np.flip(-vGrid), np.flip(mNeepMean[3, :]), yerr=np.flip(mNeepStd[3, :]), fmt='d', label='RNEEP-seq11')
# ax1.errorbar(np.flip(-vGrid), np.flip(mNeepMean[2, :]), yerr=np.flip(mNeepStd[2, :]), fmt='d', label='RNEEP-seq6')
ax1.errorbar(np.flip(-vGrid), np.flip(mNeepMean[1, :]), yerr=np.flip(mNeepStd[1, :]), fmt='d', color=(0.2660, 0.5740, 0.1380), markersize=4, label='$\sigma_{RNEEP,8}$(full-CG)')
ax1.errorbar(np.flip(-vGrid), np.flip(mNeepMean[0, :]), yerr=np.flip(mNeepStd[0, :]), fmt='d', color=(0.6660, 0.7740, 0.3880), markersize=2, label='$\sigma_{RNEEP,4}$(full-CG)')

ax1.plot(np.flip(-vGridInterp), np.flip(vAffinity), '-', lw=0.5, color=(0.6350, 0.0780, 0.1840), label='$\sigma_{aff}$')

if addGilisRes:
    ax1.plot(gilisRes["Force.1"][:7], gilisRes.KLD_Exp[:7], ':c', label='KLDgili')

handles, labels = plt.gca().get_legend_handles_labels()
order = np.concatenate((np.delete(np.arange(len(labels)),1), np.array([1]))) # Handle a bug of legend plotting

ax1.set_yscale('log')
ax1.set_xlabel('x', fontsize='small')
ax1.set_ylabel('Entropy Production rate $[s^{-1}]$', fontsize='small')
ax1.tick_params(axis="both", labelsize=6)
ax1.set_ylim(bottom=1e-4, top=250)
ax1.set_xlim(right=0.8, left=-2.3)
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], prop={'size': 5},loc=4)

######################################
# Addd inset around the stalling force
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.07,0.07,0.35,0.33])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
#mark_inset(ax1, ax2, loc1=1, loc2=3, fc="none", ec='0.5')

# The data: only around st
threshInset = 1e-1
vMaskSt = vAffinity <= threshInset
vMaskStRNEEP = mNeepMean[4, :] <= threshInset # use the tighter bound as mask for RNEEP
vMaskStIntrp = vPluginInf.mean(axis=1) <= threshInset
if addPlugin:
    ax2.errorbar(np.flip(-vGridInterpCoarse[vMaskStIntrp]), np.flip(vPluginInf.mean(axis=1)[vMaskStIntrp]), yerr=np.flip(vPluginInf.std(axis=1)[vMaskStIntrp]), fmt='-', lw=0.5,
                 color=(0.9290, 0.6940, 0.1250), label='$\sigma_{plug}$(full-CG)')

ax2.errorbar(np.flip(-vGrid[vMaskStRNEEP]), np.flip(mNeepMean[4, vMaskStRNEEP]), yerr=np.flip(mNeepStd[5, vMaskStRNEEP]), fmt='d', color=(0.1660, 0.3740, 0.0880), markersize=5, label='$\sigma_{RNEEP,128}$(full-CG)')
#ax2.errorbar(np.flip(-vGrid[vMaskStRNEEP]), np.flip(mNeepMean[2, vMaskStRNEEP]), yerr=np.flip(mNeepStd[2, vMaskStRNEEP]), fmt='d', color=(0.2660, 0.5740, 0.1380), markersize=4, label='$\sigma_{RNEEP,6}$(full-CG)')
ax2.errorbar(np.flip(-vGrid[vMaskStRNEEP]), np.flip(mNeepMean[1, vMaskStRNEEP]), yerr=np.flip(mNeepStd[0, vMaskStRNEEP]), fmt='d', color=(0.6660, 0.7740, 0.3880), markersize=2, label='$\sigma_{RNEEP,4}$(full-CG)')


ax2.plot(np.flip(-vGridInterp[vMaskSt]), np.flip(vAffinity[vMaskSt]), '-', lw=0.5, color=(0.6350, 0.0780, 0.1840), label='$\sigma_{aff}$')
# Rneep around ST
#ax2.plot(T_E[T_E<=Tmax], CV_E[T_E<=Tmax], c='m', lw=2, alpha=0.5,
#         label='Einstein model')
handles, labels = plt.gca().get_legend_handles_labels()
orderInset = np.array([1,2,3,0])
ax2.legend([handles[idx] for idx in orderInset],[labels[idx] for idx in orderInset], prop={'size': 5},loc=3)
#ax2.legend(loc=3, prop={'size': 5})
ax2.set_yscale('log')
ax2.set_ylim(bottom=1e-8)
#ax2.set_xlabel('x', fontsize='small')
#ax2.set_ylabel('EPR $[s^{-1}]$', fontsize='small')
ax2.tick_params(axis="both", labelsize=5)
#ax2.set_yticks(np.arange(0,2,0.4))
#ax2.set_xticklabels(ax2.get_xticks(), backgroundcolor='w')
#ax2.tick_params(axis='x', which='major', pad=8)

plt.show()
resFig.set_size_inches((3.38582677*2, 3.38582677))
resFig.savefig(
    os.path.join(pathWoTimekld,
                 f'Plot_{subFolder}Analysis.pdf'))




