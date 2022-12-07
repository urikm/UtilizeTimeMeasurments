# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:46:37 2021

@title : Plot results for Flashing ratched model

@author: Uri Kapustin
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PhysicalModels.PartialTrajectories import  CalcKLDPartialEntropyProdRate
from PhysicalModels.UtilityTraj import EntropyRateCalculation
import PhysicalModels.ratchet as rt
import Utility.FindPluginInfEPR as infEPR


# %% UI
addPlugin = True
addFullCG = False
addSemiCG = True
nRunsC = 15  # number of runs for collecting statistics of plugin/KLD


# %% Init
vGrid = np.array([0.5, 1, 1.5, 2])  # Used potentials for RNEEP evaluation
vGridInterp = np.arange(0.5, 2 + 0.1, 0.1)

nLast = np.size(vGrid) - 1
nTimeStamps = int(1e7)

pathWoTime = 'FlashingRatchetSummary'  # 'Analysis_RNEEP_22_01_11'#'Analysis_RNEEP_collection'#'Analysis_RNEEP_21_08_09'# 'RNEEP_21_05_27' # Example, you should the wanted recording for plot

# %% Calculate analytic boundries
vFull = np.zeros(np.size(vGridInterp))
vPluginInf = np.zeros([np.size(vGridInterp), nRunsC])
vPluginInfSemi = np.zeros([np.size(vGridInterp), nRunsC])
vKld2 = np.zeros([np.size(vGridInterp), nRunsC])
vKldSemi2 = np.zeros([np.size(vGridInterp), nRunsC])
vTsemi = np.zeros(np.size(vGridInterp))

# %% Calculate  estimators
# Calculate full entropy rate
nDim = 6
for ix, x in enumerate(vGridInterp):
    vTau = np.zeros((nDim, ))
    # The full entropy rate
    mTraj, _, _ = rt.CreateNEEPTrajectory(nTimeStamps, x, isCG=False)
    for iS in range(nDim):
        vTau[iS] = mTraj[mTraj[:, 0] == iS, 1].mean()
    Tfactor = mTraj[:, 1].mean()
    vPi = (rt.p_ss(x) * vTau) / Tfactor
    tmp = EntropyRateCalculation(6, rt.rate_matrix(x), vPi) # Just for sanity comapre
    vFull[ix] = rt.ep_per_step(x) / Tfactor

# Calculate KLD and Plugin estimators
for ix, x in enumerate(vGridInterp):
    if addFullCG:
        for iRun in range(nRunsC):
            mCgTraj, _, vHiddenStatesF = rt.CreateNEEPTrajectory(nTimeStamps, x, fullCg=True)
            vKld2[ix, iRun], T, _, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vHiddenStatesF)
            if addPlugin:
                vPluginInf[ix, iRun] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=1e-3) / T

    # Add the SCG with times
    if addSemiCG:
        for iRun in range(nRunsC):
            mCgTraj, _, vHiddenStatesS = rt.CreateNEEPTrajectory(nTimeStamps, x, fullCg=False, remap=True)
            vStates = np.unique(mCgTraj[:, 0])
            vKldSemi2[ix, iRun], Tsemi2, _, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vHiddenStatesS)
            vKldSemi2[ix, iRun] = vKldSemi2[ix, iRun] #* Tsemi2
            if addPlugin:
                mCgTraj, _, vHiddenStatesS = rt.CreateNEEPTrajectory(nTimeStamps, x, fullCg=False)
                vPluginInfSemi[ix, iRun] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=1e-3) / Tsemi2
        vTsemi[ix] = Tsemi2

# %% Read results of RNEEP estimator
if addFullCG:
    mNeepMean = scipy.io.loadmat('..\Results\FlashingRatchetSummary\mMeansFull.mat')
    mNeepMean = mNeepMean['mMean']
    mNeepStd = scipy.io.loadmat('..\Results\FlashingRatchetSummary\mStdFull.mat')
    mNeepStd = mNeepStd['mStd']

if addSemiCG:
    mNeepMeanSemi = scipy.io.loadmat('..\Results\FlashingRatchetSummary\mMeansSemi.mat')
    mNeepMeanSemi = mNeepMeanSemi['mMeanSemi']
    mNeepStdSemi = scipy.io.loadmat('..\Results\FlashingRatchetSummary\mStdSemi.mat')
    mNeepStdSemi = mNeepStdSemi['mStdSemi']


# %% Plot
resFig = plt.figure(0)
plt.plot(vGridInterp, vFull, color=(0.6350, 0.0780, 0.1840), label='$\sigma_{tot}$')

plt.errorbar(vGridInterp, vKldSemi2.mean(axis=1), yerr=vKldSemi2.std(axis=1), fmt='-', lw= 0.5, color=(0, 0.4470, 0.7410), label='$\sigma_{KLD}$(Semi-CG)')  # add to vKld -> vKld[:, 0]
# plt.errorbar(vGridInterp, vKld2.mean(axis=1), yerr=vKld2.std(axis=1), fmt='.:', color=(0.3010, 0.7450, 0.9330), label='KLD')  # add to vKld -> vKld[:, 0]

if addPlugin:
    plt.errorbar(vGridInterp, vPluginInfSemi.mean(axis=1), yerr=np.flip(vPluginInfSemi.std(axis=1)), fmt='-', lw= 0.5,
                 color=(0.8500, 0.3250, 0.0980), label='$\sigma_{plug}$(Semi-CG)')
    # plt.errorbar(vGridInterp, vPluginInf.mean(axis=1), yerr=np.flip(vPluginInf.std(axis=1)), fmt=':',
    #              color=(0.9290, 0.6940, 0.1250), label='Plugin')

# plt.errorbar(vGrid, mNeepMean[0, :], yerr=mNeepStd[0, :], fmt='x', color=(0.4660, 0.6740, 0.1880), markersize=4, label='RNEEP(F)-seq2')
# plt.errorbar(vGrid, mNeepMean[1, :], yerr=mNeepStd[1, :], fmt='x', color=(0.3660, 0.5740, 0.1380), markersize=4, label='RNEEP(F)-seq3')
# plt.errorbar(vGrid, mNeepMean[2, :], yerr=mNeepStd[2, :], fmt='xr', label='RNEEP-seq6')
# plt.errorbar(vGrid, mNeepMean[3, :], yerr=mNeepStd[3, :], fmt='xy', label='RNEEP-seq11')
# plt.errorbar(vGrid, mNeepMean[4, :], yerr=mNeepStd[4, :], fmt='x', color=(0.2660, 0.4740, 0.0880), markersize=4, label='RNEEP(F)-seq12')

if addSemiCG:
    plt.errorbar(vGrid, mNeepMeanSemi[5, :] / vTsemi[[0,5,10,15]], yerr=mNeepStdSemi[5, :] / vTsemi[[0,5,10,15]], fmt='d',
                 color=(0.2940, 0.1140, 0.3560), markersize=5, label='$\sigma_{RNEEP,128}$(S-CG)')
    # plt.errorbar(vGrid, mNeepMeanSemi[4, :], yerr=mNeepStdSemi[4, :], fmt='x',
    #              color=(0.2940, 0.1140, 0.3560), markersize=4, label='$\sigma_{RNEEP,64}$(S-CG)')
    # plt.errorbar(vGrid, mNeepMeanSemi[3, :], yerr=mNeepStdSemi[3, :], fmt='Xy',
    #              label='SemiCG-seq11')
    # plt.errorbar(vGrid, mNeepMeanSemi[2, :], yerr=mNeepStdSemi[2, :], fmt='Xr',
    #              label='SemiCG-seq6')
    plt.errorbar(vGrid, mNeepMeanSemi[1, :] / vTsemi[[0,5,10,15]], yerr=mNeepStdSemi[1, :] / vTsemi[[0,5,10,15]], fmt='d',
                 color=(0.4940, 0.2840, 0.5560), markersize=4, label='$\sigma_{RNEEP,8}$(S-CG)')
    plt.errorbar(vGrid, mNeepMeanSemi[0, :] / vTsemi[[0,5,10,15]], yerr=mNeepStdSemi[0, :] / vTsemi[[0,5,10,15]], fmt='d',
                 color=(0.7940, 0.4840, 0.8560), markersize=3, label='$\sigma_{RNEEP,2}$(S-CG)')




# plt.yscale('log')
plt.xlabel('V', fontsize='small')
plt.ylabel('Entropy Production rate $[s^{-1}]$', fontsize='small')
plt.tick_params(axis="both", labelsize=6)
# plt.ylim(bottom=4e-3)
# plt.xlim(right=-1.55, left=-2.2)
plt.legend(prop={'size': 5})
plt.show()
resFig.set_size_inches((3.38582677, 3.38582677))
resFig.savefig(
    os.path.join(pathWoTime,
                 f'Plot_Analysis_FR.pdf'))




