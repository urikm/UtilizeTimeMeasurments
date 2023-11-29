# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:46:37 2021

@title : Plot results for Flashing ratched model

@author: Uri Kapustin
"""
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as matick
import numpy as np
import scipy.io
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
from PhysicalModels.PartialTrajectories import CalcKLDPartialEntropyProdRate
from PhysicalModels.UtilityTraj import EntropyRateCalculation
import PhysicalModels.ratchet as rt
import Utility.FindPluginInfEPR as infEPR



# %% UI
addPlugin = True
addFullCG = True
addSemiCG = False
nRunsC = 5#10  # number of runs for collecting statistics of plugin/KLD


# %% Init
vGrid = np.array([0.5, 1., 1.5, 2.])  # Used potentials for RNEEP evaluation
vGridInterp = np.arange(0.5, 2 + 0.1, 0.1)
vGridCoarse = np.arange(0.5, 2 + 0.3, 0.3)
nTimeStamps = int(1e7)

pathWoTime = 'FlashingRatchetSummary'  # 'Analysis_RNEEP_22_01_11'#'Analysis_RNEEP_collection'#'Analysis_RNEEP_21_08_09'# 'RNEEP_21_05_27' # Example, you should the wanted recording for plot

# %% Calculate analytic boundries
vFull = np.zeros(np.size(vGridInterp))
vPluginInf = np.zeros([np.size(vGridCoarse), nRunsC])
vPluginInfSemi = np.zeros([np.size(vGridCoarse), nRunsC])
vKld2 = np.zeros([np.size(vGridCoarse), nRunsC])
vKldSemi2 = np.zeros([np.size(vGridCoarse), nRunsC])
vTsemi = np.zeros(np.size(vGridCoarse))
vTfull = np.zeros(np.size(vGridCoarse))
# %% Calculate  estimators
# Calculate full entropy rate
nDim = 6
for ix, x in enumerate(vGridInterp):
    vTau = np.zeros((nDim, ))
    # The full entropy rate
    vP0 = np.random.uniform(size=(nDim))
    vP0 = vP0 / sum(vP0)
    n, vPi, mWx, vWPn = MESolver(nDim, vP0, rt.rate_matrix(x), 0.01)
    mTraj, _, _, _ = rt.CreateNEEPTrajectory(nTimeStamps=nTimeStamps, V=x, isCG=False)
    for iS in range(nDim):
        vTau[iS] = mTraj[mTraj[:, 0] == iS, 1].mean()
    Tfactor = mTraj[:, 1].mean()
    vPi = (rt.p_ss(x) * vTau) / Tfactor
    vFull[ix] = EntropyRateCalculation(6, rt.rate_matrix(x), vPi)  # Just for sanity comapre
    vFull[ix] = rt.ep_per_step(x) / Tfactor

# Calculate KLD and Plugin estimators
for ix, x in enumerate(vGridCoarse):
    if addFullCG:
        for iRun in range(nRunsC):
            mCgTraj, _, vHiddenStatesF, _ = rt.CreateNEEPTrajectory(nTimeStamps=nTimeStamps, V=x, fullCg=True)
            vKld2[ix, iRun], T, fcgAffKLD_DEBUG, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vHiddenStatesF)
            #vKld2[ix, iRun] *= T
            if addPlugin:
                vPluginInf[ix, iRun] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=0) / T
            vTfull[ix] = T
    # Add the SCG with times
    if addSemiCG:
        for iRun in range(nRunsC):
            mCgTraj, nCgDim, vHiddenStatesS, tauFactor = rt.CreateNEEPTrajectory(nTimeStamps=nTimeStamps, V=x, fullCg=False, isCG=True, remap=True)
            vStates = np.unique(mCgTraj[:, 0])
            vKldSemi2[ix, iRun], Tsemi2, affDebug, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vHiddenStatesS,states2Omit=[])
            #vKldSemi2[ix, iRun] *= Tsemi2
            #vKldSemi2[ix, iRun] *= (Tsemi2 / tauFactor) / (mCgTraj[:, 0].size / nTimeStamps)  # in case of reformulation - we want to use the calculated tau factor from the semi-CG as it is more accurate than the one from the remap - which will always be bigger
            if addPlugin:
                mCgTraj, _, vHiddenStatesS, tauFactor = rt.CreateNEEPTrajectory(nTimeStamps=nTimeStamps, V=x, fullCg=False, isCG=True, remap=False)
                vPluginInfSemi[ix, iRun] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=0) / np.mean(mCgTraj[:, 1])

        vTsemi[ix] = tauFactor

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

if not addFullCG: # Fulll-CG much lower than full
    plt.plot(vGridInterp, vFull, color=(0.6350, 0.0780, 0.1840), label='$\sigma_{\mathrm{tot}}$')

if addSemiCG:
    plt.errorbar(vGridCoarse, vKldSemi2.mean(axis=1), yerr=vKldSemi2.std(axis=1), fmt='-', lw= 0.5, color=(0, 0.4470, 0.7410), label='$\sigma_{\mathrm{KLD}}$')  # add to vKld -> vKld[:, 0]
if addFullCG:
    plt.errorbar(vGridCoarse, vKld2.mean(axis=1), yerr=vKld2.std(axis=1), fmt='-', color=(0.3010, 0.7450, 0.9330), label='$\sigma_{KLD}$ (full-CG)')  # add to vKld -> vKld[:, 0]

if addPlugin:
    if addSemiCG:
        plt.errorbar(vGridCoarse, vPluginInfSemi.mean(axis=1), yerr=np.flip(vPluginInfSemi.std(axis=1)), fmt='-', lw= 0.5,
                     color=(0.8500, 0.3250, 0.0980), label='$\sigma_{\mathrm{plug}}$')
    if addFullCG:
        plt.errorbar(vGridCoarse, vPluginInf.mean(axis=1), yerr=np.flip(vPluginInf.std(axis=1)), fmt=':',
                      color=(0.9290, 0.6940, 0.1250), label='$\sigma_{\mathrm{plug}}$  (full-CG)')

if addFullCG:
    plt.errorbar(vGrid, mNeepMean[5, :] / vTfull[[0,2,3,5]], yerr=mNeepStd[5, :] / vTfull[[0,2,3,5]], fmt='d', color=(0.1660, 0.3740, 0.0880), markersize=5, label='$\sigma_{RNEEP,128}$ (full-CG)')
    # plt.errorbar(vGrid, mNeepMean[1, :], yerr=mNeepStd[1, :], fmt='x', color=(0.3660, 0.5740, 0.1380), markersize=4, label='RNEEP(F)-seq8')
    # plt.errorbar(vGrid, mNeepMean[2, :], yerr=mNeepStd[2, :], fmt='xr', label='RNEEP-seq6')
    plt.errorbar(vGrid, mNeepMean[1, :] / vTfull[[0,2,3,5]], yerr=mNeepStd[1, :] / vTfull[[0,2,3,5]], fmt='d', color=(0.2660, 0.5740, 0.1380), label='$\sigma_{RNEEP,8}$ (full-CG)')
    plt.errorbar(vGrid, mNeepMean[0, :] / vTfull[[0,2,3,5]], yerr=mNeepStd[0, :] / vTfull[[0,2,3,5]], fmt='d', color=(0.6660, 0.7740, 0.3880), markersize=3, label='$\sigma_{RNEEP,2}$ (full-CG)')

if addSemiCG:
    plt.errorbar(vGrid, mNeepMeanSemi[5, :] / vTsemi[[0,2,3,5]], yerr=mNeepStdSemi[5, :] / vTsemi[[0,2,3,5]] , fmt='d',
                 color=(0.2940, 0.1140, 0.3560), markersize=5, label='$\sigma_{\mathrm{RNEEP,128}}$')
    # plt.errorbar(vGridInterp, mNeepMeanSemi[4, :], yerr=mNeepStdSemi[4, :], fmt='x',
    #              color=(0.2940, 0.1140, 0.3560), markersize=4, label='$\sigma_{RNEEP,64}$(S-CG)')
    # plt.errorbar(vGridInterp, mNeepMeanSemi[3, :], yerr=mNeepStdSemi[3, :], fmt='Xy',
    #              label='SemiCG-seq11')
    # plt.errorbar(vGridInterp, mNeepMeanSemi[2, :], yerr=mNeepStdSemi[2, :], fmt='Xr',
    #              label='SemiCG-seq6')
    plt.errorbar(vGrid, mNeepMeanSemi[1, :] / vTsemi[[0,2,3,5]] , yerr=mNeepStdSemi[1, :] / vTsemi[[0,2,3,5]] , fmt='d',
                 color=(0.4940, 0.2840, 0.5560), markersize=4, label='$\sigma_{\mathrm{RNEEP,8}}$')
    plt.errorbar(vGrid, mNeepMeanSemi[0, :] / vTsemi[[0,2,3,5]] , yerr=mNeepStdSemi[0, :] / vTsemi[[0,2,3,5]] , fmt='d',
                 color=(0.7940, 0.4840, 0.8560), markersize=3, label='$\sigma_{\mathrm{RNEEP,2}}$')

# TODO : change vTsemi to vTsemi[[0,5,10,15]]

#plt.yscale('log')
plt.xlabel('V', fontsize='small')
plt.ylabel('Entropy Production rate $[s^{-1}]$', fontsize='small')
plt.tick_params(axis="both", labelsize=6)
# plt.ylim(bottom=4e-3)
# plt.xlim(right=-1.55, left=-2.2)
plt.legend(prop={'size': 5})#, title='Semi-CG', title_fontsize='xx-small')

if addFullCG:
    aa = matick.ScalarFormatter(useOffset=True, useMathText=True)
    aa.set_scientific(True)
    aa.set_powerlimits([-4, -4])
    plt.gca().yaxis.set_major_formatter(aa)  # No decimal places

    resFig.set_size_inches((1*3.38582677,  3.38582677))
    resFig.savefig(
        os.path.join(pathWoTime,
                     f'Plot_Analysis_FR_fullCG.pdf'))
else:
    resFig.set_size_inches((2*3.38582677,  3.38582677))
    resFig.savefig(
        os.path.join(pathWoTime,
                     f'Plot_Analysis_FR.pdf'))


plt.show()


