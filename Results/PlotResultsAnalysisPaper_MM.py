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
from numba import njit
import pandas as pd
import PhysicalModels.PartialTrajectories as pt
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
from PhysicalModels.TrajectoryCreation import CreateTrajectory
from PhysicalModels.PartialTrajectories import CreateCoarseGrainedTraj, CalcKLDPartialEntropyProdRate
from PhysicalModels.UtilityTraj import EntropyRateCalculation
import Utility.FindPluginInfEPR as infEPR
from Utility.Params import MolecularMotor



# %% UI
addPlugin = True
addSemiCG = True

pathWoTimekld = ''

nRunsC = 1  # number of runs for collecting statistics of plugin/KLD
trajlength = 1e7

# -----------Grid----------------
res = 0.1
resInterp = 0.03
vMu = np.array([[1, 2, 3]])
vFl = np.expand_dims(np.arange(-1 + res / 2, 1 + res / 2, res), 1)
vFlinterp = np.expand_dims(np.arange(-1 + resInterp / 2, 1 + resInterp / 2, resInterp), 1)
#vFlinterp = vFlinterp[1:]  # TODO remove line
nMu = vMu.size
nFl = vFl.size
nFlinterp = vFlinterp.size

vGridInterpCoarse = np.reshape((vMu.repeat(nFl, 0) + vFl.repeat(nMu, 1)).T, -1)
vGridInterp = np.reshape((vMu.repeat(nFlinterp, 0) + vFlinterp.repeat(nMu, 1)).T, -1)

# Addition for the NEEP
vMuSemi = np.array([[1, 2, 3]])
resSemi = 0.5
vFlSemi = np.expand_dims(np.arange(-1 + resSemi / 2, 1 + resSemi / 2, resSemi), 1)
# vFlinterp = np.expand_dims(np.arange(-1 + resInterp / 2, 1 + resInterp / 2, resInterp), 1)
# vFlinterp = vFlinterp[1:]  # TODO remove line
nMuSemi = vMuSemi.size
nFlSemi = vFlSemi.size

vGridSemi = np.reshape((vMuSemi.repeat(nFlSemi, 0) + vFlSemi.repeat(nMuSemi, 1)).T, -1)


# %% Create Flashing Ratchet CG trajectory  ;   Note: deprecates use of simulation
def CreateMMTrajectory(mu, F, nTimeStamps, fullCg=False, isCG=True, remap=False):
    nTimeStamps = int(nTimeStamps)
    mW, nDim, vHiddenStates, timeRes = MolecularMotor(mu, F)
    vP0 = np.ones((nDim, ))/nDim  # np.array([0.25,0.25,0.25,0.25])
    n, vPi, mW, vWPn = MESolver(nDim, vP0, mW, timeRes)
    initState = np.random.choice(nDim, 1, p=vPi).item()
    tauFactorBeforeRemap = -1

    mTrajectory, _ = CreateTrajectory(nDim, nTimeStamps, initState, mW) # Create FR trajectory
    vHiddenStates = np.array([])
    if isCG:
        mTrajectory, tauFactorBeforeRemap, vHiddenStates, nDim, nCountHid = CoarseGrainMolecularMotor(mTrajectory, fullCg=fullCg, remap=remap)
        vHiddenStates = vHiddenStates[:nCountHid]
    return mTrajectory, nDim, vHiddenStates, tauFactorBeforeRemap

@njit
def CoarseGrainMolecularMotor(mCgTrajectory, fullCg=False, remap=False):
    # Note : the input is actually the full trajectory!
    mCgTrajectory[:, 0] = mCgTrajectory[:, 0] // 2 # Semi-CG

    if not fullCg: # in case of semi-CG we want to calculate Tau factor before remaping
        tauFactorBeforeRemap = mCgTrajectory[:, 1].mean()
    if fullCg:
        # prepare the trajectory -if semiCG make pseudo CG for than remapping. in FullCG it does al the CG
        trainBuffer = np.ones(mCgTrajectory[:, 0].size)
        for iStep in range(mCgTrajectory[:, 0].size):
        # now decimate train and test set
            # handle special case of first element
            if iStep == 0:
                continue
            # create mask for train set
            if mCgTrajectory[iStep-1, 0] == mCgTrajectory[iStep, 0]:
                trainBuffer[iStep-1] = 0
                mCgTrajectory[iStep, 1] += mCgTrajectory[iStep - 1, 1]

    if fullCg:
        mCgTrajectory = mCgTrajectory[trainBuffer == 1, :]
        tauFactorBeforeRemap = mCgTrajectory[:, 1].mean()

    vStates = np.unique(mCgTrajectory[:, 0])
    if remap and not fullCg:  # for each state we need to repeat the remap process
        nCountHid = 0
        vHiddenStates = np.zeros(1000, dtype=np.float64)
        for iHid in vStates:
            mCgTrajectory, nDim, vNewHidden, countAdded = pt.RemapStates(mCgTrajectory, iHid, baseState=int((iHid + 1) * 100))
            # Add the new vHiddenStates
            vHiddenStates[nCountHid:nCountHid+countAdded] = vNewHidden[:countAdded]
            nCountHid += countAdded
        nDim = np.unique(mCgTrajectory[:, 0]).size
    else:
        vHiddenStates = vStates
        nDim = vStates.size
        nCountHid = nDim

    return mCgTrajectory, tauFactorBeforeRemap, vHiddenStates, nDim, nCountHid



if __name__ == '__main__':
    # %% Calculate analytic boundries
    vFull = np.zeros(np.size(vGridInterp))

    vPluginInf = np.zeros([np.size(vGridInterpCoarse), nRunsC])
    vPluginInfSemi = np.zeros([np.size(vGridInterpCoarse), nRunsC])
    vKld2 = np.zeros([np.size(vGridInterpCoarse), nRunsC])
    vKldSemi2 = np.zeros([np.size(vGridInterpCoarse), nRunsC])

    # Calculate full entropy rate
    for ix, x in enumerate(vGridInterp):
        mu = vMu[0][ix // nFlinterp]
        F = x
        mWx, nDim, vHiddenStates, timeRes = MolecularMotor(mu, F)  # Calculate W matrix after applying force
        # Passive partial entropy production rate
        vP0 = np.random.uniform(size=(nDim))
        vP0 = vP0 / sum(vP0)
        n, vPiX, mWx, vWPn = MESolver(nDim, vP0, mWx, timeRes)

        # The full entropy rate
        #mTraj, nCgDim, vHiddenStatesF, _ = CreateMMTrajectory(mu, F, int(trajlength), fullCg=False, isCG=False, remap=False)  # TODO : TODEL
        vFull[ix] = EntropyRateCalculation(nDim, mWx, vPiX)# * mTraj[:, 1].mean()


    # %% Calculate KLD and Plugin estimators

    for ix, x in enumerate(vGridInterpCoarse):
        mu = vMu[0][ix // vFl.size]
        F = x
        for iRun in range(nRunsC):
            mCgTraj, nCgDim, vHiddenStatesF, _ = CreateMMTrajectory(mu, F, int(trajlength), fullCg=True, isCG=True, remap=False)
            vKld2[ix, iRun], T, _, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vHiddenStatesF)
            #vKld2[ix, iRun] = vKld2[ix, iRun] * T
            if addPlugin:
                vPluginInf[ix, iRun] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=0) / T

        # Add the SCG with times
        if addSemiCG:
            for iRun in range(nRunsC):
                mCgTraj, nCgDim, vHiddenStatesS2, tauFactor = CreateMMTrajectory(mu, F, int(trajlength), fullCg=False, isCG=True, remap=True)
                vStates = np.unique(mCgTraj[:, 0])
                states2Omit = [] # list(np.concatenate((np.arange(1011, 1025), np.arange(2011, 2025), np.arange(3011, 3025))))  # np.array([1007, 1009])  # vStates[vStates > 1006]  #
                vKldSemi2[ix, iRun], Tsemi2, _, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vHiddenStatesS2,
                                                                        states2Omit=states2Omit)
                #vKldSemi2[ix, iRun] *= (Tsemi2 )/ tauFactor  #) / (mCgTraj[:, 0].size / trajlength)   # in case of reformulation - we want to use the calculated tau factor from the semi-CG as it is more accurate than the one from the remap - which will always be bigger
                if addPlugin:
                    mCgTraj, _, vHiddenStatesS, tauFactor = CreateMMTrajectory(mu, F, int(trajlength), fullCg=False, isCG=True, remap=False)
                    vPluginInfSemi[ix, iRun] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=0) / np.mean(mCgTraj[:, 1])

    pathWoTime = 'Analysis_MM_23_11_28'
    subFolderSemi = 'AnalysisMM_'  # 'AnalysisFullSemi_'

    ## TODO : detele its for DEBUG
    nRunsSemi = 10  # number of runs for collectiong statistics
    nSeqsSemi = 3  # number of different seq size input - 3,16,32,64,128
    nLastSemi = np.size(vGridSemi) - 1
    if addSemiCG:
        print("Plotting RNEEP results without time data - Semi coarse grained")
        specialPath = '_x_' + str(nLastSemi, )
        mNeepRawSemi = np.zeros([nSeqsSemi, nLastSemi + 1, nRunsSemi])
        mNeepSemi = np.zeros([nSeqsSemi, nLastSemi + 1, nRunsSemi])
        vKldSemi = np.ones([nLastSemi + 1, nRunsSemi])
        for iRun in range(1, nRunsSemi + 1):
            with open(pathWoTime + os.sep + subFolderSemi + str(iRun) + os.sep + 'vKld' + specialPath + '.pickle',
                      'rb') as handle:
                vKldSemi[:, iRun - 1 - 10] = pickle.load(handle)
            with open(pathWoTime + os.sep + subFolderSemi + str(iRun) + os.sep + 'mNeep' + specialPath + '.pickle',
                      'rb') as handle:
                mNeepRawSemi[:, :, iRun - 1 - 10] = pickle.load(handle)

        for i in range(mNeepSemi.shape[1]):
            mNeepSemi[:, i, :] = mNeepRawSemi[:, i, :]
        mNeepMeanSemi = np.mean(mNeepSemi, axis=2)
        mNeepStdSemi = np.std(mNeepSemi, axis=2)

# %% FCG neep
    pathWoTimeF = 'Analysis_MM_23_09_28'
    subFolder = 'AnalysisMM_'  # 'AnalysisFullSemi_'
    print("Plotting RNEEP results without time data - Full coarse grained")
    mNeepRaw = np.zeros([nSeqsSemi, nLastSemi + 1, nRunsSemi])
    mNeep = np.zeros([nSeqsSemi, nLastSemi + 1, nRunsSemi])
    vKldFcg = np.ones([nLastSemi + 1, nRunsSemi])
    for iRun in range(1, nRunsSemi + 1):
        with open(pathWoTimeF + os.sep + subFolder + str(iRun) + os.sep + 'vKld' + specialPath + '.pickle',
                  'rb') as handle:
            vKldFcg[:, iRun - 1 - 10] = pickle.load(handle)
        with open(pathWoTimeF + os.sep + subFolder + str(iRun) + os.sep + 'mNeep' + specialPath + '.pickle',
                  'rb') as handle:
            mNeepRaw[:, :, iRun - 1 - 10] = pickle.load(handle)

    for i in range(mNeep.shape[1]):
        mNeep[:, i, :] = mNeepRaw[:, i, :]
    mNeepMean = np.mean(mNeep, axis=2)
    mNeepStd = np.std(mNeep, axis=2)

    # %% Plot
    # TODO : grid axes are flipped due to convention discrepancy with 2017 Gili's paper. need to first CalcW4DrivingForce, run all the estimators again and than we can get rid of the flipping od the axes
    resFig, ax1 = plt.subplots()

    # Save buffers
    vFull_Orig = vFull
    vPluginInfSemi_Orig = vPluginInfSemi
    vKldSemi2_Orig = vKldSemi2
    vKld2_Orig = vKld2
    vPluginInf_Orig = vPluginInf

    # Making it look better by disconnecting branches
    for iMu in range(nMu - 1):
        vFull[nFlinterp * (iMu + 1)] = np.NAN
        vPluginInfSemi[nFl * (iMu + 1)] = np.NAN
        vKldSemi2[nFl * (iMu + 1)] = np.NAN
        vKld2[nFl * (iMu + 1)] = np.NAN
        vPluginInf[nFl*(iMu + 1)] = np.NAN


    # Plot full trajectory EPR
    ax1.plot(vGridInterp, vFull, linestyle='-', color=(0.6350, 0.0780, 0.1840), label='$\sigma_{\mathrm{tot}}$')

    # Plot S-CG estimators
    ax1.errorbar(vGridInterpCoarse, (vKldSemi2.mean(axis=1)), yerr=(vKldSemi2.std(axis=1)), fmt='-.', lw=0.5, color=(0, 0.4470, 0.7410), label='$\sigma_{\mathrm{KLD}}$')  # add to vKld -> vKld[:, 0]

    if addPlugin:
        plt.errorbar(vGridInterpCoarse, (vPluginInfSemi.mean(axis=1)), yerr=(vPluginInfSemi.std(axis=1)), fmt='-.', lw=0.5,
                     color=(0.8500, 0.3250, 0.0980), label='$\sigma_{\mathrm{plug}}$')

    vFiltNeep = [1,2,5,6,9,10]
    if addSemiCG:
        ax1.errorbar(vGridSemi[vFiltNeep], (mNeepMeanSemi[nSeqsSemi - 1, vFiltNeep]),
                     yerr=(mNeepStdSemi[nSeqsSemi - 1, vFiltNeep] ), fmt='d',
                     color=(0.2940, 0.1140, 0.3560), markersize=5, label='$\sigma_{\mathrm{RNEEP,128}}$')
        ax1.errorbar(vGridSemi[vFiltNeep], (mNeepMeanSemi[2, vFiltNeep]), yerr=(mNeepStdSemi[2, vFiltNeep] ),
                     fmt='d',
                     color=(0.4940, 0.2840, 0.5560), markersize=4, label='$\sigma_{\mathrm{RNEEP,16}}$')
        ax1.errorbar(vGridSemi[vFiltNeep], (mNeepMeanSemi[1, vFiltNeep]), yerr=(mNeepStdSemi[1, vFiltNeep] ),
                     fmt='d',
                     color=(0.7940, 0.4840, 0.8560), markersize=2, label='$\sigma_{\mathrm{RNEEP,3}}$')


    # Plot FCG estimators
    ax1.errorbar((vGridInterpCoarse), (vKld2.mean(axis=1)), yerr=(vKld2.std(axis=1)), fmt='-.', lw=0.5, color=(0.3010, 0.7450, 0.9330), label='$\sigma_{\mathrm{KLD-FCG}}$ ')  # add to vKld -> vKld[:, 0]


    if addPlugin:
        ax1.errorbar(vGridInterpCoarse, (vPluginInf.mean(axis=1)), yerr=(vPluginInf.std(axis=1)), fmt='-.', lw=0.5,
                     color=(0.9290, 0.6940, 0.1250), label='$\sigma_{\mathrm{plug-FCG}}$')


    ax1.errorbar(vGridSemi[vFiltNeep], (mNeepMean[nSeqsSemi - 1, vFiltNeep]),
                 yerr=(mNeepStdSemi[nSeqsSemi - 1, vFiltNeep]), fmt='d',
                 color=(0.1660, 0.3740, 0.0880), markersize=5, label='$\sigma_{\mathrm{RNEEP,128}}$')
    ax1.errorbar(vGridSemi[vFiltNeep], (mNeepMean[2, vFiltNeep]), yerr=(mNeepStd[2, vFiltNeep]),
                 fmt='d',
                 color=(0.2660, 0.5740, 0.1380), markersize=4, label='$\sigma_{\mathrm{RNEEP,16}}$')
    ax1.errorbar(vGridSemi[vFiltNeep], (mNeepMean[1, vFiltNeep]), yerr=(mNeepStd[1, vFiltNeep]),
                 fmt='d',
                 color=(0.6660, 0.7740, 0.3880), markersize=2, label='$\sigma_{\mathrm{RNEEP,3}}$')


    ax1.set_yscale('log')
    newPos = ax1.get_position()
    newPos.x0 += 0.02
    ax1.set_position(newPos)
    ax1.set_xlabel('F', fontsize='small')
    ax1.set_ylabel('Entropy Production rate $[s^{-1}]$', fontsize='small')
    ax1.tick_params(axis="both", labelsize=6)
    # plt.ylim(bottom=1e-4)
    # plt.xlim(right=-1.55, left=-2.2)
    ax1.legend(prop={'size': 5})#, title='Semi-CG', title_fontsize='xx-small')

    # handle legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = np.arange(len(labels))  # Handle a bug of legend plotting
    aa = ax1.legend([handles[idx] for idx in order[[0]]],
                    [labels[idx] for idx in order[[0]]], prop={'size': 4.8}, loc=4, ncol=1,
                    title='Theoretical Bounds', title_fontsize='xx-small')
    ax1.add_artist(aa)
    ax1.legend([handles[idx] for idx in order[1:]], [labels[idx] for idx in order[1:]], prop={'size': 4.8}, loc=3,
               ncol=2, title='     Empirical Bounds\nSemi-CG           Full-CG', title_fontsize='xx-small')

    plt.show()
    resFig.set_size_inches((2*3.38582677, 3.38582677))
    resFig.savefig(
        os.path.join(pathWoTimekld,
                     f'Plot_MM_Analysis.pdf'))







