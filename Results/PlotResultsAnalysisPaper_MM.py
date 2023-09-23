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

nRunsC = 10  # number of runs for collecting statistics of plugin/KLD
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




    # %% Plot
    # TODO : grid axes are flipped due to convention discrepancy with 2017 Gili's paper. need to first CalcW4DrivingForce, run all the estimators again and than we can get rid of the flipping od the axes
    resFig = plt.figure(0)

    # Making it look better by disconnecting branches
    for iMu in range(nMu - 1):
        vFull[nFlinterp * (iMu + 1)] = np.NAN
        vPluginInfSemi[nFl * (iMu + 1)] = np.NAN
        vKldSemi2[nFl * (iMu + 1)] = np.NAN
        vKld2[nFl * (iMu + 1)] = np.NAN
        vPluginInf[nFl*(iMu + 1)] = np.NAN


    # Plot full trajectory EPR
    plt.plot(vGridInterp, vFull, linestyle='-', color=(0.6350, 0.0780, 0.1840), label='$\sigma_{\mathrm{tot}}$')

    # Plot S-CG estimators
    plt.errorbar(vGridInterpCoarse, (vKldSemi2.mean(axis=1)), yerr=(vKldSemi2.std(axis=1)), fmt='-.', lw=0.5, color=(0, 0.4470, 0.7410), label='$\sigma_{\mathrm{KLD-SCG}}$')  # add to vKld -> vKld[:, 0]

    if addPlugin:
        plt.errorbar(vGridInterpCoarse, (vPluginInfSemi.mean(axis=1)), yerr=(vPluginInfSemi.std(axis=1)), fmt='-.', lw=0.5,
                     color=(0.8500, 0.3250, 0.0980), label='$\sigma_{\mathrm{plug}}$')


    # Plot FCG estimators
    plt.errorbar((vGridInterpCoarse), (vKld2.mean(axis=1)), yerr=(vKld2.std(axis=1)), fmt='-.', lw=0.5, color=(0.3010, 0.7450, 0.9330), label='$\sigma_{\mathrm{KLD-FCG}}$ ')  # add to vKld -> vKld[:, 0]

    if addPlugin:
        plt.errorbar(vGridInterpCoarse, (vPluginInf.mean(axis=1)), yerr=(vPluginInf.std(axis=1)), fmt='-.', lw=0.5,
                     color=(0.9290, 0.6940, 0.1250), label='$\sigma_{\mathrm{plug}}$')




    plt.yscale('log')
    plt.xlabel('F', fontsize='small')
    plt.ylabel('Entropy Production rate $[s^{-1}]$', fontsize='small')
    plt.tick_params(axis="both", labelsize=6)
    # plt.ylim(bottom=1e-4)
    # plt.xlim(right=-1.55, left=-2.2)
    plt.legend(prop={'size': 5})#, title='Semi-CG', title_fontsize='xx-small')

    plt.show()
    resFig.set_size_inches((2*3.38582677, 3.38582677))
    resFig.savefig(
        os.path.join(pathWoTimekld,
                     f'Plot_MM_Analysis.pdf'))







