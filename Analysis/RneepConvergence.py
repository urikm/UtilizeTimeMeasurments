# -*- coding: utf-8 -*-
"""
Created on Fri Nov 2022

@title : Plot convergence of RNEEP as function of input sequence length

@author: Uri Kapustin
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import PhysicalModels.ratchet as rt
import Utility.FindPluginInfEPR as infEPR
from PhysicalModels.PartialTrajectories import CalcW4DrivingForce, CreateCoarseGrainedTraj, CalcKLDPartialEntropyProdRate
from Utility.Params import BaseSystem, ExtForcesGrid

# %% Define data to plot
pathsShortSeqs = 'Analysis_RNEEP_paper_22_03_07'
pathsLongSeqs = 'Analysis_RNEEP_paper_22_04_06'
vGrid, vGridInterp, subFolder = ExtForcesGrid('converege', interpRes=1e-1)
nLast = np.size(vGrid) - 1
nRuns = 10
subFolder = 'AnalysisPaperSemi_'
idxX2Plot = 10
trajLength = 1e7
# %% Calculate semi-CG KLD
mW, nDim, vHiddenStates, timeRes = BaseSystem()
mWx = CalcW4DrivingForce(mW, vGrid[idxX2Plot])
mCgTraj, nCgDim, vHiddenStatesS2 = CreateCoarseGrainedTraj(4, int(trajLength), mWx, vHiddenStates, timeRes, semiCG=True, remap=True)
vStates = np.unique(mCgTraj[:, 0])
states2Omit = []  # np.array([1007, 1009])  # vStates[vStates > 1006]  #
kldSemi, _, _, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vHiddenStatesS2, states2Omit=states2Omit)
mCgTraj, nCgDim, vHiddenStatesS2 = CreateCoarseGrainedTraj(4, int(trajLength), mWx, vHiddenStates, timeRes, semiCG=True)
T4state = np.mean(mCgTraj[:, 1])
pluginInf = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=0) / T4state

# %% Read data Shorts
nSeqs = 5
# NOTE : data is recorded after each 'x' - force, thus after each iteration data is incremented and not overriden so its sufficient to read the data for the last applied force
specialPath = '_x_' + str(nLast, )
mNeepRaw = np.zeros([nSeqs, nLast + 1, nRuns])
mNeep = np.zeros([nSeqs, nLast + 1, nRuns])
vKld = np.ones([nLast + 1, nRuns])
for iRun in range(1 + 10, nRuns + 1 + 10):
    with open('..' + os.sep + 'Results' + os.sep + pathsShortSeqs + os.sep + subFolder + str(iRun) + os.sep + 'vKld' + specialPath + '.pickle',
              'rb') as handle:
        vKld[:, iRun - 1 - 10] = pickle.load(handle)
    with open('..' + os.sep + 'Results' + os.sep + pathsShortSeqs + os.sep + subFolder + str(iRun) + os.sep + 'mNeep' + specialPath + '.pickle', 'rb') as handle:
        mNeepRaw[:, :, iRun - 1 - 10] = pickle.load(handle)

for i in range(mNeep.shape[1]):
    mNeep[:, i, :] = mNeepRaw[:, i, :]
mNeepMean = np.mean(mNeep, axis=2) / 1
mNeepStd = np.std(mNeep, axis=2) / 1

# %% Read data Longs
nSeqs = 6
specialPath = '_x_' + str(nLast, )
mNeepRawLongs = np.zeros([nSeqs, nLast + 1, nRuns])
mNeepLongs = np.zeros([nSeqs, nLast + 1, nRuns])
vKldLongs = np.ones([nLast + 1, nRuns])
for iRun in range(1 + 10, nRuns + 1 + 10):
    with open('..' + os.sep + 'Results' + os.sep + pathsLongSeqs + os.sep + subFolder + str(iRun) + os.sep + 'vKld' + specialPath + '.pickle',
              'rb') as handle:
        vKldLongs[:, iRun - 1 - 10] = pickle.load(handle)
    with open('..' + os.sep + 'Results' + os.sep + pathsLongSeqs + os.sep + subFolder + str(iRun) + os.sep + 'mNeep' + specialPath + '.pickle',
              'rb') as handle:
        mNeepRawLongs[:, :, iRun - 1 - 10] = pickle.load(handle)

for i in range(mNeepLongs.shape[1]):
    mNeepLongs[:, i, :] = mNeepRawLongs[:, i, :]
mNeepMeanLongs = np.mean(mNeepLongs, axis=2) / 1
mNeepStdLongs = np.std(mNeepLongs, axis=2) / 1

############################################################
# %% Load data for FR
# load RNEEP
idxX2PlotFR = 3
vGrid = np.array([0.5, 1, 1.5, 2])  # Used potentials for RNEEP evaluation
vSeqGrid = [2, 8, 16, 32, 64, 128]

# KLD and plugin
mCgTraj, _, vHiddenStatesS, _ = rt.CreateNEEPTrajectory(int(trajLength), vGrid[idxX2PlotFR], fullCg=False, remap=True)
vStates = np.unique(mCgTraj[:, 0])

kldSemiFR, _, _, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vHiddenStatesS)
mCgTraj, _, vHiddenStatesS, _ = rt.CreateNEEPTrajectory(int(trajLength), vGrid[idxX2PlotFR], fullCg=False)
Tfr = np.mean(mCgTraj[:, 1])
pluginInfFR = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=0) / Tfr

mNeepMeanSemi = scipy.io.loadmat('..\Results\FlashingRatchetSummary\mMeansSemi.mat')
mNeepMeanSemi = mNeepMeanSemi['mMeanSemi'] / Tfr
mNeepStdSemi = scipy.io.loadmat('..\Results\FlashingRatchetSummary\mStdSemi.mat')
mNeepStdSemi = mNeepStdSemi['mStdSemi'] / Tfr
############################################################

# %% Plot Convergences
reSfig, (ax1, ax2) = plt.subplots(1, 2)

# Plot 4-states
ax1.plot(np.array([2, 4, 3, 6, 8, 11, 12, 16, 32, 64, 128]), np.repeat(kldSemi, 11), '-',
                 color=(0, 0.4470, 0.7410), markersize=2, label='$\sigma_{\mathrm{KLD}}$')

ax1.plot(np.array([2, 4, 3, 6, 8, 11, 12, 16, 32, 64, 128]), np.repeat(pluginInf, 11), '-',
                 color=(0.8500, 0.3250, 0.0980), markersize=2, label='$\sigma_{\mathrm{plug}}$')

#ax1.errorbar(np.array([2, 3, 6, 11, 12]), mNeepMean[:, idxX2Plot], yerr=mNeepStd[:, idxX2Plot], fmt='-x',
#                 color=(0.7940, 0.4840, 0.8560), markersize=4, label='$\sigma_{RNEEP,m}$ (semi-CG)')
ax1.errorbar(np.array([4, 8, 16, 32, 64, 128]), mNeepMeanLongs[:, idxX2Plot], yerr=mNeepStdLongs[:, idxX2Plot], fmt='-x',
                 color=(0.7940, 0.4840, 0.8560), markersize=4, label='$\sigma_{\mathrm{RNEEP,}m}$')

ax1.set_xlabel('$m$', fontsize='small')
ax1.set_ylabel('Entropy Production rate $[s^{-1}]$', fontsize='small')
ax1.tick_params(axis="both", labelsize=6)
ax1.legend(prop={'size': 6}, title_fontsize='xx-small', loc=(0.695, 0.5))

ax1.set_title('4-states')

# plot FR
ax2.plot(np.array(vSeqGrid), np.repeat(kldSemiFR, 6), '-',
                 color=(0, 0.4470, 0.7410), markersize=2, label='$\sigma_{\mathrm{KLD}}$')

ax2.plot(np.array(vSeqGrid), np.repeat(pluginInfFR, 6), '-',
                 color=(0.8500, 0.3250, 0.0980), markersize=2, label='$\sigma_{\mathrm{plug}}$')

ax2.errorbar(np.array(vSeqGrid), mNeepMeanSemi[:, idxX2PlotFR], yerr=mNeepStdSemi[:, idxX2PlotFR], fmt='-x',
                 color=(0.7940, 0.4840, 0.8560), markersize=4, label='$\sigma_{\mathrm{RNEEP,}m}$')

ax2.set_xlabel('$m$', fontsize='small')
#ax2.set_ylabel('$\sigma [s^{-1}]$', fontsize='small')
ax2.tick_params(axis="both", labelsize=6)
ax2.legend(prop={'size': 6}, title_fontsize='xx-small', loc=(0.695, 0.5))

ax2.set_title('Flashing Ratchet')
#plt.tick_params(axis="both", labelsize=6)
plt.show()
reSfig.set_size_inches((3.38582677*2, 3.38582677))
reSfig.savefig(f'PlotRneepConvergence.pdf')




