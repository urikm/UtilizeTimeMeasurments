import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utility.Params import GenRateMat
from PhysicalModels.PartialTrajectories import CreateCoarseGrainedTraj, CalcKLDPartialEntropyProdRate, RemapStates
from PhysicalModels.UtilityTraj import EntropyRateCalculation
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
import Utility.FindPluginInfEPR as infEPR


mW =np.array([[-60.04317146,  41.65500697,  28.34586306,  27.38919262], [ 13.26985049, -60.39559452,  39.777567,    30.26297999], [ 21.92144092,  13.5200609,  -84.96338939,  21.30186995], [ 24.85188006,   5.22052664,  16.83995933, -78.95404255]])
nDim = 4
timeRes = 0.001
trajLength = 1e7
vP0 = np.ones((nDim,)) / nDim  # np.array([0.25,0.25,0.25,0.25])

n, vPi, mW, vWPn = MESolver(nDim, vP0, mW, timeRes)
n == 5e3 or abs(1 - vPi.sum()) > 1e-5
initState = np.random.choice(nDim, 1, p=vPi).item()
vHiddenStates = np.array([2, 3])

mCgTraj, _, vHiddenStates = CreateCoarseGrainedTraj(nDim, int(trajLength), mW, vHiddenStates, timeRes, semiCG=True,
                                                    remap=False)
mCgTraj, nCgDim, vNewHidden, nHid = RemapStates(mCgTraj, 2)  # NOTE: assuming 4-states system!
vKldSemi, Tsemi2, vKldSemiAff, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vNewHidden)

# %% Check Probability splitting affect on KLD (log sum inequality)

forwardP = 0.09
backwardP = 0.91

vGridSplitsNum = np.arange(20)
randsPerPoint = 1000
mResCheck = np.zeros((randsPerPoint, vGridSplitsNum.size))
vTrueKLD = np.zeros((randsPerPoint,))  # represents the CG 2-states
for iRand in range(randsPerPoint):
    for iGrid in vGridSplitsNum:
        if iGrid == 0:
            mResCheck[iRand, iGrid] = (forwardP - backwardP) * np.log10(forwardP/backwardP)
        else:
            # Random split of the probabilty according to wanted states - forward and backward
            vPseudoRanges = np.random.uniform(0, 1 / (iGrid), iGrid) + (np.arange(iGrid) / (iGrid))
            vPseudoRanges = vPseudoRanges.tolist()
            vPseudoRanges.append(1)
            vSplitedRangesForward = np.diff(np.array(vPseudoRanges)) * forwardP

            vPseudoRangesB = np.random.uniform(0, 1 / (iGrid), iGrid) + (np.arange(iGrid) / (iGrid))
            vPseudoRangesB = vPseudoRangesB.tolist()
            vPseudoRangesB.append(1)
            vSplitedRangesBackward = np.diff(np.array(vPseudoRangesB)) * backwardP

            mResCheck[iRand, iGrid] = np.sum((vSplitedRangesForward - vSplitedRangesBackward) * np.log10(vSplitedRangesForward / vSplitedRangesBackward))
    # the mean KLD for every CG 2-states which yields the forwardP and backwardP
    iGrid = 1
    vPseudoRanges = np.random.uniform(0, 1 / (iGrid), iGrid) + (np.arange(iGrid) / (iGrid))
    vPseudoRanges = vPseudoRanges.tolist()
    vPseudoRanges.append(1)
    vSplitedRangesForward = np.diff(np.array(vPseudoRanges)) * forwardP

    vPseudoRangesB = np.random.uniform(0, 1 / (iGrid), iGrid) + (np.arange(iGrid) / (iGrid))
    vPseudoRangesB = vPseudoRangesB.tolist()
    vPseudoRangesB.append(1)
    vSplitedRangesBackward = np.diff(np.array(vPseudoRangesB)) * backwardP

    vTrueKLD[iRand] = np.sum(
        (vSplitedRangesForward - vSplitedRangesBackward) * np.log10(vSplitedRangesForward / vSplitedRangesBackward))

# Visualize
resFig = plt.figure(0)
plt.errorbar(vGridSplitsNum + 1, (mResCheck.mean(axis=0)), yerr=(mResCheck.std(axis=0)), fmt='-', lw=0.5, color=(0, 0.4470, 0.7410), label='$\mathrm{KLD}_{n}$')  # add to vKld -> vKld[:, 0]
#plt.errorbar(np.array([vGridSplitsNum.min(), vGridSplitsNum.max()]) + 1,  [vTrueKLD.mean(), vTrueKLD.mean()], yerr=[vTrueKLD.std(), vTrueKLD.std()], color=(0.9290, 0.6940, 0.1250), label='True 2-state KLD')
plt.plot(np.array([vGridSplitsNum.min(), vGridSplitsNum.max()]) + 1, [(forwardP - backwardP) * np.log10(forwardP/backwardP), (forwardP - backwardP) * np.log10(forwardP/backwardP)], 'k', label='KLD of CG state')
plt.xlabel('#Splits', fontsize='small')
plt.ylabel('$P_{\mathrm{forward}}\cdot log(P_{\mathrm{forward}}/P_{\mathrm{backward}})$', fontsize='small')
plt.legend(prop={'size': 5})#, title='Semi-CG', title_fontsize='xx-small')
plt.show()

