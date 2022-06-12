# Compare EPR bounds :  KLD, FCG, with time data  VS Plugin, SCG, without time data

import numpy as np
import matplotlib.pyplot as plt
import time

from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
import PhysicalModels.PartialTrajectories as pt
import PhysicalModels.UtilityTraj as ut
import Utility.FindPluginInfEPR as infEPR
from Utility.Params import ExtForcesGrid, BaseSystem, HiddenControl


# %% Comparison params
checkHidden = True
checkRing = False # relevant for checkHidden == False
maxSeq = 9
semiCGruns = 5
naiveTrajLen = int(1e7)
gamma = 1e-11

# %% Define system
if not checkHidden:
    mW, nDim, vHiddenStates, timeRes = BaseSystem()
    vGrid, _, _ = ExtForcesGrid('converege',interpRes=1e-3)
else:
    vGrid = np.array([0, 1, 2, 3, 5])  #np.array([10,30,50,150]) #np.array([0,10,22,30,50,60,100,150])  # np.array([10,20,22,30,40,50,60,70]) # not checkHidden: np.array([0, 0.005, 0.05, 0.5, 5])
vEPRfcg = np.zeros((len(vGrid),))
mEPRscg = np.zeros((len(vGrid), semiCGruns))
vEPRful = np.zeros((len(vGrid),))

# %% Calculate KLD for FCG with time data
for ix, x in enumerate(vGrid):
    if checkHidden:
        mWx, nDim, vHiddenStates, timeRes = HiddenControl(hid2to3=x, hid3to2=x, rate0to2=10)
    else:
        mWx = pt.CalcW4DrivingForce(mW, x)
    mCgTraj, nCgDim, vHiddenStates = pt.CreateCoarseGrainedTraj(nDim, naiveTrajLen, mWx, vHiddenStates, timeRes, semiCG=False)
    vEPRfcg[ix], T, _, _ = pt.CalcKLDPartialEntropyProdRate(mCgTraj, nCgDim, vHiddenStates)
    vEPRfcg[ix] = vEPRfcg[ix]*T
    # Calc full EPR
    vP0 = np.array([0.25, 0.25, 0.25, 0.25])
    n, vPn, mWx, _ = MESolver(nDim, vP0, mWx, 0.01)
    vPn = vPn/np.sum(vPn) # fro numerical stability
    vEPRful[ix] = ut.EntropyRateCalculation(nDim, mWx, vPn)

# %% Calculate Plugin for SCG without time data
for ix, x in enumerate(vGrid):
    if checkHidden:
        mWx, nDim, vHiddenStates, timeRes = HiddenControl(hid2to3=x, hid3to2=x, rate0to2=10)
    else:
        mWx = pt.CalcW4DrivingForce(mW, x)
    for iRun in range(semiCGruns):
        mCgTraj, nCgDim, vHiddenStates = pt.CreateCoarseGrainedTraj(nDim, naiveTrajLen, mWx, vHiddenStates, timeRes, semiCG=True)
        mEPRscg[ix, iRun] = infEPR.EstimatePluginInf(mCgTraj[:, 0], maxSeq=maxSeq, gamma=gamma)

# %% Plot
figT2 = plt.figure()
plt.plot(np.flip(-vGrid), np.flip(vEPRfcg), label='FullCGwTime')
plt.errorbar(np.flip(-vGrid), np.flip(mEPRscg.mean(1)), yerr=np.flip(mEPRscg.std(1)), label='SemiCGwoTime')
#plt.plot(np.flip(-vGrid), np.flip(vEPRful), label='FullEPR')
plt.title('Gilis-EPR')
plt.legend()
plt.show()
#figT2.set_size_inches((16, 16))
#figT2.savefig(f'CompareKLDnPLGIN.png')
