# Compare EPR bounds :  KLD, FCG, with time data  VS Plugin, SCG, without time data

import numpy as np
import matplotlib.pyplot as plt
import time

from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
import PhysicalModels.TrajectoryCreation as ct
import PhysicalModels.PartialTrajectories as pt
import PhysicalModels.UtilityTraj as ut
import Utility.FindPluginInfEPR as infEPR
from Utility.Params import ExtForcesGrid, BaseSystem, HiddenControl


# %% Comparison params
checkHidden = True
checkRing = False # relevant for checkHidden == False
maxSeq = 9
threshold = 0
naiveTrajLen = int(1e7)
gamma = 1e-16

# %% Define system
if not checkHidden:
    mW, nDim, vHiddenStates, timeRes = BaseSystem()
    vGrid, _, _ = ExtForcesGrid('converege',interpRes=1e-3)
else:
    vGrid = np.array([10,20,25,30,50,60])  #np.array([0, 0.005, 0.05, 0.5, 5])
vEPRfcg = np.zeros((len(vGrid),))
vEPRscg = np.zeros((len(vGrid),))
vEPRful = np.zeros((len(vGrid),))

# %% Calculate KLD for FCG with time data
for ix, x in enumerate(vGrid):
    if checkHidden:
        mWx, nDim, vHiddenStates, timeRes = HiddenControl(hidBond=0, rate0to2=x)
    else:
        mWx = pt.CalcW4DrivingForce(mW, x)
    vP0 = np.array([0.25, 0.25, 0.25, 0.25])
    n, vPn, mWx, _ = MESolver(nDim, vP0, mWx, 0.01)
    vPn = vPn/np.sum(vPn) # fro numerical stability
    initState = np.random.choice(nDim, 1, p=vPn).item()
    mTrajectory, mW = ct.CreateTrajectory(nDim, naiveTrajLen, initState, mWx)  # Run Create Trajectory
    # Calc Full CG EPR estimator( KLD method) with time data
    mCgTrajF, nCgDim, _, _ = pt.CoarseGrainTrajectory(mTrajectory, nDim, vHiddenStates, semiCG=False)
    vEPRfcg[ix], Tf, _, _ = pt.CalcKLDPartialEntropyProdRate(mCgTrajF, nCgDim, vHiddenStates)
    vEPRfcg[ix] = vEPRfcg[ix]*Tf
    # vEPRfcg[ix] = infEPR.EstimatePluginInf(mCgTrajF[:, 0], maxSeq=maxSeq, gamma=gamma)
    # Calc Semi CG EPR estimator( Plugin method) without time data
    mCgTrajS, nCgDim, _, _ = pt.CoarseGrainTrajectory(mTrajectory, nDim, vHiddenStates, semiCG=True)
    # vEPRscg[ix] = infEPR.EstimatePluginInf(mCgTrajS[:, 0], maxSeq=maxSeq, gamma=gamma)
    vEPRscg[ix], Ts, _, _ = pt.CalcKLDPartialEntropyProdRate(mCgTrajS, nCgDim, vHiddenStates)
    vEPRscg[ix] = vEPRscg[ix]*Ts
    # Calc full EPR
    vEPRful[ix] = ut.EntropyRateCalculation(nDim, mWx, vPn)*Tf

# %% Plot
figT2 = plt.figure()
plt.plot(np.flip(-vGrid), np.flip(vEPRfcg), label='FullCGwTime')
plt.plot(np.flip(-vGrid), np.flip(vEPRscg), label='SemiCGwoTime')
plt.plot(np.flip(-vGrid), np.flip(vEPRful), label='FullEPR')
plt.title('Gilis-EPR')
plt.legend()
plt.show()
