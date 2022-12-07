# Randomize systems and look for those with Full-CG better than S-CG

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
import PhysicalModels.PartialTrajectories as pt
import PhysicalModels.UtilityTraj as ut
import Utility.FindPluginInfEPR as infEPR
import Utility.Params as pr

# %% Init parameters
nRuns = 50
fullTrajLength = int(1e7)
gamma = 1e-4
maxSeq = 9
nDim = 4
vHiddenStates = np.array([2,3])
timeRes = 0.01
dResults = {'mW':[], 'F-EPR':[], 'S-EPR':[], 'Full-EPR':[], 'tauFull':[], 'tauSemi':[], 'isSemiBetter': []}

for iSystem in range(nRuns):
    # Randomize system
    vP0 = np.array([0.25, 0.25, 0.25, 0.25])
    # mW, nDim, vHiddenStates, timeRes = pr.GenRateMat(nDim, 30)
    mW, nDim, vHiddenStates, timeRes = pr.HiddenControl(hid2to3=np.random.uniform(30, size=1)[0],
                                                     hid3to2=np.random.uniform(30, size=1)[0],
                                                     rate0to2=np.random.uniform(30, size=1)[0],
                                                     rate2to0=np.random.uniform(30, size=1)[0])
    # mW, nDim, vHiddenStates, timeRes = pr.HiddenControl(hid2to3=np.random.uniform(15 - 5, 15 + 5, size=1)[0],
    #                                                     hid3to2=np.random.uniform(55 - 5, 55 + 5, size=1)[0],
    #                                                     rate0to2=4,
    #                                                     rate2to0=20) #np.random.uniform(24-16,24+16, size=1)[0])

    n, vPn, mW, vWPn = MESolver(nDim, vP0, mW)
    if vPn.sum() > 0.999:
        vPn = vPn/vPn.sum()
    else:
        assert 0, "The Master equation solver doesnt converge for this system - you should look at it"
    # Create Coarse Grained trajectories
    mCgTrajF, nCgDimF, vHiddenStates = pt.CreateCoarseGrainedTraj(nDim, fullTrajLength, mW, vHiddenStates, timeRes, semiCG=False)
    mCgTrajS, nCgDimS, vHiddenStates2 = pt.CreateCoarseGrainedTraj(nDim, fullTrajLength, mW, vHiddenStates, timeRes, semiCG=True, remap=True)
    vStates = np.unique(mCgTrajS[:, 0])
    states2Omit = []
    # Calculate EPR
    try:
        EPRfcg, T, _, _ = pt.CalcKLDPartialEntropyProdRate(mCgTrajF, vHiddenStates)
        EPRscg, Tsemi, _, _  = pt.CalcKLDPartialEntropyProdRate(mCgTrajS, vHiddenStates2, states2Omit=states2Omit) #infEPR.EstimatePluginInf(mCgTrajS[:, 0], maxSeq=maxSeq, gamma=gamma)
        # Tsemi = np.mean(mCgTrajS[:, 1])
        # EPRscg = EPRscg/Tsemi  # Convert to  per time
        EPRful = ut.EntropyRateCalculation(nDim, mW, vPn)
    except:
        continue

    # Save relevant results
    if EPRfcg >= EPRscg:
        dResults['isSemiBetter'].append(False)
    else:
        dResults['isSemiBetter'].append(True)

    dResults['mW'].append(mW)
    dResults['F-EPR'].append(EPRfcg)  # Save as "per time"
    dResults['S-EPR'].append(EPRscg)  # Save as "per time"
    dResults['tauFull'].append(T)
    dResults['tauSemi'].append(Tsemi)
    dResults['Full-EPR'].append(EPRful)  # Save as "per time"

torch.save(dResults, 'dResults.pt')