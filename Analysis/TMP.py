# Import
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utility.Params import GenRateMat
from PhysicalModels.PartialTrajectories import CreateCoarseGrainedTraj, CalcKLDPartialEntropyProdRate, RemapStates
from PhysicalModels.UtilityTraj import EntropyRateCalculation
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
import Utility.FindPluginInfEPR as infEPR
y = 20.
eps = 0.2
z = y / 4
mW = np.array([[-90.,   10.,   z,          z],
               [ 45.,  -20.,   y,          y],
               [ 22.5,  5.,   -(z+y+eps),  eps],
               [ 22.5,  5.,    eps,       -(z+y+eps)]])
nDim = 4
timeRes = 0.01
vHiddenStates = np.array([2, 3])
trajLength = 1e7


# Find the steady state
vP0 = np.ones((nDim,)) / nDim  # np.array([0.25,0.25,0.25,0.25])
n, vPi, mW, vWPn = MESolver(nDim, vP0, mW, timeRes)
initState = np.random.choice(nDim, 1, p=vPi).item()

fullEPR = EntropyRateCalculation(nDim, mW, vPi)

# Semi-CG case
mCgTraj, _, vNewHidden = CreateCoarseGrainedTraj(nDim, int(trajLength), mW, vHiddenStates, timeRes, semiCG=True,
                                                    remap=True)

vKldSemi, Tsemi2, vKldSemiAff, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vNewHidden)

# Full CG case
mCgTrajF, _, vHiddenStatesFCG = CreateCoarseGrainedTraj(nDim, int(trajLength), mW, vHiddenStates, timeRes, semiCG=False,
                                                    remap=False)

vKldSemiFCG, TsemiFCG, vKldSemiAffFCG, _ = CalcKLDPartialEntropyProdRate(mCgTrajF, vHiddenStatesFCG)