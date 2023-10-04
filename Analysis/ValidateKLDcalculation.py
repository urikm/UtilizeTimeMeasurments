# Specifically we focus on validation the affinity part as we check on the full trajectory



# Import
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utility.Params import GenRateMat
from PhysicalModels.TrajectoryCreation import EstimateTrajParams, CreateTrajectory
from PhysicalModels.PartialTrajectories import CreateCoarseGrainedTraj, CalcKLDPartialEntropyProdRate
from PhysicalModels.UtilityTraj import EntropyRateCalculation
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver

# Define the sweep
nIterations = 5
maxGenRate = 50
trajLength = 1e7
nDim = 4

# Init memory lists
savedRateMatrix = []
savedEstRateMatrix = []
savedKld = []
savedFullEpr = []
savedFullEprEst = []

nDumpedSystems = 0

# run over all systems, create trajectory and calculate the plugin and the KLD on the transform
for iIter in range(nIterations):
    if iIter % 50 == 0:
        if iIter > 0:
            print('The block executed in: ' + str(time.time() - tic) + '[s]')
        print('Starting iteration #'+str(iIter))
        tic = time.time()
    # Randomize rate matrix
    mW, nDim, vHiddenStates, timeRes = GenRateMat(nDim, maxGenRate)
    y = 20.
    eps = 0.5
    z = y / 5
    mW = np.array([[-90., 10., z, z],
                   [45., -20., y, y],
                   [22.5, 5., -(z + y + eps), eps],
                   [22.5, 5., eps, -(z + y + eps)]])

    mW = np.array([[-3, 1., 0., 1.],
                   [2., -2., 1., 0.],
                   [0., 1., -2., iIter + 0.1],
                   [1., 0., 1., -(iIter + 0.1 + 1)]])
    timeRes = 0.05

    vHiddenStates = []
    # Find the steady state
    vP0 = np.ones((nDim, )) / nDim  # np.array([0.25,0.25,0.25,0.25])
    n, vPi, mW, vWPn = MESolver(nDim, vP0, mW, timeRes)
    if n == 5e3 or abs(1 - vPi.sum()) > 1e-5:
        nDumpedSystems += 1
        continue
    initState = np.random.choice(nDim, 1, p=vPi).item()

    # Save rate matrix if valid system
    savedRateMatrix.append(mW)

    # Save Full EPR
    savedFullEpr.append(EntropyRateCalculation(nDim, mW, vPi))

    mCgTraj, mW = CreateTrajectory(nDim, int(trajLength), initState, mW)
    mCgTraj, _, vHiddenStates = CreateCoarseGrainedTraj(nDim, int(trajLength), mW, np.array([2, 3]), timeRes, semiCG=False,
                                                        remap=False)

    # Collect statistics on the estimated system ( adressing the issue of traj creation)
    mIndStates, mWaitTimes, vEstLambdas, mWest, vSimSteadyState = EstimateTrajParams(mCgTraj)
    savedEstRateMatrix.append(mWest)
    # n, vPiEst, mWest, vWPn = MESolver(nDim, vP0, mWest, timeRes)
    # initState = np.random.choice(nDim, 1, p=vPiEst).item()
    # savedFullEprEst.append(EntropyRateCalculation(nDim, mWest, vPiEst))

    vKldSemi, Tsemi2, vKldSemiAff, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vHiddenStates)
    savedKld.append(vKldSemi)


# %% Statistics no the results

# First convert all results to numpy array
mResults = pd.DataFrame(np.array([savedFullEpr, savedKld]).T,
                           columns=['FullEpr', 'KLD'])

# Count overestimation
mOverEst = np.expand_dims(mResults.values[:, 0], 1).repeat(len(mResults.values[0, 1:]), 1) - mResults.values[:, 1:]
vOverEstSys = (mOverEst < 0).sum(1)
print('Overestimated systems: ' + str((vOverEstSys > 0).sum()) + ' [' + str((vOverEstSys > 0).mean() * 100) + '%]')


# %% Visualize comparison
plt.hist((mResults.FullEpr - mResults.KLD) / mResults.FullEpr * 100, 10)
plt.xlabel('$(\sigma_{FULL}-\sigma_{KLD}) / \sigma_{FULL}$[%]')
plt.show()

plt.hist((mResults.FullEprEst - mResults.KLD) / mResults.FullEprEst * 100, 10)
plt.xlabel('$(\sigma_{FULL_{est}}-\sigma_{KLD}) / \sigma_{FULL_{est}}$[%]')
plt.show()

