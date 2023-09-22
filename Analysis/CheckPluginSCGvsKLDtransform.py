

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

# Define the sweep
nIterations = 1
maxGenRate = 50
trajLength = 1e7
nDim = 4
maxNumStates = 12  # for validty of systems

# Init memory lists
savedRateMatrix = []
savedKldTrns = []
savedPlgnScg = []
savedPlgnTrns = []
savedFullEpr = []
compatibleFlag = np.ones((nIterations,))
numStates = np.zeros((nIterations,))
nDumpedSystems = 0

# run over all systems, create trajectory and calculate the plugin and the KLD on the transform
# for iIter in range(nIterations):
iIter = 0
nIgnored = 0
while iIter < nIterations:
    if iIter % 50 == 0:
        if iIter > 0:
            print('The block executed in: ' + str(time.time() - tic) + '[s]')
        print('Starting iteration #'+str(iIter))
        tic = time.time()
    # Randomize rate matrix
    mW, nDim, vHiddenStates, timeRes = GenRateMat(nDim, maxGenRate)

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

    # Save SCG plugin
    mCgTraj, _, vHiddenStates = CreateCoarseGrainedTraj(nDim, int(trajLength), mW, vHiddenStates, timeRes, semiCG=True,
                                                        remap=False)

    savedPlgnScg.append(infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=0) / np.mean(mCgTraj[:, 1]))

    # Save SCG plugin and KLD after transform
    mCgTraj, nCgDim, vNewHidden, nHid = RemapStates(mCgTraj, 2)  # NOTE: assuming 4-states system!
    # Mark systems that are not compatible
    if vNewHidden.size + 2 > maxNumStates: # +2 because in these systems there are 2 markov states ; 12 is the thresh M <= log3(trajLen/10)
        compatibleFlag[iIter] = 0

    vKldSemi, Tsemi2, vKldSemiAff, _ = CalcKLDPartialEntropyProdRate(mCgTraj, vNewHidden)
    savedKldTrns.append(vKldSemi)
    savedPlgnTrns.append(infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=0) / np.mean(mCgTraj[:, 1]))

    if savedKldTrns[-1] <= savedFullEpr[-1]:
        savedFullEpr.pop()
        savedKldTrns.pop()
        savedPlgnScg.pop()
        savedPlgnTrns.pop()
        savedRateMatrix.pop()
        nIgnored += 1
    else:
        numStates[iIter] = vNewHidden.size + 2
        iIter += 1




# %% Statistics no the results

# First convert all results to numpy array
mResults = pd.DataFrame(np.array([savedFullEpr, savedPlgnScg, savedKldTrns, savedPlgnTrns]).T,
                           columns=['FullEpr',  'ScgPlgn', 'TrnsKLD', 'TrnsPlgn'])

# Count overestimation
mOverEst = np.expand_dims(mResults.values[:, 0], 1).repeat(len(mResults.values[0, 1:]), 1) - mResults.values[:, 1:]
vOverEstSys = (mOverEst < 0).sum(1)
print('Overestimated systems: ' + str((vOverEstSys > 0).sum()) + ' [' + str((vOverEstSys > 0).mean() * 100) + '%]')

vOverEstSys = (mOverEst < 0).sum(0)
print('Overestimated by PLGN-SCG: ' + str(vOverEstSys[0]) + ' [' + str(vOverEstSys[0] / mOverEst.shape[0] * 100) + '%]')
print('Overestimated by KLD-TRNS: ' + str(vOverEstSys[1]) + ' [' + str(vOverEstSys[1] / mOverEst.shape[0] * 100) + '%]')
print('Overestimated by PLGN-TRNS: ' + str(vOverEstSys[2]) + ' [' + str(vOverEstSys[2] / mOverEst.shape[0] * 100) + '%]')
print("\n")

# Count the maximum estimators
mResTmp = mResults.values[:, 1:]
print('KLD-transformed maximum estimator in ' + str((mResTmp.argmax(1) == 1).sum()) + ' times [' + str((mResTmp.argmax(1) == 1).mean() * 100) + '%]')
print('Plugin-SCG maximum estimator in ' + str((mResTmp.argmax(1) == 0).sum()) + ' times [' + str((mResTmp.argmax(1) == 0).mean() * 100) + '%]')
print('Plugin-transformed maximum estimator in ' + str((mResTmp.argmax(1) == 2).sum()) + ' times [' + str((mResTmp.argmax(1) == 2).mean() * 100) + '%]')
print("\n")
# Compare KLD transformed and Plugin SCG
print('KLD transformed is higher than Plugin-SCG count: ' + str((mResTmp[:, 0] < mResTmp[:, 1]).sum()) + ' [' + str((mResTmp[:, 0] < mResTmp[:, 1]).mean()  * 100) + '%]')


# %% Visualize comparison
plt.scatter(mResults.TrnsKLD, mResults.TrnsPlgn, 8)
plt.plot([0, max(mResults.TrnsKLD.max(), mResults.TrnsPlgn.max())], [0, max(mResults.TrnsKLD.max(), mResults.TrnsPlgn.max())], 'k')
plt.xlabel("TrnsKLD")
plt.ylabel("TrnsPLGN")
plt.show()

plt.scatter(mResults.TrnsKLD, mResults.ScgPlgn, 8)
plt.plot([0, max(mResults.TrnsKLD.max(), mResults.ScgPlgn.max())], [0, max(mResults.TrnsKLD.max(), mResults.ScgPlgn.max())], 'k')
plt.xlabel("TrnsKLD")
plt.ylabel("ScgPLGN")
plt.show()

plt.scatter(mResults.ScgPlgn, mResults.TrnsPlgn, 8)
plt.plot([0, max(mResults.ScgPlgn.max(), mResults.TrnsPlgn.max())], [0, max(mResults.ScgPlgn.max(), mResults.TrnsPlgn.max())], 'k')
plt.xlabel("ScgPLGN")
plt.ylabel("TrnsPLGN")
plt.show()

plt.scatter(mResults.FullEpr, mResults.TrnsKLD, 8)
plt.plot([0, max(mResults.FullEpr.max(), mResults.TrnsKLD.max())], [0, max(mResults.FullEpr.max(), mResults.TrnsKLD.max())], 'k')
plt.xlabel("FullEpr")
plt.ylabel("TrnsKLD")
plt.show()

plt.scatter(mResults.FullEpr, mResults.TrnsPlgn, 8)
plt.plot([0, max(mResults.FullEpr.max(), mResults.TrnsPlgn.max())], [0, max(mResults.FullEpr.max(), mResults.TrnsPlgn.max())], 'k')
plt.xlabel("FullEpr")
plt.ylabel("TrnsPLGN")
plt.show()

plt.scatter(mResults.FullEpr, mResults.ScgPlgn, 8)
plt.plot([0, max(mResults.FullEpr.max(), mResults.ScgPlgn.max())], [0, max(mResults.FullEpr.max(), mResults.ScgPlgn.max())], 'k')
plt.xlabel("FullEpr")
plt.ylabel("ScgPLGN")
plt.show()

