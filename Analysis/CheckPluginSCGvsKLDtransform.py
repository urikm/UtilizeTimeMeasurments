

# Import
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.utils.data as dat
from torch.optim import Adam

#from Utility.Params import GenRateMat
from PhysicalModels.PartialTrajectories import CreateCoarseGrainedTraj, CalcKLDPartialEntropyProdRate, RemapStates
from PhysicalModels.UtilityTraj import EntropyRateCalculation
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
import Utility.FindPluginInfEPR as infEPR
import LearningModels.Neep as neep
from Dataset import CGTrajectoryDataSet

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def GenRateMat(nDim, limit):
	mW = np.random.uniform(limit, size=(nDim, nDim))
	for k in range(nDim):
		mW[k, k] = mW[k, k] - np.sum(mW[:, k])
	vHiddenStates = np.array([2, 3])
	timeRes = 0.01
	return mW, nDim, vHiddenStates, timeRes
# Define the sweep
nIterations = 50
maxGenRate = 50
trajLength = 1e7
nDim = 4
maxNumStates = 12  # for validty of systems

# Init memory lists
savedRateMatrix = []
savedKldTrns = []
savedPlgnScg = []
savedNeepScg = []
savedFullEpr = []
compatibleFlag = np.ones((nIterations,))
numStates = np.zeros((nIterations,))
nDumpedSystems = 0

# %% run over all systems, create trajectory and calculate the plugin and the KLD on the transform
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
    #savedPlgnTrns.append(infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=0) / np.mean(mCgTraj[:, 1]))

    # Save NEEP
    seqSize = 128
    # Create Datasets - special mode that generate 4-states
    trainDataSet = CGTrajectoryDataSet(seqLen=seqSize, batchSize=20, lenTrajFull=trajLength,
                                       extForce=-1e4, rootDir='StoredDataSetsCompare', semiCG=True)
    T = trainDataSet.timeFactor

    k = 0
    trainLoader = torch.utils.data.DataLoader(trainDataSet)

    model = neep.RNEEP()
    outFileadd = ''
    # if device == 'cuda:0':
    #     model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
    model.to(device)
    # defining the optimizer
    # optimizeurikr = SGD(model.parameters(),lr=vLrate[k])
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    trainRnn = neep.make_trainNoValid(model, optimizer, seqSize, device)
    bestLoss = 1e3
    bestEp = -1e4

    # Define sampler - train and validation
    for epoch in range(int(10)):
        tic = time.time()
        bestLossEpoch, epRate, bestEpErr = trainRnn(trainLoader, epoch)
        toc = time.time()
        if bestLossEpoch < bestLoss:
            bestEp = epRate / T
            bestLoss = bestLossEpoch
        print('Elapsed time of Epoch ' + str(epoch + 1) + ' is: ' + str(toc - tic) + " ; KLD est: " + str(
            bestEp))
    savedNeepScg.append(bestEp)

if savedKldTrns[-1] <= savedFullEpr[-1]:
    savedFullEpr.pop()
    savedKldTrns.pop()
    savedPlgnScg.pop()
    savedNeepScg.pop()
    savedRateMatrix.pop()
    nIgnored += 1
else:
    numStates[iIter] = vNewHidden.size + 2
    iIter += 1




# %% Statistics no the results

# First convert all results to numpy array
mResults = pd.DataFrame(np.array([savedFullEpr, savedPlgnScg, savedKldTrns, savedNeepScg]).T,
                           columns=['FullEpr',  'ScgPlgn', 'TrnsKLD', 'NeepScg'])
mResults.to_csv('StatisticalCompare.csv')

# %% Un-comment if you want to read saved data
mResults = pd.read_csv("StatisticalCompare.csv")

# %% Count overestimation
mOverEst = np.expand_dims(mResults.values[:, 1], 1).repeat(len(mResults.values[0, 2:]), 1) - mResults.values[:, 2:]
vOverEstSys = (mOverEst < 0).sum(1)
print('Overestimated systems: ' + str((vOverEstSys > 0).sum()) + ' [' + str((vOverEstSys > 0).mean() * 100) + '%]')

vOverEstSys = (mOverEst < 0).sum(0)
print('Overestimated by PLGN-SCG: ' + str(vOverEstSys[0]) + ' [' + str(vOverEstSys[0] / mOverEst.shape[0] * 100) + '%]')
print('Overestimated by KLD-TRNS: ' + str(vOverEstSys[1]) + ' [' + str(vOverEstSys[1] / mOverEst.shape[0] * 100) + '%]')
print('Overestimated by NEEP-SCG: ' + str(vOverEstSys[2]) + ' [' + str(vOverEstSys[2] / mOverEst.shape[0] * 100) + '%]')
print("\n")

# Count the maximum estimators
mResTmp = mResults.values[:, 2:]
print('KLD-transformed maximum estimator in ' + str((mResTmp.argmax(1) == 1).sum()) + ' times [' + str((mResTmp.argmax(1) == 1).mean() * 100) + '%]')
print('Plugin-SCG maximum estimator in ' + str((mResTmp.argmax(1) == 0).sum()) + ' times [' + str((mResTmp.argmax(1) == 0).mean() * 100) + '%]')
print('NEEP-SCG maximum estimator in ' + str((mResTmp.argmax(1) == 2).sum()) + ' times [' + str((mResTmp.argmax(1) == 2).mean() * 100) + '%]')
print("\n")
# Compare KLD transformed and Plugin SCG
print('KLD transformed is higher than Plugin-SCG count: ' + str((mResTmp[:, 0] < mResTmp[:, 1]).sum()) + ' [' + str((mResTmp[:, 0] < mResTmp[:, 1]).mean()  * 100) + '%]')


# Visualize comparison
# plt.scatter(mResults.TrnsKLD, mResults.NeepScg, 8)
# plt.plot([0, max(mResults.TrnsKLD.max(), mResults.NeepScg.max())], [0, max(mResults.TrnsKLD.max(), mResults.NeepScg.max())], 'k')
# plt.xlabel("TrnsKLD")
# plt.ylabel("NeepScg")
# plt.show()

resFig1 = plt.figure()
plt.hist(mResults.TrnsKLD - mResults.ScgPlgn, 30)
# plt.xlabel("TrnsKLD")
# plt.ylabel("ScgPLGN")
plt.show()
resFig1.set_size_inches((2 * 3.38582677, 3.38582677))
# resFig1.savefig(f'NeepRobustness.pdf')
# plt.scatter(mResults.ScgPlgn, mResults.NeepScg, 8)
# plt.plot([0, max(mResults.ScgPlgn.max(), mResults.NeepScg.max())], [0, max(mResults.ScgPlgn.max(), mResults.NeepScg.max())], 'k')
# plt.xlabel("ScgPLGN")
# plt.ylabel("NeepScg")
# plt.show()

# compare against full EPR
resFig = plt.figure()
plt.scatter(mResults.FullEpr, mResults.TrnsKLD, 9, label='$\sigma_{\mathrm{KLD,S-CG}}$')
plt.scatter(mResults.FullEpr, mResults.ScgPlgn, 7, label='$\sigma_{\mathrm{plug,S-CG}}$')
plt.scatter(mResults.FullEpr, mResults.NeepScg, 5, label='$\sigma_{\mathrm{neep,S-CG}}$')
vRange = [0, max(mResults.FullEpr.max(), mResults.TrnsKLD.max(), mResults.NeepScg.max(), mResults.ScgPlgn.max())]
plt.plot(vRange, vRange, 'k')
plt.xlabel("$\sigma_{\mathrm{tot}}$", fontsize='small')
plt.ylabel("Estimated EPR", fontsize='small')
plt.legend(prop={'size': 5})
plt.tick_params(axis="both", labelsize=6)
plt.show()
resFig.set_size_inches((2 * 3.38582677, 3.38582677))
resFig.savefig(f'NeepRobustness.pdf')
# plt.scatter(mResults.FullEpr, mResults.NeepScg, 8)
# plt.plot([0, max(mResults.FullEpr.max(), mResults.NeepScg.max())], [0, max(mResults.FullEpr.max(), mResults.NeepScg.max())], 'k')
# plt.xlabel("FullEpr")
# plt.ylabel("NeepScg")
# plt.show()
#
# plt.scatter(mResults.FullEpr, mResults.ScgPlgn, 8)
# plt.plot([0, max(mResults.FullEpr.max(), mResults.ScgPlgn.max())], [0, max(mResults.FullEpr.max(), mResults.ScgPlgn.max())], 'k')
# plt.xlabel("FullEpr")
# plt.ylabel("ScgPLGN")
# plt.show()

