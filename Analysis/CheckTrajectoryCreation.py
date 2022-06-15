# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:23:27 2020

@title: Compare methods of trajectory creation : MonteCarlo and straight method

@author: Uri Kapustin

"""
import time
import numpy as np

from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
from PhysicalModels.TrajectoryCreation import CreateTrajectory, EstimateTrajParams
from PhysicalModels.PartialTrajectories import CreateCoarseGrainedTraj, CoarseGrainTrajectory
from Utility.Params import BaseSystem


def AggregateTrajectory(mW, nDim, timeRes, nWanted, nSeq=32):
    nInters = int(np.floor(nWanted/nSeq))
    mTraj = np.zeros((nWanted, 2))
    nEffective = 0

    for iIter in range(nInters):
        if iIter == 0:
            tic = time.time()
            mSeq, nCgDim = CreateTrajectory(nDim, nSeq + 1, [0], mW)
            toc = time.time()
            print(toc-tic)
        else:
            mSeq, _ = CreateTrajectory(nDim, nSeq + 1, [int(mSeq[-1:, 0].item())], mW)  # Run Create Trajectory
        mSeq = mSeq[1:, :]
        currLen = mSeq.shape[0]
        mTraj[nEffective:(nEffective + currLen), :] = mSeq
        nEffective += currLen
    mTraj = mTraj[:nEffective, :]
    return mTraj

    # Define Ground truth system
mW, nDim, vHiddenStates, timeRes = BaseSystem()

# Define analysis parameters
nIters = 10
nTrajLength = int(5e5)

# Initialize memory for results
vErrStr = np.zeros(nIters)
vTimeStr = np.zeros(nIters)
vErrMc  = np.zeros(nIters)
vTimeMc = np.zeros(nIters)
vErrMcB  = np.zeros(nIters)
vTimeMcB = np.zeros(nIters)

for iIter in range(nIters):
    # First handle straight creation method
    tic = time.time()
    mTraj, _ = CreateTrajectory(nDim, nTrajLength, [0], mW)
    toc = time.time()
    _, _, _, mWest, _ = EstimateTrajParams(mTraj)
    vErrStr[iIter] = np.linalg.norm(mW-mWest)
    vTimeStr[iIter] = toc-tic

    # Than handle MC creation method
    tic = time.time()
    mTraj, _ = CreateTrajectory(nDim, nTrajLength, [0], mW, True)
    toc = time.time()
    _, _, _, mWest, _ = EstimateTrajParams(mTraj)
    vErrMc[iIter] = np.linalg.norm(mW-mWest)
    vTimeMc[iIter] = toc-tic

    # Aggregate sequences
    tic = time.time()
    mTraj = AggregateTrajectory(mW, nDim, timeRes, int(4096*128), nSeq=int(4096*128))
    toc = time.time()
    _, _, _, mWest, _ = EstimateTrajParams(mTraj)
    vErrMcB[iIter] = np.linalg.norm(mW-mWest)
    vTimeMcB[iIter] = toc-tic
    print('Finished Iter #'+str(iIter))

print('Straight method - Mean Error: ' +str(vErrStr.mean())+ ' | Std Error'+ str(vErrStr.std()))
print('Straight method - Mean Time: ' +str(vTimeStr.mean())+ ' | Std Time'+ str(vTimeStr.std()))
print('MonteCarlo method - Mean Error: ' +str(vErrMc.mean())+ ' | Std Error'+ str(vErrMc.std()))
print('MonteCarlo method - Mean Time: ' +str(vTimeMc.mean())+ ' | Std Time'+ str(vTimeMc.std()))
print('MC sequences method - Mean Error: ' +str(vErrMcB.mean())+ ' | Std Error'+ str(vErrMcB.std()))
print('MC sequences method - Mean Time: ' +str(vTimeMcB.mean())+ ' | Std Time'+ str(vTimeMcB.std()))