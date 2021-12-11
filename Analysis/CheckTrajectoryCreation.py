# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:23:27 2020

@title: Compare methods of trajectory creation : MonteCarlo and straight method

@author: Uri Kapustin

"""
import time
import numpy as np

from PhysicalModels.TrajectoryCreation import CreateTrajectory, EstimateTrajParams
from Utility.Params import BaseSystem


# Define Ground truth system
mW, nDim, vHiddenStates, timeRes = BaseSystem()

# Define analysis parameters
nIters = 10
nTrajLength = int(1e7)

# Initialize memory for results
vErrStr = np.zeros(nIters)
vTimeStr = np.zeros(nIters)
vErrMc  = np.zeros(nIters)
vTimeMc = np.zeros(nIters)

for iIter in range(nIters):
    # First handle straight creation method
    tic = time.time()
    mTraj, _ = CreateTrajectory(nDim, nTrajLength, [0], mW)
    toc = time.time()
    _, _, _, mWest, _ = EstimateTrajParams(nDim, mTraj)
    vErrStr[iIter] = np.linalg.norm(mW-mWest)
    vTimeStr[iIter] = toc-tic
    # Than handle MC creation method
    tic = time.time()
    mTraj, _ = CreateTrajectory(nDim, nTrajLength, [0], mW, True)
    toc = time.time()
    _, _, _, mWest, _ = EstimateTrajParams(nDim, mTraj)
    vErrMc[iIter] = np.linalg.norm(mW-mWest)
    vTimeMc[iIter] = toc-tic
    print('Finished Iter #'+str(iIter))
print('Straight method - Mean Error: ' +str(vErrStr.mean())+ ' | Std Error'+ str(vErrStr.std()))
print('Straight method - Mean Time: ' +str(vTimeStr.mean())+ ' | Std Time'+ str(vTimeStr.std()))
print('MonteCarlo method - Mean Error: ' +str(vErrMc.mean())+ ' | Std Error'+ str(vErrMc.std()))
print('MonteCarlo method - Mean Time: ' +str(vTimeMc.mean())+ ' | Std Time'+ str(vTimeMc.std()))
