# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:04:45 2020

@title: Contains the methods to calculate EPR KLD plugin estimator (Edgar Rold´an and Juan M. R. Parrondo, 2012)

@author: Uri Kapustin

@description: -

"""

import numpy as np

from scipy.optimize import curve_fit

import PhysicalModels.TrajectoryCreation as tc
import PhysicalModels.PartialTrajectories as pt
import PhysicalModels.ratchet as rt
import PhysicalModels.UtilityTraj as ut
# %% Create trajectories
def CreateGilisTrajectory(nTimeStamps=5e5, x=0.):
    nTimeStamps = int(nTimeStamps)
    nDim = 4
    mW = np.array([[-11., 2., 0., 1.], [3., -52.2, 2., 35.], [0., 50., -77., 0.7], [8., 0.2, 75., -36.7]])
    timeRes = 0.001
    vHiddenStates = np.array([2, 3])  # states 3 and 4 for 4-D state sytem
    # Adjust the rate matrix according to the applied external force
    mWx = pt.CalcW4DrivingForce(mW, x)
    mCgTrajectory, nCgDim = pt.CreateCoarseGrainedTraj(nDim, nTimeStamps, mWx, vHiddenStates, timeRes)
    return mCgTrajectory, nCgDim


def CreateNEEPTrajectory(nTimeStamps=5e4, x=0., fullCg=False):
    nTimeStamps = int(nTimeStamps)
    data = rt.simulation(1, nTimeStamps, x, seed=0)
    mCgTrajectory = data[0] % 3
    trainBuffer = np.zeros(mCgTrajectory.size)
    if fullCg:
        for iStep in range(mCgTrajectory.size):
        # now decimate train and test set
            # handle special case of first element
            if iStep == 0:
                trainBuffer[iStep] = 1
                continue
            # create mask for train set
            if mCgTrajectory[iStep-1] != mCgTrajectory[iStep]:
                trainBuffer[iStep] = 1
        mCgTrajectory = mCgTrajectory[trainBuffer==1]
    return mCgTrajectory


# %% Estimate \hat{d}_m - plugin estimator with sequence of m
def EstimatePluginM(mCgTrajectory, m):
    mMemory, nUniqueSeq, nTotSeq = CountSequences(mCgTrajectory, m)
    mProb = mMemory[:,-2:]/nTotSeq
    kldEstm = 0

    for iSeq in range(nUniqueSeq):
        if mProb[iSeq,0] > mProb[iSeq,1]:
            probF = mProb[iSeq,0]
            probR = mProb[iSeq,1]
        else:
            probF = mProb[iSeq,1]
            probR = mProb[iSeq,0]
        if probF > 0 and probR > 0:
            kldEstm += probF*np.log(probF/probR)
    return kldEstm

def CountSequences(vStatesTraj, m):
    assert m >= 2, "Length of sequence must be greater or equal to 2! Otherwise it's not physical"
    assert m <= 9, "Length of sequence need to be less than 10 in order to work with floats" # TODO: increase support by using double

    #vStatesTraj = trainDataSet[:, 0]
    #mMemory = np.zeros((3*2**int(m-1), 3))  # number of rows is as number of sequence options and columns are - <sequence in decimal base/2, repitions, rep's of reverse
    mMemory = np.zeros((3 * 2 ** 9 * 10, 3)) # upper limit, most probably zeros but therer is no memory problem
    nUniqueSeq = 0
    nEffLength = len(vStatesTraj) - m + 1#np.floor(len(vStatesTraj)/m)#
    vSeq2Dec = np.power(10, range(m))
    # Iterate over all sequences in the trajectory
    for iStep in range(nEffLength):#range(0,int(nEffLength*m),m):#
        newFlag = True  # used to understand if a new sequence is observed
        decSeq = np.dot(vSeq2Dec, vStatesTraj[iStep:(iStep+m)])
        decSeqRev = np.dot(vSeq2Dec, np.flip(vStatesTraj[iStep:(iStep+m)]))
        # Search if this sequence already observed and count it if needed
        for i in range(nUniqueSeq):
            if decSeq == mMemory[i, 0]:
                mMemory[i, 1] += 1
                newFlag = False
                break
            elif decSeqRev == mMemory[i, 0]:
                mMemory[i, 2] += 1
                newFlag = False
                break
        # Case the sequence not found in memory - add a new entry to the memory
        if newFlag:
            nUniqueSeq += 1
            mMemory[nUniqueSeq-1, :] = np.array([decSeq, 1, 0])

    return mMemory, nUniqueSeq, nEffLength

# %% Estimate \hat{d}_\infty - plugin estimator in infinity
def EstimatePluginInf(mCgTrajectory, vMgrid):
    # By fitting plugin estimator of order m we find the infinity plugin
    vKldM = np.ones(vMgrid.shape)
    # Collect data for fitting
    for m, iM in enumerate(vMgrid):
        print('Estimating KLD for seq size: '+ str(iM))
        vKldM[m] = EstimatePluginM(mCgTrajectory, iM)
    print('Completed gather fitting points. Start fit...')
    # Fit the data
    popt, _ = curve_fit(FitFunc, vMgrid, vKldM)
    kldInf = popt[0]

    return kldInf


def FitFunc(x, b, c, g):
    return b - c*np.divide(np.log(x), x**g)


if __name__ == '__main__':
    nTimeStamps=10e5
    x = 1.
    #trainDataSet, nCgDim = CreateGilisTrajectory(nTimeStamps=nTimeStamps, x=x)
    #trainDataSet = trainDataSet[:, 0]
    mCgTrajectory = CreateNEEPTrajectory(nTimeStamps=nTimeStamps, x=x, fullCg=False)
    vMgrid = np.array([2,3,5,7,9])
    kldInf = EstimatePluginInf(mCgTrajectory, vMgrid)
    eprNeepAnalytic = rt.ep_per_step(x)
