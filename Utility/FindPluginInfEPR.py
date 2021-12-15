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
def CreateGilisTrajectory(nTimeStamps=5e6, x=0.):
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
def EstimatePluginM(vStatesTraj, m):
    # We will use the symmetry(anti-symmetry) of the KLD for our favor.
    # The following algorithm uses this property of the target KLD.
    nEffLength = len(vStatesTraj) - m + 1#np.floor(len(vStatesTraj)/m)#
    vSeq2Dec = np.power(10, range(m))
    kldEstm = 0
    endFlag = False
    countF = 0
    countB = 0
    eps = 1e-9
    # Collect sequence statistics for one direction sequences
    vIndicateSeq = np.correlate(vStatesTraj, vSeq2Dec) # this correlation equal to coding each sequence
    vSeqs1, vProbOfSeq1 = np.unique(vIndicateSeq, return_counts=True) # this function is also sorting values and their counts
    vProbOfSeq1 = vProbOfSeq1/nEffLength # these are the "true" probabilities for each sequence

    # Same for other direction
    vIndicateSeq = np.correlate(vStatesTraj, np.flip(vSeq2Dec)) # this correlation equal to coding each sequence
    vSeqs2, vProbOfSeq2 = np.unique(vIndicateSeq, return_counts=True) # this function is also sorting values and their counts
    vProbOfSeq2= vProbOfSeq2/nEffLength # used to calculate the probabilities log ratio of forward and backward trajectory

    while not endFlag:
        if vSeqs1[countF] == vSeqs2[countB]:
            kldEstm += vProbOfSeq1[countF]*np.log(vProbOfSeq1[countF]/vProbOfSeq2[countB])
            countF += 1
            countB += 1
        elif vSeqs1[countF] > vSeqs2[countB]:
            countB += 1
        else:
            countF += 1
        if countF >= len(vProbOfSeq1) or countB >= len(vProbOfSeq1):
            endFlag = True
    ## This works only if all the possible sequences are observed
    # kldEstm = np.sum(np.multiply(vProbOfSeq1, np.log(vProbOfSeq1/vProbOfSeq2)))
    return kldEstm


# %% Estimate \hat{d}_\infty - plugin estimator in infinity
def EstimatePluginInf(mCgTrajectory):
    # By fitting plugin estimator of order m we find the infinity plugin
    maxSeq = 7
    vMgrid = np.linspace(2, maxSeq, maxSeq-2+1, dtype=np.intc)
    vGrid2Fit = np.concatenate(([2], range(3, maxSeq+1, 2)))
    vKldM = np.ones(vMgrid.shape)
    vEprEst = np.ones(vGrid2Fit.shape)
    # Collect data for fitting
    for m, iM in enumerate(vMgrid):
        print('Estimating KLD for seq size: ' + str(iM))
        vKldM[m] = EstimatePluginM(mCgTrajectory, iM)
    print('Completed gather fitting points. Start fit...')
    # Fit the data

    vEprEst[0] = vKldM[0]
    vEprEst[1:] = vKldM[1::2] - vKldM[::2]
    popt, _ = curve_fit(FitFunc, vGrid2Fit, vEprEst)
    kldInf = popt[0]

    return kldInf


def FitFunc(x, b, c, g):
    return b - c*(np.log(x)/x**g)

def SemiAnalyticalKLD(vTraj, mW):
    return True

if __name__ == '__main__':
    nTimeStamps=1e7
    x = 2.
    # trainDataSet, nCgDim = CreateGilisTrajectory(nTimeStamps=nTimeStamps, x=x)
    # mCgTrajectory = trainDataSet[:, 0]
    mCgTrajectory = CreateNEEPTrajectory(nTimeStamps=nTimeStamps, x=x, fullCg=True)
    # vMgrid = np.array([2,3,5,7,9])
    kldInf = EstimatePluginInf(mCgTrajectory)
    eprNeepAnalytic = rt.ep_per_step(x)
