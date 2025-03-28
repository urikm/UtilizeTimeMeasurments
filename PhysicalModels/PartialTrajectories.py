# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:30:24 2020

@title: Partial Information from Trajectories Analyis(Reproduction of heirarchial bounds paper from 2017)
    
@author: Uri Kapustin

@description: Create coarse-grained trajectories which are not markov anymore
"""

# %% Imports
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity as KD
from sklearn.model_selection import GridSearchCV, LeaveOneOut

from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver

from PhysicalModels.TrajectoryCreation import CreateTrajectory, EstimateTrajParams
from PhysicalModels.UtilityTraj import CalcSteadyStateCurrent, EntropyRateCalculation, MapStates2Indices
from numba import njit


# %% Assuming only states 0,1 are observable. TODO : maybe make more generic?
# Calculate passive entropy production rate as explained in 2017 hierarchical bounds paper
def CalcPassivePartialEntropyProdRate(mW, vPi):
    # This function used for informed and passive rates when expecting the right 'W' and 'Pi' state for each rate
    sigmaPP = 0
    # Use pairs of state where i<j, as described in paper
    for iState in range(1):  # range(nDim):
        for jState in range(1, 2):  # range(iState+1,nDim):
            Jij, J_j2i, J_i2j = CalcSteadyStateCurrent(mW, vPi, iState, jState)  # Calculate Current between two states
            if Jij != 0:
                sigmaPP += Jij * np.log((mW[iState, jState] * vPi[jState]) / (mW[jState, iState] * vPi[iState]))
    return sigmaPP


# Calculate informed entropy production rate as explained in 2017 hierarchical bounds paper
def CalcInformedPartialEntropyProdRate(mW, vPi, vPiSt):
    # This function used for informed and passive rates when expecting the right 'W' and 'Pi' state for each rate    
    sigmaIP = 0
    # Use pairs of state where i<j, as described in paper
    for iState in range(1):  # range(nDim):
        for jState in range(1, 2):  # range(iState+1,nDim):
            Jij, J_j2i, J_i2j = CalcSteadyStateCurrent(mW, vPi, iState, jState)  # Calculate Current 0->1
            if Jij != 0:
                sigmaIP += Jij * np.log((mW[iState, jState] * vPiSt[jState]) / (mW[jState, iState] * vPiSt[iState]))
    return sigmaIP


# Calculate stalling steady state and independet rates as explained in 2017 hierarchical bounds paper Appendix B
def CalcStallingData(mW):
    # Calcualte the needed determinants
    mTmp = mW[(0, 2, 3), :]
    mTmp = mTmp[:, (1, 2, 3)]
    detWno10 = np.linalg.det(mTmp)
    mTmp2 = mW[(1, 2, 3), :]
    mTmp2 = mTmp2[:, (0, 2, 3)]
    detWno01 = np.linalg.det(mTmp2)
    detWno01Atall = np.linalg.det(mW[2:, 2:])

    # Calculate stalling steady state 
    vPiSt, mWst = CalcStallingSteadyState(mW)

    # Calculate independent rates
    r01 = detWno10 / detWno01Atall
    r10 = detWno01 / detWno01Atall

    # Check if calculated steady state hold the independant rate ratio
    assert (np.abs((vPiSt[1] / vPiSt[0]) - (r10 / r01)) < 1e1), "Calculated Stalling Steady State is wrong"

    # Calculate stalling force
    xSt = CalcStallingForce(mW, vPiSt)

    return vPiSt, xSt, r01, r10


# Calculate stalling Force as explained in 2017 hierarchical bounds paper Simulation
def CalcStallingForce(mW, vPiSt):
    xSt = -0.5 * np.log((mW[1, 0] * vPiSt[0]) / (mW[0, 1] * vPiSt[1]))
    return xSt


# Calculate stalling steady state by manually setting to zero rates
def CalcStallingSteadyState(mW):
    mWst = np.copy(mW)
    # Update diagonal elements according to the new zero rates
    mWst[0, 0] = mWst[0, 0] + mWst[1, 0]
    mWst[1, 1] = mWst[1, 1] + mWst[0, 1]

    # Set to zero wanted rates
    mWst[0, 1] = 0
    mWst[1, 0] = 0

    nDim = np.size(mWst, 0)
    vP0 = np.random.uniform(size=(nDim))
    vP0 = vP0 / sum(vP0)
    n, vPiSt, mWst, vWPn = MESolver(nDim, vP0, mWst, 0.01)
    return vPiSt, mWst


# Calculate stalling W matrix when external force(x) applied on control states
# TODO : change x -> -x and vise verca due to my confusion with previous definitions
def CalcW4DrivingForce(mW, x, control=[0,1]):
    # assume exponential force over the rate between 0<->1
    mWx = np.array(mW)

    # Update diagonal values
    mWx[control[0], control[0]] += mWx[1, 0] * (1 - np.exp(x))
    mWx[control[1], control[1]] += mWx[control[0], control[1]] * (1 - np.exp(-x))

    # Update rates according to driving force
    mWx[control[1], control[0]] = mWx[control[1], control[0]] * np.exp(x)
    mWx[control[0], control[1]] = mWx[control[0], control[1]] * np.exp(-x)

    return mWx


################################# Priority to be updated
# %% Coarse Grain trajectory ; TODO: support several hidden states(not in remap mode)
@njit
def CoarseGrainTrajectory(mTrajectory, nFullDim, vHiddenStates, semiCG=False, remap=False, maxAddedStates=30):
    mCgTrajectory = np.copy(mTrajectory)
    nJumps = mCgTrajectory.shape[0]
    iHidden = nFullDim - vHiddenStates.shape[0]
    vNewHidden = np.zeros(maxAddedStates, dtype=np.int64)
    nHid = 1

    if semiCG:
        hState = iHidden
    else:
        hState = -2  # iHidden #  define as intermediate state every hidden state that is not last in it's sequence

    for iJump in range(nJumps):

        if mCgTrajectory[iJump, 0] in vHiddenStates:
            if iJump < nJumps - 1:
                if mCgTrajectory[iJump + 1, 0] in vHiddenStates:
                    mCgTrajectory[iJump, 0] = hState
                    if not semiCG:
                        mCgTrajectory[iJump + 1, 1] += mCgTrajectory[iJump, 1]  # cumsum waiting times for each sequence
                else:
                    mCgTrajectory[iJump, 0] = iHidden
            else:
                mCgTrajectory[iJump, 0] = iHidden

    nCgDim = iHidden + 1
    vNewHidden[0] = iHidden

    # Final step of CG
    if semiCG:
        if remap:  # remap into "infinite" state representation
            mCgTrajectory, nCgDim, vNewHidden, nHid = RemapStates(mCgTrajectory, hState, maxAddedStates=maxAddedStates)  # override dim of states in remap mode
        else:
            pass  # Dont do nothing in semi CG without remap
    else:
        mCgTrajectory = mCgTrajectory[(mCgTrajectory[:, 0] != hState), :]  # remove '-2' states in Full CG


    return mCgTrajectory, nCgDim, vNewHidden, nHid

# Remap semi CG trajectories to system with, possibly, infinite states. A new state for sequence size of hidden jumps
@njit
def RemapStates(mCgTrajectory, hidState, maxAddedStates=30, baseState=100):
    nJumps = mCgTrajectory.shape[0]
    vNewHidden = np.arange(baseState + 1, baseState + maxAddedStates + 1)
    vMask = np.zeros(vNewHidden.shape, dtype=np.int64)

    # identify and clamp the different states(which depends on the sequence size og hidden jumps)
    hidSeqCount = 0
    for iJump in range(nJumps):
        if mCgTrajectory[iJump, 0] == hidState:
            if hidSeqCount > 0:
                mCgTrajectory[iJump, 1] += mCgTrajectory[iJump - 1, 1]
            hidSeqCount += 1

        elif hidSeqCount > 0:  # end of hidden sequence
            mCgTrajectory[iJump - 1, 0] = hidSeqCount + baseState  # mark the end of sequence which hold the cumsum WTD
            if hidSeqCount + baseState in vNewHidden:
                mCgTrajectory[iJump - 1, 0] = hidSeqCount + baseState   # in this case tag it as the state with max transition possible
                vMask[hidSeqCount + baseState == vNewHidden] = 1
            else:  # convert it to the max possible sequences length, in order to avoid numerical issues
                mCgTrajectory[iJump - 1, 0] = vNewHidden.max()
                vMask[-1] = 1
            hidSeqCount = 0
    vNewHidden = vNewHidden[(vMask == 1)]
    mCgTrajectory = mCgTrajectory[(mCgTrajectory[:, 0] != hidState), :]
    nCgDim = np.unique(mCgTrajectory[:, 0]).size

    nAdded = np.sum(vMask)
    return mCgTrajectory, nCgDim, vNewHidden, nAdded


# %% Estimate 2nd order statistics of the trajectory (i->j->k transitions)
def EstimateTrajParams2ndOrder(mTrajectory, vHiddenStates, states2Omit=[]):
    vInitStates = np.unique(mTrajectory[:, 0])
    vInitStates, dMap = MapStates2Indices(vInitStates, states2Omit=states2Omit)
    nDim = vInitStates.size

    mP2ndOrderTransitions = np.zeros((nDim,) * 3)  # all the Pijk with i!=k
    mWtd = []
    vDebug = []

    for omit in states2Omit:  # avoid analysing unwanted hidden states
        vHiddenStates = np.delete(vHiddenStates, np.where(vHiddenStates == omit))

    for iState in vInitStates:
        vIndInitTrans = np.array(np.where(mTrajectory[:-2, 0] == iState)[0])
        nTrans4State = np.size(vIndInitTrans, 0)
        vFirstTrans = np.roll(vInitStates, -dMap[iState])[1:]
        for jState in vFirstTrans:
            vIndFirstTrans = vIndInitTrans[np.array(np.where(mTrajectory[vIndInitTrans + 1, 0] == jState)[0])] + 1
            vSecondTrans = np.roll(vInitStates, -dMap[jState])[1:]
            for kState in vSecondTrans:
                #if kState == iState:
                #    continue
                vIndSecondTrans = vIndFirstTrans[np.array(np.where(mTrajectory[vIndFirstTrans + 1, 0] == kState)[0])] + 1
                mP2ndOrderTransitions[dMap[iState], dMap[jState], dMap[kState]] = vIndSecondTrans.size / (mTrajectory[:, 0].size - 2) #np.size(vIndSecondTrans, 0) / (nTrans4State + 1e-17)
                if (jState in vHiddenStates) & (iState != kState):  # The middle (j) state is POI and other 2 should be different to not vanish
                    # we interseted in the waiting times of the middle state (j)
                    mWtd.append(mTrajectory[vIndSecondTrans - 1, 1])
                    vDebug.append(iState*100 + jState*10 + kState)

    return mP2ndOrderTransitions, mWtd, vDebug


# Calculate KLD entropy production rate as explained in 2019  paper
def CalcKLDPartialEntropyProdRate(mCgTrajectory, vHiddenStates, states2Omit=[]):
    # Extract the possible states and map them to indices of rate matrix
    vStates = np.unique(mCgTrajectory[:, 0])
    vStates, dMap = MapStates2Indices(vStates, states2Omit=states2Omit)
    nStates = len(vStates)


    # Estimate 1st and 2nd order statistics from hidden trajectory
    mIndStates, _, _, _, _ = EstimateTrajParams(mCgTrajectory, states2Omit=states2Omit)
    mP2ndOrdTrans, mWtd, vDebug = EstimateTrajParams2ndOrder(mCgTrajectory, vHiddenStates, states2Omit=states2Omit)

    # Calculate steady state distribution occurence of each state(per jump)
    vR = np.zeros(nStates)
    for iState in vStates:
        vR[dMap[iState]] = np.size(mIndStates[dMap[iState]], 0)
    nTot = mCgTrajectory.shape[0]
    vR /= nTot

    eps = 1e-8  # used for numerical stability

    # Time factor('tau') to convert from "per jump" to "per step" - mean dwelling time per jump
    T = np.mean(mCgTrajectory[:, 1])  # Also can be the mean over mean dwelling per state - np.dot(vTau, vR)

    ## Find affinity part
    # Math: R12 = p21*R[1] = (tau[1]*w21)*(Pi[1]*T/tau[1])=w21*Pi[1]*T
    vRij = np.zeros((nStates,) * 2)  # Number of possible jump i->j = nStates*(nStates-1)
    for iState in vStates:
        iState = int(iState)
        vTransStates = np.roll(vStates, -dMap[iState])[1:]
        vTmp = mIndStates[dMap[iState]] + 1
        if vTmp[-1] == nTot:
            vTmp = vTmp[:-1]

        for jState in vTransStates:
            vSecJump = np.where(mCgTrajectory[vTmp, 0] == jState)
            vRij[dMap[iState], dMap[jState]] = vSecJump[0].size / (nTot - 1)

    # Pijk = Pr{to observe i>j>k} => Pijk=R[ijk]*R[i](Related only to state, not time related)
    vPij_jk = np.zeros((nStates,) * 3)
    for iState in vStates:
        vFirstTrans = np.roll(vStates, -dMap[iState])[1:]
        for jState in vFirstTrans:
            vSecondTrans = np.roll(vStates, -dMap[jState])[1:]
            for kState in vSecondTrans:
                if vRij[dMap[iState], dMap[jState]] > 0:
                    # vPij_jk[dMap[iState], dMap[jState], dMap[kState]] = mP2ndOrdTrans[dMap[iState], dMap[jState], dMap[kState]] * vR[dMap[iState]] / vRij[dMap[iState], dMap[jState]]
                    vPij_jk[dMap[iState], dMap[jState], dMap[kState]] = mP2ndOrdTrans[dMap[iState], dMap[jState], dMap[kState]] / vRij[dMap[iState], dMap[jState]]
    # Calculate Affinity part
    vPij_jk[vPij_jk < eps] = eps
    sigmaDotAff = 0
    sigmaDotAffAlt = 0
    for iState in vStates:
        vFirstTrans = np.roll(vStates, -dMap[iState])[1:]
        for jState in vFirstTrans:
            vSecondTrans = np.roll(vStates, -dMap[jState])[1:]
            for kState in vSecondTrans:
                if vPij_jk[dMap[iState], dMap[jState], dMap[kState]] > eps and vPij_jk[dMap[kState], dMap[jState], dMap[iState]] > eps:
                    # The following was seperated for DDEBUG purpose
                    singleDirAff1 = mP2ndOrdTrans[dMap[iState], dMap[jState], dMap[kState]] * np.log(
                        vPij_jk[dMap[iState], dMap[jState], dMap[kState]] / vPij_jk[
                            dMap[kState], dMap[jState], dMap[iState]])
                    singleDirAff2 = mP2ndOrdTrans[dMap[kState], dMap[jState], dMap[iState]] * np.log(
                        vPij_jk[dMap[kState], dMap[jState], dMap[iState]] / vPij_jk[
                            dMap[iState], dMap[jState], dMap[kState]])
                    affAddition = singleDirAff1 + singleDirAff2
                    sigmaDotAff += affAddition / 2  # because every thing will be calculated twice
                elif (vPij_jk[dMap[iState], dMap[jState], dMap[kState]] > eps and vPij_jk[dMap[kState], dMap[jState], dMap[iState]] == eps) or (vPij_jk[dMap[iState], dMap[jState], dMap[kState]] == eps and vPij_jk[dMap[kState], dMap[jState], dMap[iState]] > eps):
                    singleDirAff1 = mP2ndOrdTrans[dMap[iState], dMap[jState], dMap[kState]] * np.log(
                        vPij_jk[dMap[iState], dMap[jState], dMap[kState]] / vPij_jk[
                            dMap[kState], dMap[jState], dMap[iState]])
                    singleDirAff2 = mP2ndOrdTrans[dMap[kState], dMap[jState], dMap[iState]] * np.log(
                        vPij_jk[dMap[kState], dMap[jState], dMap[iState]] / vPij_jk[
                            dMap[iState], dMap[jState], dMap[kState]])
                    affAddition = singleDirAff1 + singleDirAff2
                    sigmaDotAffAlt += affAddition / 2  # because every thing will be calculated twice
    sigmaDotAff /= T
    sigmaDotAffAlt /= T
    # Calculate WTD part
    maxWTD = np.percentile(mCgTrajectory[:, 1], 98)
    minWTD = np.percentile(mCgTrajectory[:, 1], 0.1)
    nPoints = 500
    vGridDest = np.linspace(minWTD, maxWTD, nPoints)  # np.linspace(0, 0.25, 100)  #

    sigmaDotWtd = 0
    countWtd = 0
    countDebug = 0

    # initial distribution is a delta on the lowest bin
    mWtdBuffer = np.zeros((nStates, nStates, nStates, len(vGridDest)))
    # mWtdBuffer[:, :, :, 0] = 1 - eps*(len(vGridDest)-1)  # normalize the distribution
    # Accumulate the pdf of each relevant i->j->k
    for iState in vStates:
        vFirstTrans = np.roll(vStates, -dMap[iState])[1:]
        for jState in vFirstTrans:
            vSecondTrans = np.roll(vStates, -dMap[jState])[1:]
            for kState in vSecondTrans:
                if (jState in vHiddenStates) & (iState != kState):
                    if mWtd[countWtd].size >= 1000:
                        # Silverman's rule of thumb - The internal implementation is buggy so did it myself for now
                        bw = 0.9 * min(mWtd[countWtd].std(), (np.percentile(mWtd[countWtd], 75) - np.percentile(mWtd[countWtd], 25)) / 1.34) \
                            * mWtd[countWtd].size**(-1/5)
                        kde = KD(bandwidth=bw)
                        kde.fit(mWtd[countWtd][:, None])
                        pDdiHk = np.exp(kde.score_samples(vGridDest[:, None]))
                        mWtdBuffer[dMap[iState], dMap[jState], dMap[kState], :] = pDdiHk
                        countDebug += 1
                    countWtd += 1
    mWtdBuffer[mWtdBuffer < eps] = eps
    for iState in vStates:
        vFirstTrans = np.roll(vStates, -dMap[iState])[1:]
        for jState in vFirstTrans:
            vSecondTrans = np.roll(vStates, -dMap[jState])[1:]
            for kState in vSecondTrans:
                if (jState in vHiddenStates) and (iState != kState):
                    vPSIijk = mWtdBuffer[dMap[iState], dMap[jState], dMap[kState]]
                    vPSIkji = mWtdBuffer[dMap[kState], dMap[jState], dMap[iState]]
                    if vPSIijk.max() == eps or vPSIkji.max() == eps:  #ADD: or np.isnan(vPSIijk).sum() > 0 or np.isnan(vPSIkji).sum() > 0 :
                       continue
                    #kldPsi = np.sum(np.multiply(vPSIijk, np.log(vPSIijk) - np.log(vPSIkji)))
                    kldPsi = np.trapz(np.multiply(vPSIijk, np.log(vPSIijk) - np.log(vPSIkji)), vGridDest)
                    # sigmaDotWtd += (mP2ndOrdTrans[dMap[iState], dMap[jState], dMap[kState]] * vR[dMap[iState]] / T) * kldPsi
                    sigmaDotWtd += (mP2ndOrdTrans[dMap[iState], dMap[jState], dMap[kState]] / T) * kldPsi
    sigmaDotKld = sigmaDotAff + sigmaDotWtd
    return sigmaDotKld, T, sigmaDotAff, sigmaDotWtd

# TODO : vHiddenStates actually means which states should be hidden. In estimation process it used as the states that defined hidden
def CreateCoarseGrainedTraj(nDim, nTimeStamps, mW, vHiddenStates, timeRes, semiCG=False, isCG=True, remap=False):
    # randomize init state from the steady-state distribution
    vP0 = np.array([0.25, 0.25, 0.25, 0.25])  # , dtype=np.float32)
    n, vPi, mW, vWPn = MESolver(nDim, vP0, mW, timeRes)
    normP = vPi.sum()  # due to numeric precision problems we need to normalize to 1
    if normP > 0.999:
        vPi = vPi / normP
    else:
        assert 0, "The Master equation solver doesnt converge for this system - you should look at it"
    initState = np.random.choice(nDim, 1, p=vPi).item()
    # Create trajectory
    mTrajectory, mW = CreateTrajectory(nDim, nTimeStamps, initState, mW)  # Run Create Trajectory
    if isCG:
        mCgTrajectory, nCgDim, vHiddenStates, nHid = CoarseGrainTrajectory(mTrajectory, nDim, vHiddenStates, semiCG=semiCG, remap=remap)
        vHiddenStates = vHiddenStates[:nHid]  # support for numba compilation
    else:
        mCgTrajectory = mTrajectory
        nCgDim = nDim
    return mCgTrajectory, nCgDim, vHiddenStates


# %% Analysis Partial Entropy production on single trajectory
if __name__ == '__main__':
    ## UI
    flagPlot = True
    nDim = 4  # dimension of the problem
    # nTimeStamps = int(1e5) # how much time stamps will be saved
    initState = rd.randrange(nDim)  # Define initial state in T=0
    # mW = GenRateMat(nDim) # transition matrix
    mW = np.array([[-11., 1., 0., 7.],
                   [9., -11., 10., 1.],
                   [0., 4., -15., 8.],
                   [2., 6., 5., -16.]])

    # mTrajectory,mW = CreateTrajectory(nDim,nTimeStamps,initState,mW) # Run Create Trajectory
    # # Calculate Steady state
    # Calculate Stalling data
    vPiSt, xSt, r01, r10 = CalcStallingData(mW)

    # Init vectors for plotting
    vGrid = np.arange(-7., 7., 0.1)  # TODO make smarter grid, use stalling data
    vInformed = np.zeros(np.size(vGrid))
    vPassive = np.zeros(np.size(vGrid))
    vFull = np.zeros(np.size(vGrid))
    i = 0
    for x in vGrid:
        mWx = CalcW4DrivingForce(mW, x)  # Calculate stalling W matrix
        # Passive partial entropy production rate
        vP0 = np.random.uniform(size=nDim)
        vP0 = vP0 / sum(vP0)
        n, vPiX, mWst, vWPn = MESolver(nDim, vP0, mWx, 0.0001)
        vPassive[i] = CalcPassivePartialEntropyProdRate(mWx, vPiX)
        # Informed partial entropy production rate
        vInformed[i] = CalcInformedPartialEntropyProdRate(mWx, vPiX, vPiSt)
        # The full entropy rate
        vFull[i] = EntropyRateCalculation(nDim, mWx, vPiX)
        i += 1
    # %% plot
    plt.plot(vGrid, vInformed, 'r-.')
    plt.plot(vGrid, vPassive, 'b--')
    plt.plot(vGrid, vFull, 'y')
    plt.xlabel('x - Driving Force')
    plt.ylabel('Entropy Production rate')
    plt.legend(['Informed', 'Passive', 'Total - Full Trajectory'])
