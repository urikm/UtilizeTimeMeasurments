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
from sklearn.model_selection import GridSearchCV,LeaveOneOut

from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
from PhysicalModels.TrajectoryCreation import CreateTrajectory,EstimateTrajParams
from PhysicalModels.UtilityTraj import CalcSteadyStateCurrent, EntropyRateCalculation
from numba import njit

# %% Assuming only states 0,1 are observable. TODO : maybe make more generic?

# Calculate passive entropy production rate as explained in 2017 hierarchical bounds paper
def CalcPassivePartialEntropyProdRate(mW, vPi):
    # This function used for informed and passive rates when expecting the right 'W' and 'Pi' state for each rate
    nDim = np.size(vPi)
    sigmaPP = 0
    # Use pairs of state where i<j, as described in paper
    for iState in range(1):#range(nDim):     
        for jState in range(1,2):#range(iState+1,nDim):
            Jij,J_j2i,J_i2j = CalcSteadyStateCurrent(mW,vPi,iState,jState)# Calculate Current between two states
            if Jij != 0:
                sigmaPP += Jij*np.log((mW[iState,jState]*vPi[jState])/(mW[jState,iState]*vPi[iState]))
    return sigmaPP

# Calculate informed entropy production rate as explained in 2017 hierarchical bounds paper
def CalcInformedPartialEntropyProdRate(mW,vPi,vPiSt):
    # This function used for informed and passive rates when expecting the right 'W' and 'Pi' state for each rate    
    nDim = np.size(vPi)
    sigmaIP = 0
    # Use pairs of state where i<j, as described in paper
    for iState in range(1):#range(nDim):   
        for jState in range(1,2):#range(iState+1,nDim):
            Jij,J_j2i,J_i2j = CalcSteadyStateCurrent(mW,vPi,iState,jState)# Calculate Current 0->1
            if Jij != 0:
                sigmaIP += Jij*np.log((mW[iState,jState]*vPiSt[jState])/(mW[jState,iState]*vPiSt[iState]))
    return sigmaIP

# Calculate stalling steady state and independet rates as explained in 2017 hierarchical bounds paper Appendix B
def CalcStallingData(mW):
    # Calcualte the needed determinants
    mTmp = mW[(0,2,3),:]
    mTmp = mTmp[:,(1,2,3)]
    detWno10 = np.linalg.det(mTmp)
    mTmp2 = mW[(1,2,3),:]
    mTmp2 = mTmp2[:,(0,2,3)]
    detWno01 = np.linalg.det(mTmp2)
    detWno01Atall = np.linalg.det(mW[2:,2:])
    
    # Calculate stalling steady state 
    vPiSt,mWst = CalcStallingSteadyState(mW)
    
    # Calculate independent rates
    r01 = detWno10/detWno01Atall
    r10 = detWno01/detWno01Atall
    
    # Check if calculated steady state hold the independant rate ratio
    assert (np.abs((vPiSt[1]/vPiSt[0])-(r10/r01)) < 1e1), "Calculated Stalling Steady State is wrong"
        
    # Calculate stalling force
    xSt = CalcStallingForce(mW,vPiSt)
    
    return vPiSt,xSt,r01,r10

# Calculate stalling Force as explained in 2017 hierarchical bounds paper Simulation
def CalcStallingForce(mW,vPiSt):
    xSt = -0.5*np.log((mW[1,0]*vPiSt[0])/(mW[0,1]*vPiSt[1]))
    return xSt

# Calculate stalling steady state by manually setting to zero rates
def CalcStallingSteadyState(mW):
    mWst= np.copy(mW)
    # Update diagonal elements according to the new zero rates
    mWst[0,0] = mWst[0,0] + mWst[1,0]
    mWst[1,1] = mWst[1,1] + mWst[0,1]
    
    # Set to zero wanted rates
    mWst[0,1]=0
    mWst[1,0]=0
    
    nDim = np.size(mWst,0)
    vP0 = np.random.uniform(size=(nDim))
    vP0 = vP0/sum(vP0)
    n,vPiSt,mWst,vWPn = MESolver(nDim,vP0,mWst,0.01)
    return vPiSt,mWst

# Calculate stalling W matrix
def CalcW4DrivingForce(mW,x):
    # assume exponential force over the rate between 0<->1
    mWx = np.array(mW)
 
    # Update diagonal values
    mWx[0,0] += mWx[1,0]*(1-np.exp(x))
    mWx[1,1] += mWx[0,1]*(1-np.exp(-x))  
    
    # Update rates according to driving force
    mWx[1,0]=mWx[1,0]*np.exp(x)
    mWx[0,1]=mWx[0,1]*np.exp(-x)
    
    return mWx


# %% Coarse Grain trajectory
@njit
def CoarseGrainTrajectory(mTrajectory, nFullDim, vHiddenStates, semiCG=False):
    mCgTrajectory = np.copy(mTrajectory)
    nJumps = mCgTrajectory.shape[0]
    iHidden = nFullDim - vHiddenStates.shape[0]
    if semiCG:
        hState = iHidden
    else:
        hState = -2  # iHidden #  define as intermediate state every hidden state that is not last in it's sequence

    for iJump in range(nJumps):

        if (mCgTrajectory[iJump, 0] in vHiddenStates):
            if (iJump < nJumps - 1):
                if (mCgTrajectory[iJump + 1, 0] in vHiddenStates):
                    mCgTrajectory[iJump, 0] = hState
                    mCgTrajectory[iJump + 1, 1] += mCgTrajectory[iJump, 1]  # cumsum waiting times for each sequence
                else:
                    mCgTrajectory[iJump, 0] = iHidden
            else:
                mCgTrajectory[iJump, 0] = iHidden

    if semiCG:
        pass  # Dont do nothing in semi CG
    else:
        mCgTrajectory = mCgTrajectory[(mCgTrajectory[:, 0] != hState), :]  # remove '-2' states in Full CG

    nCgDim = iHidden + 1
    return mCgTrajectory, nCgDim

# Estimate 2nd order statistics of the trajectory (i->j->k transitions)
# TODO: make competible for every combination , now competible only for 2019's paper example
def EstimateTrajParams2ndOrder(nDim,mTrajectory):
    mP2ndOrderTransitions = np.zeros((nDim,) * nDim) # all the Pijk - 3D tenzor
    mWtd = []
    vInitStates = np.arange(nDim)
    for iState in vInitStates:
        vIndInitTrans = np.array(np.where(mTrajectory[:-2,0] == iState)[0])
        nTrans4State = np.size(vIndInitTrans,0)
        vFirstTrans = np.roll(vInitStates,-iState)[1:]
        for jState in vFirstTrans:
            vIndFirstTrans = vIndInitTrans[np.array(np.where(mTrajectory[vIndInitTrans + 1,0] == jState)[0])]+1
            # find which is the k state ( assuming only 3 states!!!)
            kState = np.argwhere((vInitStates!=iState)&(vInitStates!=jState))[0][0]
            vIndSecondTrans = vIndFirstTrans[np.array(np.where(mTrajectory[vIndFirstTrans + 1,0] == kState)[0])]+1
            mP2ndOrderTransitions[iState,jState,kState] = np.size(vIndSecondTrans,0)/(nTrans4State+2)
            if (jState == 2) & (iState != 2) & (kState != 2) & (iState != kState):
                mWtd.append(mTrajectory[vIndSecondTrans-1,1]) 
            
    return mP2ndOrderTransitions, mWtd
# Calculate KLD entropy production rate as explained in 2019  paper
def CalcKLDPartialEntropyProdRate(mCgTrajectory, nDim):
    # First estimate all statistics from hidden trajectory
    mIndStates, mWaitTimes, vEstLambdas, mWest, vSS = EstimateTrajParams(nDim, mCgTrajectory)
    mP2ndOrdTrans, mWtd = EstimateTrajParams2ndOrder(nDim, mCgTrajectory)
    
    # Calculate common paramerters
    vTau = np.zeros(3)
    vTau[0] = np.sum(mWaitTimes[0])/np.size(mWaitTimes[0], 0)
    vTau[1] = np.sum(mWaitTimes[1])/np.size(mWaitTimes[1], 0)
    vTau[2] = np.sum(mWaitTimes[2])/np.size(mWaitTimes[2], 0)
    
    
    vR = np.zeros(3)
    nTot = np.size(mIndStates[0], 0)+np.size(mIndStates[1], 0)+np.size(mIndStates[2], 0)
    vR[0] = np.size(mIndStates[0], 0)/nTot
    vR[1] = np.size(mIndStates[1], 0)/nTot
    vR[2] = np.size(mIndStates[2], 0)/nTot
    
    T = np.dot(vTau,vR)
    assert np.abs(T - np.mean(mCgTrajectory[:, 1])) < 1e-15, "Check Tau factor calculation! its not the mean WTD"
    ## Find affinity part 
    # Math: R12 = p21*R[1] = (tau[1]*w21)*(Pi[1]*T/tau[1])=w21*Pi[1]*T
    R12 = mWest[1, 0]*vSS[0]*T
    R13 = mWest[2, 0]*vSS[0]*T
    R21 = mWest[0, 1]*vSS[1]*T
    R23 = mWest[2, 1]*vSS[1]*T
    R31 = mWest[0, 2]*vSS[2]*T
    R32 = mWest[1, 2]*vSS[2]*T
    # Pijk = Pr{to observe i>j>k} => Pijk=R[ijk]*R[i](this probaibility related to markoc chain, not time related)
    p12_23 = mP2ndOrdTrans[0, 1, 2]*vR[0]/R12
    p23_31 = mP2ndOrdTrans[1, 2, 0]*vR[1]/R23
    p31_12 = mP2ndOrdTrans[2, 0, 1]*vR[2]/R31
    p13_32 = mP2ndOrdTrans[0, 2, 1]*vR[0]/R13
    p32_21 = mP2ndOrdTrans[2, 1, 0]*vR[2]/R32
    p21_23 = mP2ndOrdTrans[1, 0, 2]*vR[1]/R21
    
    sigmaDotAff = ((R12-R21)/T*np.log(p12_23*p23_31*p31_12/p13_32/p32_21/p21_23))
    
    ## Find Wtd part
    p1H2 = mP2ndOrdTrans[0, 2, 1]*vR[0]
    p2H1 = mP2ndOrdTrans[1, 2, 0]*vR[1]
    ## Use KDE to build Psi functions
    # First estimate bandwidths
    # bandwidths = np.linspace(-0.1, 0.1, 20)
    # grid = GridSearchCV(KD(kernel='gaussian'),{'bandwidth': bandwidths},cv=LeaveOneOut())
    # grid.fit(np.int64(mWtd[0][:, None]))
    # b1H2 = grid.best_params_
    # grid.fit(np.int64(mWtd[1][:, None]))
    # b2H1 = grid.best_params_
    b1H2 = 0.0043 # manually fixed after running some optimization, see the lines commented before
    b2H1 = b1H2
    # Define density destribution grid
    vGridDest = np.linspace(0, 0.25, 100)
    # kde1H2 = KD(bandwidth=b1H2['bandwidth'])
    # kde2H1 = KD(bandwidth=b2H1['bandwidth'])
    kde1H2 = KD(bandwidth=b1H2)
    kde2H1 = KD(bandwidth=b2H1)
    kde1H2.fit(mWtd[0][:, None])
    kde2H1.fit(mWtd[1][:, None])
    dd1H2 = np.exp(kde1H2.score_samples(vGridDest[:, None]))  # density distribution 1->H->2
    dd2H1 = np.exp(kde2H1.score_samples(vGridDest[:, None]))  # density distribution 2->H->1
    pDd1H2 = dd1H2/np.sum(dd1H2)  # Probability density distribution
    pDd2H1 = dd2H1/np.sum(dd2H1)  # Probability density distribution
    kld1H2 = np.sum(np.multiply(pDd1H2, np.log(np.divide(pDd1H2, pDd2H1))))
    kld2H1 = np.sum(np.multiply(pDd2H1, np.log(np.divide(pDd2H1, pDd1H2))))
    
    sigmaDotWtd = (p1H2*kld1H2+p2H1*kld2H1)/T
    
    sigmaDotKld = sigmaDotAff + sigmaDotWtd
    return sigmaDotKld, T, sigmaDotAff, sigmaDotWtd, dd1H2, dd2H1

def CreateCoarseGrainedTraj(nDim,nTimeStamps,mW,vHiddenStates,timeRes,semiCG=False,isCG=True):
    # randomize init state from the steady-state distribution
    vP0 = np.array([0.25,0.25,0.25,0.25])  #, dtype=np.float32)
    n,vPi,mW,vWPn = MESolver(nDim,vP0,mW,timeRes)
    normP = vPi.sum() # due to numeric precision problems we need to normalize to 1
    if normP > 0.999:
        vPi = vPi/normP
    else:
        assert 0, "The Master equation solver doesnt converge for this system - you should look at it"
    initState = np.random.choice(nDim,1,p=vPi).item()
    # Create trajectory
    mTrajectory, mW = CreateTrajectory(nDim,nTimeStamps,initState,mW) # Run Create Trajectory
    if isCG:
        mCgTrajectory,nCgDim = CoarseGrainTrajectory(mTrajectory,nDim,vHiddenStates,semiCG=semiCG)
    else:
        mCgTrajectory = mTrajectory
        nCgDim = nDim
    return mCgTrajectory,nCgDim




# %% Analysis Partial Entropy production on single trajectory
if __name__ == '__main__':
    ## UI
    flagPlot = True
    nDim = 4 # dimension of the problem
    # nTimeStamps = int(1e5) # how much time stamps will be saved
    initState = rd.randrange(nDim) # Define initial state in T=0
    # mW = GenRateMat(nDim) # transition matrix
    mW = np.array([[-11.,1.,0.,7.],[9.,-11.,10.,1.],[0.,4.,-15.,8.],[2.,6.,5.,-16.]])
    
    # mTrajectory,mW = CreateTrajectory(nDim,nTimeStamps,initState,mW) # Run Create Trajectory
    # # Calculate Steady state
    # mIndStates,mWaitTimes,vEstLambdas,mWest,vSimSteadyState = EstimateTrajParams(nDim,mTrajectory)       
    # Calculate Stalling data
    vPiSt,xSt,r01,r10  = CalcStallingData(mW)
    
    # Init vectors for plotting
    vGrid = np.arange(-7.,7.,0.1) # TODO make smarter grid, use stalling data
    vInformed = np.zeros(np.size(vGrid))
    vPassive = np.zeros(np.size(vGrid))
    vFull = np.zeros(np.size(vGrid))
    i=0
    for x in vGrid: 
        mWx = CalcW4DrivingForce(mW,x) # Calculate stalling W matrix
        # Passive partial entropy production rate
        vP0 = np.random.uniform(size=(nDim))
        vP0 = vP0/sum(vP0)
        n,vPiX,mWst,vWPn = MESolver(nDim,vP0,mWx,0.0001)
        vPassive[i] = CalcPassivePartialEntropyProdRate(mWx,vPiX)
        # Informed partial entropy production rate
        vInformed[i] = CalcInformedPartialEntropyProdRate(mWx,vPiX,vPiSt)
        # The full entropy rate
        vFull[i] = EntropyRateCalculation(nDim,mWx,vPiX)
        i += 1
# %% plot        
    plt.plot(vGrid,vInformed,'r-.')
    plt.plot(vGrid,vPassive,'b--') 
    plt.plot(vGrid,vFull,'y')   
    plt.xlabel('x - Driving Force')
    plt.ylabel('Entropy Production rate')
    plt.legend(['Informed','Passive','Total - Full Trajectory'])