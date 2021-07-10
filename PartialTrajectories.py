# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:30:24 2020

@title: Partial Information from Trajectories Analyis(Reproduction of heirarchial bounds paper from 2017)
    
@author: Uri Kapustin
"""

# %% Imports
import numpy as np
import random as rd
import matplotlib.pyplot as plt
# import scipy.stats as stats
from MasterEqSim import MasterEqSolver as MESolver
from TrajectoryCreation import *


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