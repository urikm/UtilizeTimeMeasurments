# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:23:27 2020

@title: Continuous Time Markov Chain (CTMC)
    
@author: Uri Kapustin

@description: Creates trajectory of CTMC according to given rate matrix and initial condition
"""
# %% Imports
import time
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.stats as stats
from Utility.Params import GenRateMat
from PhysicalModels.UtilityTraj import EntropyRateCalculation, MapStates2Indices
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
from PhysicalModels.MasterEqSim import PrintMasterEqRes
from numba import njit

# %% Create single trajectory
@njit
def CreateTrajectory(nDim, nTimeStamps, initState, *args):
    # NOTE: 1st *args should be mW and it's optional!
    ## Inits   
    if len(args) == 0:
        # Init adjacency matrix
        mW = GenRateMat(nDim)
        mcFlag = True
    elif len(args) == 1:
        mW = args[0]
        mcFlag = True
    elif len(args) == 2:
        mW = args[0]
        mcFlag = args[1]

    mDiag = np.multiply(np.eye(nDim), mW)  #For competability with jit
    mP = (mW-mDiag)/np.abs(np.dot(mDiag, np.ones(4)))  # calculate discrete PDF for states jumps
    mTrajectory = np.zeros((nTimeStamps, 2))

    currState = initState
    # memory for indices for each state
    mMem = np.zeros((int(nDim), int(nTimeStamps)))
    vCount = np.zeros(nDim)
    
    # For loop to create trajectory (according to Gillespie Algorithm)
    for iStep in range(nTimeStamps):
        # Using the calculated PDF randomize jump
        if not mcFlag:
            pass
            # nextState = np.random.choice(nDim,1,p=mP[:,currState])
            # nextState = np.array(rd.choices(range(nDim),weights=mP[:,currState].reshape(nDim)))
            ### from neep paper
        else:
            mc = np.random.uniform(0.0, 1.0)
            interval = np.cumsum(mP[:, currState])
            nextState = np.sum(interval < mc)

    ### This implementation is jit competible - if removing jit - DONT USE THIS!
        mTrajectory[iStep, 1] = np.random.exponential(1 / abs(mW[currState, currState]))
    ###

    # save jumping step    
        mTrajectory[iStep, 0] = currState
        mMem[currState, int(vCount[currState])] = iStep
        vCount[currState] = vCount[currState] + 1
    # update current state
        currState = nextState

    # ### in case of not using jit - it should be used this way!
    # # randomize waiting times   - optimized without using jit!!!
    # for iState in range(nDim):
    #     nCounts = int(vCount[iState])
    #     waitingTime = np.random.exponential(1/abs(mW[iState,iState]), nCounts)
    #     idx = np.int32(mMem[iState, 0:nCounts])
    #     mTrajectory[idx, 1] = waitingTime
    # ###s

    return mTrajectory, mW


# %% Estimate parameters of trajectory(or part of it)
def EstimateTrajParams(mTraj, states2Omit=[]):
    # Load mapping from states to indices
    vStates = np.unique(mTraj[:, 0])
    vStates, dMap = MapStates2Indices(vStates, states2Omit=states2Omit)
    nDim = vStates.size

    # Inits
    mIndStates = []
    mWaitTimes = []
    vEstLambdas = np.zeros(nDim)
    totWaitTime = 0

    # For each state - collect statistics and estimate params
    for iState in vStates:
        mIndStates.extend(np.where(mTraj[:, 0] == iState))  # Indices for each state
        mWaitTimes.append(mTraj[mIndStates[dMap[iState]], 1])  # Waiting times for each state
        vEstLambdas[dMap[iState]] = 1/np.average(mWaitTimes[dMap[iState]])  # Estimated lambda for each state
        totWaitTime += np.sum(mWaitTimes[dMap[iState]])  # accumulate the total time of input trajectory
        
    # Reconstruct rate matrix, column by column
    mWest = np.zeros((nDim, nDim))
    for iState in vStates:
        vTmp = np.array(mIndStates[dMap[iState]])+1
        vTmp = vTmp[:-1]  # Avoid reaching max ind
        for iState2 in vStates:
            if iState2 == iState:  # diagonals are known - lambda_i
                mWest[dMap[iState2], dMap[iState]] = -vEstLambdas[dMap[iState]]
            else:
                vIndTmp = np.where(mTraj[vTmp, 0] == iState2)  # indices of jumping from State->State2
                if np.size(vTmp) != 0:
                    mWest[dMap[iState2], dMap[iState]] = np.size(vIndTmp)/np.size(vTmp)*vEstLambdas[dMap[iState]]
                else:
                    mWest[dMap[iState2], dMap[iState]] = 0

    # Estimate steady-state from trajectory by calculating dwell time on each state
    vSimSteadyState = np.zeros(nDim)
    for iState in vStates:
        vSimSteadyState[dMap[iState]] = sum(mWaitTimes[dMap[iState]])/totWaitTime
    return mIndStates, mWaitTimes, vEstLambdas, mWest, vSimSteadyState


# %% Estimate Entropy rate from Partial(long enough) trajectory in steady-state
def EntropyRateEstimation(mPartTraj, states2Omit=[]):
    mIndStates, mWaitTimes, vEstLambdas, mWest, vEstPi = EstimateTrajParams(mPartTraj, states2Omit=states2Omit)
    
    # Init entropy
    entropy = np.log(vEstPi[int(mPartTraj[0, 0])]/vEstPi[int(mPartTraj[-1:, 0])])
    # Sum the addition of each step to the entropy
    for iT in range(np.size(mPartTraj, 0)-1):
        i_n = int(mPartTraj[iT, 0])  # state of the system in current time stamp
        i_np1 = int(mPartTraj[iT+1, 0])  # state of the system in next time stamp
        entropy += np.log(mWest[i_np1, i_n]/mWest[i_n, i_np1])
    totTime = np.sum(mPartTraj[:, 1])
    estEntropyRate = entropy/totTime
    return estEntropyRate
     
    
# %% Prints for single trajectory analysis and statistics
def PrintSingleTraj(vP0, mW, vEstLambdas, vSimSteadyState, vMeSteadyState, vEstSteadyState, nDim):
    print('################ Create Trajectory ##################')
    # Print Ground-Truth and estimated from trajectory rates
    for iState in range(nDim):
        print('Theoretic Rate State ', iState, ':', abs(mW[0, 0]), ' Estimated:', vEstLambdas[iState])
    # Compare Steady state calculated using Master Equation to one estimated from simulation
    print('\n\nMaster Eq SS(Steady state):', vMeSteadyState)
    print('SS from simulation:', vSimSteadyState)
    print('ME SS result with estimated W:', vEstSteadyState)  # Compare acheived steady-state from estimated W
    print('\nRMS of SS elements:', np.sqrt(sum((vMeSteadyState-vEstSteadyState)**2)/nDim))
    print('Maximal error for all dimensions:', np.max(abs(vMeSteadyState-vEstSteadyState)))
    print('#####################################################\n\n')


# %% Plots for single trajectory analysis and statistics
def PlotSingleTraj(mTrajectory,mPartTraj,mW,mWaitTimes,vMeSteadyState,nDim):
    # # plot the chosen portion of trajectory
    # fig0, ax0 = plt.subplots()
    # startTime = np.cumsum(mTrajectory[1:startInd+1,1])
    # startTime = startTime[-1:]
    # ax0.plot(startTime+np.cumsum(mPartTraj[:,1]),mPartTraj[:,0]) 
    
    # Plot Convergence of Estimated Steady state
    nFullTraj = np.size(mTrajectory, 0)
    gtEntropyRate = EntropyRateCalculation(nDim, mW, vMeSteadyState)
    vSample4Integration = np.logspace(3, np.log10(nFullTraj),num=int(np.log10(nFullTraj))+1)
    vError = np.zeros(np.shape(vSample4Integration))
    for iRun in range(np.size(vSample4Integration)):
        estEntropyRate = EntropyRateEstimation(mTrajectory[:int(vSample4Integration[iRun]), :])
        vError[iRun] = abs(gtEntropyRate-estEntropyRate)/gtEntropyRate*100
    fig0, ax0 = plt.subplots()
    ax0.plot(vSample4Integration, vError)
    ax0.set_xlabel('Length of Trajectory[#Samples]')
    ax0.set_ylabel('Entropy rate relative error[%]')   
    ax0.set_xscale('log')
    ax0.axes.set_yticks(np.arange(0, max(vError), 2.0))
    
    # Create figures and utility for exponential PDF comparison
    if nDim <= 10: # avoid ugly and problematic graphs
        nRows = int(np.ceil(nDim/2))
        fig1,axes = plt.subplots(nrows=nRows,ncols=2)
        fig1.suptitle('Measured waiting time PDF vs Theoretic for each state')
        x = np.linspace(stats.expon.ppf(0.0001),stats.expon.ppf(0.9999), 1000) # used for exp pdf
        for iCol in range(2):
            for iRow in range(nRows):  # plot subplots for each state's pdf
                nCurrState = nRows*(iCol)+iRow
                axes[iRow][iCol].hist(mWaitTimes[nCurrState],density=True,label=['Simulated Waiting Times State',
                                                                                 nCurrState])
                axes[iRow][iCol].plot(x, stats.expon.pdf(x, 0, 1/abs(mW[nCurrState, nCurrState])), 'r-', lw=5,
                                      alpha=0.6, label='Theoretic pdf')
                axes[iRow][iCol].legend()
                axes[iRow][iCol].set_xlabel('time[s]')
    
                

            
# %% Analysis on single trajectory
if __name__ == '__main__':
    ## UI
    flagPlot = True
    nDim = 4 # dimension of the problem
    nTimeStamps = int(1e7) # how much time stamps will be saved
    initState = rd.randrange(nDim) # Define initial state in T=0
    if 1:
        mW = GenRateMat(nDim) # transition matrix
        timeStamp = 1
    else:
        mW = np.array([[-11., 1., 0., 7.], [9., -11., 10., 1.], [0., 4., -15., 8.],[2., 6., 5., -16.]])
        timeStamp = 0.001
    tic = time.time()
    mTrajectory, mW = CreateTrajectory(nDim, nTimeStamps, initState, mW) # Run Create Trajectory
    toc = time.time()
    print("Create Trajectory of length: "+str(nTimeStamps) +" in: "+str(toc-tic)+"s")
    ## Start of analysing portion of the trajectory
    startInd = 0
    endInd = nTimeStamps
    mPartTraj = mTrajectory[startInd:endInd, :]
    
    # Estimate Parameters from partial trajectory
    mIndStates, mWaitTimes, vEstLambdas, mWest, vSimSteadyState = EstimateTrajParams(mPartTraj)

    # Check if simulated steady state match the steady state which deriven from MasterEquation
    vP0 = np.zeros(nDim)
    vP0[initState] = 1
    n, vMeSteadyState, mWout, vWPn = MESolver(nDim, vP0, mW, timeStamp)   # steady state deriven by master equation
    PrintMasterEqRes(n, vMeSteadyState, vP0, mWout, vWPn) # Print ME solver results
    
    # Compare achieved steady-state from estimated W
    nEst, vEstSteadyState, mWestOut, vWestPn = MESolver(nDim, vP0, mWest, timeStamp)

    # Prints Analysis output
    PrintSingleTraj(vP0, mW, vEstLambdas, vSimSteadyState, vMeSteadyState, vEstSteadyState, nDim)
    
    # Plot Analysis
    if flagPlot:
        PlotSingleTraj(mTrajectory, mPartTraj, mW, mWaitTimes, vMeSteadyState, nDim)
        
    # Entropy rate analysis
    gtEntropyRate = EntropyRateCalculation(nDim, mW, vMeSteadyState)
    estEntropyRate = EntropyRateEstimation(mPartTraj)
    print('Ground-Truth(Known rate matrix and steady state) Entropy rate: ', gtEntropyRate)
    print('Estimated(No a-priori knowledge) Entropy rate: ', estEntropyRate)
    print('Relative Error: ', abs(gtEntropyRate-estEntropyRate)/gtEntropyRate*100, '%')
