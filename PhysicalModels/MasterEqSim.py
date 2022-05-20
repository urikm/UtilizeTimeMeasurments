# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 12:15:11 2020

@title: Master equation simulation 

@author: Uri Kapustin

@description: This module solves the master equation for constant rate matrix
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt 
from PhysicalModels.UtilityTraj import MasterEqStep
from Utility.Params import GenRateMat
# %% MasterEq solver for defined dimension. Can use generated or input rate mat 
def MasterEqSolver(nDim, vP0, *args):
    # NOTE: 1st *args should be mW and it's optional! 
    # NOTE: 2nd *args should be timejump and it's optional! 
    
    ## Init Sim
    vPn = vP0

    if len(args)==0:
        # Init adjacency matrix
        mW = GenRateMat(nDim)
        timeJump = 0.01
    elif len(args)==1:
        mW = args[0]
        timeJump = 0.01
    elif len(args)==2:
        mW = args[0]
        timeJump = args[1]


    ## Define stopping criteria
    stopThreshold = 1e-8
    
    nMaxIters = 5e3
    vWPn = np.zeros(int(nMaxIters))  #, dtype=np.float32)
    n = 0
    
    ## Run numeric solution of the Master Equation
    while n < nMaxIters and np.amax(np.abs(np.dot(mW,vPn))) > stopThreshold:
        n += 1
        vPn = MasterEqStep(mW,vPn,timeJump)
        vWPn[n-1] = np.amax(np.abs(np.dot(mW,vPn))) # convergence indicator
    return n, vPn, mW, vWPn



# %% Print results for MasterEqSolver
def PrintMasterEqRes(n, vPn,vP0, mW, vWPn):
    print('################ Master Equation Solver #############')
    print('Num of iters:',n-1)
    print('Initial State:(P0):',vP0)
    print('Final state:(Pn)',vPn)
    print('Norm of W*Pn:',np.linalg.norm(np.dot(mW,vPn)))
    print('#####################################################\n\n')
# %% Plots for MasterEqSolver
def PlotMasterEqRes(vWPn):
    fig1,ax1 = plt.subplots()
    ax1.plot(vWPn[vWPn>0])
    ax1.set_xlabel('# of ME iteration')
    ax1.set_ylabel('max(abs(W*P_{steady-state}))')
    
# %% Unit-test for master equation solver
if __name__ == "__main__":
    ## UI
    nDim = 4 # Dimension of the problem
    typeP0 = 0 # How to choose initial state
    # 0 - from random distribution
    # 1 - [1, 0, ...., 0]
    # sample initial probability
    if typeP0 == 0:
        vP0 = np.random.uniform(size=(nDim))
        vP0 = vP0/sum(vP0)
    elif typeP0 == 1:
        vP0 = np.zeros(nDim)
        vP0[1] = 1
        
    n,vPn,mW,vWPn = MasterEqSolver(nDim, vP0)    
    PrintMasterEqRes(n, vPn,vP0, mW, vWPn)
    PlotMasterEqRes(vWPn)