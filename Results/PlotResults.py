# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:46:37 2021

@title : plot results

@author: Yuri
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from TrajectoryCreation import *
from PartialTrajectories import  * 

# %% Init
mW = np.array([[-11.,2.,0.,1.],[3.,-52.2,2.,35.],[0.,50.,-77.,0.7],[8.,0.2,75.,-36.7]])
vPiSt,xSt,r01,r10  = CalcStallingData(mW)    
nDim = 4
vHiddenStates = np.array([2,3]) # Assuming 2, arbitrary chosen, states are observed as 1
nCgDim = nDim - (np.size(vHiddenStates)-1)
timeRes = 0.001
gridInterpRes = 0.2

vPiStCg = np.zeros(nCgDim)
vPiStCg[0:2] = vPiSt[0:2]
vPiStCg[2] = vPiSt[2]+vPiSt[3]
mWCg = CgRateMatrix(mW,vHiddenStates)
    
# %% UI
plotRneepFlag = True
plotRneepTFlag = False

if plotRneepFlag:
    pathWoTime = 'RNEEP_21_07_04'#'RNEEP_21_05_27' # Example, you should the wanted recording for plot
if plotRneepTFlag:
    pathWiTime = 'RNEEPT_21_06_27'#'RNEEP with time data' # Example, you should the wanted recording for plot
     
# -----------Grid---------------- 
# For the With w/o time data folders :
vGrid = np.concatenate((np.arange(-1.,xSt,1),np.arange(xSt,3.,1)))
# vGrid = np.array([-1])
# vGrid = np.arange(-1.,xSt,1)
# For record from 21_05_17:
# vGrid = np.concatenate((np.arange(xSt-0.2,xSt,0.1),np.arange(xSt,xSt+0.2,0.1))) 

# For record from 21_05_20
# vGrid = np.concatenate((np.arange(xSt-0.05,xSt,0.01),np.arange(xSt,xSt+0.05,0.01)))

# For record from 21_05_27
# vGrid = np.concatenate((np.arange(xSt-0.05,xSt-0.005,0.01),np.arange(xSt,xSt+0.05,0.01)))

# For record from 21_06_19 like for 21_05_27 just the run abrupted
# vGrid = np.concatenate((np.arange(xSt-0.05,xSt-0.005,0.01),np.arange(xSt,0.7,0.01)))
# -------------------------------
nLast = np.size(vGrid)-1
vMask = np.arange(nLast+1) #[1,2,3,4,5,7,8,9,10,11] # Can be used to plot not every point on grid
# vMask = [1,2,3,4,5,7,8,9,10,11] 
vGrid = np.take(vGrid,vMask)
vGridInterp = np.concatenate((np.arange(-1.,xSt,gridInterpRes),np.arange(xSt,3.,gridInterpRes)))#np.linspace(vGrid[0],vGrid[-1:],num=1000)




# %% Calculate analytic boundries
vFull = np.zeros(np.size(vGridInterp))
vInformed = np.zeros(np.size(vGridInterp))
vPassive = np.zeros(np.size(vGridInterp))
# vKinetic = np.zeros(np.size(vGridInterp))

# Calculate full entropy rate
i=0
for x in vGridInterp: 
    mWx = CalcW4DrivingForce(mW,x) # Calculate W matrix after applying force
    # Passive partial entropy production rate
    vP0 = np.random.uniform(size=(nDim))
    vP0 = vP0/sum(vP0)
    n,vPiX,mWx,vWPn = MESolver(nDim,vP0,mWx,timeRes)
    mWCgx = CgRateMatrix(mWx,vHiddenStates)
    vPiXCg = np.zeros(nCgDim)
    vPiXCg[0:2] = vPiX[0:2]
    vPiXCg[2] = vPiX[2]+vPiX[3]
    # vPassive[i] = CalcPassivePartialEntropyProdRate(mWx,vPiX)
    # Informed partial entropy production rate
    # vInformed[i] = CalcInformedPartialEntropyProdRate(mWx,vPiX,vPiSt)
    # The full entropy rate
    vFull[i] = EntropyRateCalculation(nDim,mWx,vPiX)
    vInformed[i] = CalcInformedPartialEntropyProdRate(mWCgx,vPiXCg,vPiStCg)
    vPassive[i] = CalcPassivePartialEntropyProdRate(mWCgx,vPiXCg)
    # vKinetic[i] = CalcKineticBoundEntProdRate(mWCgx,vPiXCg)
    i+=1
    
# %% Read data
# NOTE : data is recorded after each 'x' - force, thus after each iteration data is incremented and not overriden so its sufficient to read the data for the last applied force
# RNEEP w/o time data
if plotRneepFlag:
    print("Plotting RNEEP results without time data")
    specialPath = '_x_'+str(nLast)
    # with open(pathWoTime+os.sep+'vPassive'+specialPath+'.pickle', 'rb') as handle:
    #     vPassive = pickle.load(handle)
    #     vPassive = np.take(vPassive,vMask)
    # with open(pathWoTime+os.sep+'vInformed'+specialPath+'.pickle', 'rb') as handle:
    #     vInformed = pickle.load(handle)
    #     vInformed = np.take(vInformed,vMask)
    with open(pathWoTime+os.sep+'vKld'+specialPath+'.pickle', 'rb') as handle:
        vKld = pickle.load(handle)
        vKld = np.take(vKld,vMask)
    with open(pathWoTime+os.sep+'mNeep'+specialPath+'.pickle', 'rb') as handle:
        mNeep = pickle.load(handle)
        mNeep = np.take(mNeep,vMask,axis=1)
    
# RNEEP with time data
if plotRneepTFlag:    
    print("Plotting RNEEP results with time data")    
    specialPath = '_x_T_'+str(nLast)
    # with open(pathWiTime+os.sep+'vPassive'+specialPath+'.pickle', 'rb') as handle:
    #     vPassive = pickle.load(handle)
    #     vPassive = np.take(vPassive,vMask)
    # with open(pathWiTime+os.sep+'vInformed'+specialPath+'.pickle', 'rb') as handle:
    #     vInformed = pickle.load(handle)
    #     vInformed = np.take(vInformed,vMask)
    with open(pathWiTime+os.sep+'vKld'+specialPath+'.pickle', 'rb') as handle:
        vKld = pickle.load(handle)
    # with open(pathWiTime+os.sep+'vKld'+'_x_T_2'+'.pickle', 'rb') as handle:
    #     vKld2 = pickle.load(handle)  
    #     vKld = np.concatenate((vKld2,vKld))
    vKld = np.take(vKld,vMask)
    # TODO : its temp
    # with open(pathWiTime+os.sep+'mNeep'+'_x_T_2'+'.pickle', 'rb') as handle:
    #     mNeepT1 = pickle.load(handle)    
    with open(pathWiTime+os.sep+'mNeep'+specialPath+'.pickle', 'rb') as handle:
        mNeepT2 = pickle.load(handle) 
    # mNeepT = np.concatenate((mNeepT1,mNeepT2),1)
    mNeepT = mNeepT2
    mNeepT = np.take(mNeepT,vMask,axis=1)
# %% Plot 
# TMP! TODO: delete
vInformed[int(np.size(vGridInterp)/2)-1]=2*vPassive[int(np.size(vGridInterp)/2)-1]
# vInformed[5]=2*vPassive[5] # only for record from 19_06_21
#
plt.figure(0)
plt.plot(np.flip(-vGridInterp),np.flip(vFull),'r') 
plt.plot(np.flip(-vGridInterp),np.flip(vInformed),':k')
plt.plot(np.flip(-vGridInterp),np.flip(vPassive),':g')
# plt.plot(np.flip(-vGridInterp),np.flip(vKinetic),':c')

plt.plot(np.flip(-vGrid),np.flip(vKld),'o:b')    
if plotRneepFlag:
    plt.plot(np.flip(-vGrid),np.flip(mNeep[0,:]),'xm') 
    plt.plot(np.flip(-vGrid),np.flip(mNeep[1,:]),'xc') 
    plt.plot(np.flip(-vGrid),np.flip(mNeep[2,:]),'xr') 
    plt.plot(np.flip(-vGrid),np.flip(mNeep[3,:]),'xy') 
    plt.plot(np.flip(-vGrid),np.flip(mNeep[4,:]),'xk') 
if plotRneepTFlag:
    plt.plot(np.flip(-vGrid),np.flip(mNeepT[0,:]),'*m') 
    plt.plot(np.flip(-vGrid),np.flip(mNeepT[1,:]),'*c') 
    plt.plot(np.flip(-vGrid),np.flip(mNeepT[2,:]),'*r') 
    plt.plot(np.flip(-vGrid),np.flip(mNeepT[3,:]),'*y') 
    plt.plot(np.flip(-vGrid),np.flip(mNeepT[4,:]),'*k')    

plt.yscale('log')
plt.xlabel('x - Driving Force')
plt.ylabel('Entropy Production rate')
if plotRneepFlag and plotRneepTFlag:
    plt.legend(['Full','Informed','Passive','KLD','NEEP-seq3','NEEP-seq16','NEEP-seq32','NEEP-seq64','NEEP-seq128','NEEPT-seq3','NEEPT-seq16','NEEPT-seq32','NEEPT-seq64','NEEPT-seq128'])
elif plotRneepFlag and not plotRneepTFlag:
    plt.legend(['Full','Informed','Passive','KLD','NEEP-seq3','NEEP-seq16','NEEP-seq32','NEEP-seq64','NEEP-seq128'])
else:
    plt.legend(['Full','Informed','Passive','KLD','NEEPT-seq3','NEEPT-seq16','NEEPT-seq32','NEEPT-seq64','NEEPT-seq128'])
plt.show()


# %% Plot by sequence
for k in np.arange(len(vGrid)):
    plt.figure(k+1)
    if plotRneepFlag:
        plt.plot(np.array([3,16,32,64,128]),(mNeep[:,k])) 
    if plotRneepTFlag:
        plt.plot(np.array([3,16,32,64,128]),(mNeepT[:,k]))   

    plt.yscale('log')
    plt.xlabel('input sequence size')
    plt.ylabel('Entropy Production rate')
    plt.title(['External force x-',str(vGrid[k])])
    plt.show()