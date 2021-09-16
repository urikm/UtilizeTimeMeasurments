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
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
from PhysicalModels.PartialTrajectories import CalcPassivePartialEntropyProdRate,CalcInformedPartialEntropyProdRate,CalcStallingData,CalcW4DrivingForce
from PhysicalModels.UtilityTraj import  CgRateMatrix,EntropyRateCalculation

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
subResults = 'Analysis_RNEEP_Stalling_21_07_28'
prefixRunName = 'Analysis'
pathRes = 'Results'+os.sep+subResults
nRuns = 2
# -----------Grid---------------- 
# For the With w/o time data folders :
vGrid = np.concatenate((np.arange(-1.,xSt,1),np.arange(xSt,3.,1)))
vGrid = np.arange(-3,xSt,1)
nLast = np.size(vGrid)-1
vMask = np.arange(nLast+1) #[1,2,3,4,5,7,8,9,10,11] # Can be used to plot not every point on grid
# vMask = [1,2,3,4,5,7,8,9,10,11] 
vGrid = np.take(vGrid,vMask)
vGridInterp = np.concatenate((np.arange(-1.,xSt,gridInterpRes),np.arange(xSt,3.,gridInterpRes)))#np.linspace(vGrid[0],vGrid[-1:],num=1000)




# %% Calculate analytic boundries
vFull = np.zeros(np.size(vGridInterp))
vInformed = np.zeros(np.size(vGridInterp))
vPassive = np.zeros(np.size(vGridInterp))
mKld = np.zeros((np.size(vGrid),nLast+1))
mNeep = np.zeros((5,np.size(vGrid),nLast+1)) # 5 is the number of different sequence size that inputed

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
print("Plotting RNEEP results without time data")
specialPath = '_x_'+str(nLast)
# with open(pathRes+os.sep+'vPassive'+specialPath+'.pickle', 'rb') as handle:
#     vPassive = pickle.load(handle)
#     vPassive = np.take(vPassive,vMask)
# with open(pathRes+os.sep+'vInformed'+specialPath+'.pickle', 'rb') as handle:
#     vInformed = pickle.load(handle)
#     vInformed = np.take(vInformed,vMask)
for iRun in range(nRuns):
    with open(pathRes+os.sep+prefixRunName+'_'+str(iRun+1)+os.sep+'vKld'+specialPath+'.pickle', 'rb') as handle:
        vKld = pickle.load(handle)
        mKld[:,iRun] = vKld
    with open(pathRes+os.sep+prefixRunName+'_'+str(iRun+1)+os.sep+'mNeep'+specialPath+'.pickle', 'rb') as handle:
        mNeeptmp = pickle.load(handle)
        mNeep[:,:,iRun] = mNeeptmp
    
# Create mean and std data for plotting
vKldMean = np.mean(mKld,axis=1)
vKldStd = np.std(mKld,axis=1)
mNeepMean = np.mean(mNeep,axis=2)
mNeepStd = np.std(mNeep,axis=2)
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

plt.errorbar(np.flip(-vGrid),np.flip(vKldMean),yerr=np.flip(vKldStd))    

plt.errorbar(np.flip(-vGrid),np.flip(mNeepMean[0,:]),yerr=np.flip(mNeepStd[0,:])) 
plt.errorbar(np.flip(-vGrid),np.flip(mNeepMean[1,:]),yerr=np.flip(mNeepStd[1,:])) 
plt.errorbar(np.flip(-vGrid),np.flip(mNeepMean[2,:]),yerr=np.flip(mNeepStd[2,:])) 
plt.errorbar(np.flip(-vGrid),np.flip(mNeepMean[3,:]),yerr=np.flip(mNeepStd[3,:])) 
plt.errorbar(np.flip(-vGrid),np.flip(mNeepMean[4,:]),yerr=np.flip(mNeepStd[4,:])) 
 

plt.yscale('log')
plt.xlabel('x - Driving Force')
plt.ylabel('Entropy Production rate')
plt.legend(['Full','Informed','Passive','KLD','NEEP-seq3','NEEP-seq16','NEEP-seq32','NEEP-seq64','NEEP-seq128'])
plt.show()


# %% Plot by sequence
for k in np.arange(len(vGrid)):
    plt.figure(k+1)
    plt.plot(np.array([3,16,32,64,128]),(mNeep[:,k])) 
 

    plt.yscale('log')
    plt.xlabel('input sequence size')
    plt.ylabel('Entropy Production rate')
    plt.title(['External force x-',str(vGrid[k])])
    plt.show()