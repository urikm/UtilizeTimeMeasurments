# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 10:30:51 2021

@author: Uri
"""

# TODO : delete
import numpy as np
import os
import pickle

import PhysicalModels.PartialTrajectories as pt 




# Base rate matrix
nDim=4
nCgDim=3
vHiddenStates = np.array([2,3]) # states 3 and 4 for 4-D state syte
nTimeStamps = int(128*4096*1e2) # as used in RunRNeepAnalysis
timeRes = 0.001
mW = np.array([[-11.,2.,0.,1.],[3.,-52.2,2.,35.],[0.,50.,-77.,0.7],[8.,0.2,75.,-36.7]])
   
    
dbName = 'RneepDbStalling'#'RneepDbCoarse'
dbPath = 'StoredDataSets'+os.sep+dbName
dbFileName = 'InitRateMatAsGilis'

# Choose the wanted trajectory according to x
with open(dbPath+os.sep+'MappingVector'+'.pickle', 'rb') as handle:
    vX = pickle.load(handle)

for idx,x in enumerate(vX):
    with open(dbPath+os.sep+dbFileName+'_'+str(idx)+'.pickle', 'rb') as handle:
        dDataTraj = pickle.load(handle)    
        mWx = pt.CalcW4DrivingForce(mW,x) # Calculate W matrix after applying force
        # Passive partial entropy production rate
        vP0 = np.random.uniform(size=(nDim))
        vP0 = vP0/sum(vP0)
        mCgTrajectory = dDataTraj.pop('vStates')
        mCgTrajectory = np.array([mCgTrajectory,dDataTraj.pop('vTimeStamps')]).T
        sigmaDotKld, T, sigmaDotAff, sigmaWtd = pt.CalcKLDPartialEntropyProdRate(mCgTrajectory, vHiddenStates)
        dDataTraj['vStates'] = mCgTrajectory[:,0]
        dDataTraj['vTimeStamps'] = mCgTrajectory[:,1]
        dDataTraj['kldBound'] = sigmaDotKld
    with open(dbPath+os.sep+dbFileName+'_'+str(idx)+'.pickle', 'w+b') as handle:
        pickle.dump(dDataTraj, handle)
