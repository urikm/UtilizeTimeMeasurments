# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 07:05:53 2021

@title: DataSets Module

@author: Uri Kapustin

@description: This module creates synthetic databases of trajectories according to wanted physical models
"""
# %% TODO : This module need some "dust removal", it probably wont run 

# %% imports

import os
import shutil
import time

import torch
from torch.utils.data import Dataset
import numpy as np

from PhysicalModels.UtilityTraj import EntropyRateCalculation
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
import PhysicalModels.TrajectoryCreation as ct #CreateTrajectory, EstimateTrajParams
import PhysicalModels.PartialTrajectories as pt
from RNeepSampler import CartesianSeqSampler as CSS
from Utility.Params import BaseSystem, DataSetCreationParams, MolecularMotor
# This import is written very quickly
from Results.PlotResultsAnalysisPaper_MM import CreateMMTrajectory

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# TODO : Think about switching from batches samples to normal samples because of Hardware memory.
# TODO : add support for other systems that base system

# %% The class for managing a single coarse grained trajectory. The dataset created for single freeParam setting, e.g for specific external force
# This dataset mainly used for analysing the RNEEP implementation on markov chains with rate matrix from Gili's 2019 paper
class CGTrajectoryDataSet(Dataset):
    def __init__(self, seqLen=32, batchSize=4096, lenTrajFull=1e6, extForce=0., mode='train',
                 rootDir='StoredDataSets', loadSet=False, isBaseSystem=True, semiCG=False):
        # Note: only available modes are : 'train','valid','test'
        assert (mode in ['train', 'valid', 'test'])

        self.DataSetDir = rootDir + os.sep + mode
        self.isBaseSystem = isBaseSystem
        self.extForce = extForce
        self.seqLen = seqLen
        self.batchSize = batchSize
        self.targetKLD = -1  # Initialize target KLD estimation of EPR
        self.targetEPR = -1  # Initialize target full EPR
        self.nBatchedSamples = -1  # Initialize number of batched samples - the length of the dataset
        self.timeFactor = -1 # Initialize the time factor in ourder to move from 'per jump' 'per sec'

        lenTrajFull = int(lenTrajFull) # Assure that only int values passed

        if not loadSet:  # save the data
            if isBaseSystem:
                mW, nDim, _, timeRes = BaseSystem()
                # Calculate target EPR of the full system
                mWx = pt.CalcW4DrivingForce(mW, self.extForce)  # Calculate W matrix after applying force
                vP0 = np.random.uniform(size=(nDim))
                vP0 = vP0 / sum(vP0)
                _, vPiX, _, _ = MESolver(nDim, vP0, mWx, timeRes)
                self.targetEPR = EntropyRateCalculation(nDim, mWx, vPiX)
                # Create and save dataset. Updates targerKLD and nBatchedSamples attributes
                self.targetKLD, self.nBatchedSamples, self.timeFactor = \
                    self.CreateRNeepDataSet(lenTrajFull, mode, self.DataSetDir, semiCG=semiCG)
            # Save Dataset information for future loading
        else:
            # Read data set description
            # TODO : make this part more scalable
            _, fileSuffix, _, dsDescriptorName, _ = DataSetCreationParams()

            dDataSet = torch.load(self.DataSetDir + os.sep + dsDescriptorName + fileSuffix)
            self.seqLen = dDataSet['seqLen']
            self.batchSize = dDataSet['batchSize']
            self.isBaseSystem = dDataSet['isBaseSystem']
            self.extForce = dDataSet['extForce']
            self.targetEPR = dDataSet['targetEPR']
            self.targetKLD = dDataSet['targetKLD']
            self.nBatchedSamples = dDataSet['nBatchedSamples']
            self.timeFactor = dDataSet['timeFactor']

    def __len__(self):
        return self.nBatchedSamples

    def __getitem__(self, idx):
        samplePrefix, fileSuffix, _, _, _ = DataSetCreationParams()
        batchedSample = torch.load(self.DataSetDir + os.sep + samplePrefix + str(int(idx)) + fileSuffix)
        FullEPR = self.targetEPR
        KldEPR = self.targetKLD
        return batchedSample, FullEPR, KldEPR


    # The function creates and stores offline the trajectory by batches
    def CreateRNeepDataSet(self, lenTrajFull, mode, saveDir, semiCG=False):
        # The Dataset will be created for specific forces.
        # The Dataset is saved by BATCHES and NOT single SAMPLES

        # Get common params for dataset
        samplePrefix, fileSuffix, trainIterInEpoch, dsDescriptorName, trajFileName = DataSetCreationParams()

        # Base rate matrix
        mW, nDim, vHiddenStates, timeRes = BaseSystem()

        # Dataset descriptor
        dTrajTemplate = {'seqLen': self.seqLen, 'batchSize': self.batchSize, 'nBatchedSamples': -1,
                         'extForce': self.extForce, 'targetEPR': self.targetEPR, 'targetKLD': self.targetKLD,
                         'timeFactor':-1, 'isBaseSystem': self.isBaseSystem}  # the structure in which the trajectory will be saves per file

        try:
            os.makedirs(saveDir)
        except:  # In case existing - delete and recreate empty folder
            shutil.rmtree(saveDir)
            os.makedirs(saveDir)

        # Create Coarse-Grained trajectory
        mWx = pt.CalcW4DrivingForce(mW, self.extForce)
        dataSet, nCgDim, vHiddenStates = pt.CreateCoarseGrainedTraj(nDim, lenTrajFull, mWx, vHiddenStates, timeRes, semiCG=semiCG)
        kldEstimator, T, _, _ = pt.CalcKLDPartialEntropyProdRate(dataSet, vHiddenStates)

        dataSet = torch.from_numpy(dataSet[:, 0]).float()
        dataSet = torch.utils.data.TensorDataset(dataSet)

        # Define sampler
        if mode == 'train':
            sampler = CSS(1, dataSet.tensors[0].size()[0], self.seqLen, self.batchSize, trainIterInEpoch, train=True)
        else:
            sampler = CSS(1, dataSet.tensors[0].size()[0], self.seqLen, self.batchSize, 1, train=False)

        # Define Dataloader
        dataLoader = torch.utils.data.DataLoader(dataSet, sampler=sampler)

        # Sample batches and save them
        print("Creating Dataset")
        for iIter, x_batch in enumerate(dataLoader):
            x_batch = x_batch[0].squeeze()
            torch.save(x_batch, saveDir + os.sep + samplePrefix + str(iIter) + fileSuffix)
        print("Dataset Created!")

        # Save dataset descriptor
        dTrajTemplate['targetKLD'] = kldEstimator
        dTrajTemplate['nBatchedSamples'] = len(sampler)
        dTrajTemplate['timeFactor'] = T
        torch.save(dTrajTemplate, saveDir + os.sep + dsDescriptorName + fileSuffix)

        # Save The whole coarse grained trajectory for future modification of the samples
        torch.save(dataSet, saveDir + os.sep + trajFileName + fileSuffix)

        # Clear cache
        del dataLoader
        del dataSet

        return kldEstimator, len(sampler), T


    # Overriding existing batches with different batches created from the trajectory.
    # This function used in order to not generate full trajectories every time we want to check different input size.
    def ChangeBatchedSamples(self, batchSize=-1, seqLen=-1):
        # Get common params for dataset
        samplePrefix, fileSuffix, trainIterInEpoch, dsDescriptorName, trajFileName = DataSetCreationParams()

        if batchSize != -1:
            self.batchSize = batchSize

        if seqLen != -1:
            self.seqLen = seqLen

        # Read the trajectory in order to generate new samples
        dataSet = torch.load(self.DataSetDir + os.sep + trajFileName + fileSuffix)

        # Delete old samples
        for iSample in range(self.nBatchedSamples):
            os.remove(self.DataSetDir + os.sep + samplePrefix + str(iSample) + fileSuffix)

        ### Create New samples
        # Define sampler
        if os.path.split(self.DataSetDir)[-1] == 'train':
            sampler = CSS(1, dataSet.tensors[0].size()[0], self.seqLen, self.batchSize, trainIterInEpoch, train=True)
        else:
            sampler = CSS(1, dataSet.tensors[0].size()[0], self.seqLen, self.batchSize, 1, train=False)

        # Define Dataloader
        dataLoader = torch.utils.data.DataLoader(dataSet, sampler=sampler)


        # Sample batches and save them
        print("Modifying Dataset")
        for iIter, x_batch in enumerate(dataLoader):
            x_batch = x_batch[0].squeeze()
            torch.save(x_batch, self.DataSetDir + os.sep + samplePrefix + str(iIter) + fileSuffix)
        print("Dataset Modified!")
        self.nBatchedSamples = iIter + 1

        # Update Dataset descriptor
        dTraj = torch.load(self.DataSetDir + os.sep + dsDescriptorName + fileSuffix)
        dTraj['seqLen'] = self.seqLen
        dTraj['batchSize'] = self.batchSize
        dTraj['nBatchedSamples'] = self.nBatchedSamples
        torch.save(dTraj, self.DataSetDir + os.sep + dsDescriptorName + fileSuffix)

        return True


class CGTrajectoryDataSetMM(Dataset):
    def __init__(self, seqLen=32, batchSize=4096, lenTrajFull=1e6, mu=0., F=0., mode='train',
                 rootDir='StoredDataSets', loadSet=False, isBaseSystem=True, semiCG=False):
        # Note: only available modes are : 'train','valid','test'
        assert (mode in ['train', 'valid', 'test'])

        self.DataSetDir = rootDir + os.sep + mode
        self.isBaseSystem = isBaseSystem
        self.mu = mu
        self.F = F
        self.seqLen = seqLen
        self.batchSize = batchSize
        self.targetKLD = -1  # Initialize target KLD estimation of EPR
        self.targetEPR = -1  # Initialize target full EPR
        self.nBatchedSamples = -1  # Initialize number of batched samples - the length of the dataset
        self.timeFactor = -1 # Initialize the time factor in ourder to move from 'per jump' 'per sec'

        lenTrajFull = int(lenTrajFull) # Assure that only int values passed

        if not loadSet:  # save the data
            if isBaseSystem:
                mWx, nDim, vHiddenStates, timeRes = MolecularMotor(mu, F)  # Calculate W matrix after applying force
                vP0 = np.random.uniform(size=(nDim))
                vP0 = vP0 / sum(vP0)
                _, vPiX, _, _ = MESolver(nDim, vP0, mWx, timeRes)
                self.targetEPR = EntropyRateCalculation(nDim, mWx, vPiX)
                # Create and save dataset. Updates targerKLD and nBatchedSamples attributes
                self.targetKLD, self.nBatchedSamples, self.timeFactor = \
                    self.CreateRNeepDataSet(lenTrajFull, mode, self.DataSetDir, semiCG=semiCG)
            # Save Dataset information for future loading
        else:
            # Read data set description
            # TODO : make this part more scalable
            _, fileSuffix, _, dsDescriptorName, _ = DataSetCreationParams()

            dDataSet = torch.load(self.DataSetDir + os.sep + dsDescriptorName + fileSuffix)
            self.seqLen = dDataSet['seqLen']
            self.batchSize = dDataSet['batchSize']
            self.isBaseSystem = dDataSet['isBaseSystem']
            self.mu = dDataSet['mu']
            self.F = dDataSet['F']
            self.targetEPR = dDataSet['targetEPR']
            self.targetKLD = dDataSet['targetKLD']
            self.nBatchedSamples = dDataSet['nBatchedSamples']
            self.timeFactor = dDataSet['timeFactor']

    def __len__(self):
        return self.nBatchedSamples

    def __getitem__(self, idx):
        samplePrefix, fileSuffix, _, _, _ = DataSetCreationParams()
        batchedSample = torch.load(self.DataSetDir + os.sep + samplePrefix + str(int(idx)) + fileSuffix)
        FullEPR = self.targetEPR
        KldEPR = self.targetKLD
        return batchedSample, FullEPR, KldEPR


    # The function creates and stores offline the trajectory by batches
    def CreateRNeepDataSet(self, lenTrajFull, mode, saveDir, semiCG=False):
        # The Dataset will be created for specific forces.
        # The Dataset is saved by BATCHES and NOT single SAMPLES

        # Get common params for dataset
        samplePrefix, fileSuffix, trainIterInEpoch, dsDescriptorName, trajFileName = DataSetCreationParams()

        # Base rate matrix
        mW, nDim, vHiddenStates, timeRes = BaseSystem()

        # Dataset descriptor
        dTrajTemplate = {'seqLen': self.seqLen, 'batchSize': self.batchSize, 'nBatchedSamples': -1,
                         'extForce': self.extForce, 'targetEPR': self.targetEPR, 'targetKLD': self.targetKLD,
                         'timeFactor':-1, 'isBaseSystem': self.isBaseSystem}  # the structure in which the trajectory will be saves per file

        try:
            os.makedirs(saveDir)
        except:  # In case existing - delete and recreate empty folder
            shutil.rmtree(saveDir)
            os.makedirs(saveDir)

        # Create Coarse-Grained trajectory
        dataSet, nCgDim, vHiddenStates = CreateMMTrajectory(self.mu, self.F, int(lenTrajFull), fullCg=not(semiCG), isCG=True, remap=False)
        kldEstimator, T, _, _ = pt.CalcKLDPartialEntropyProdRate(dataSet, vHiddenStates)

        dataSet = torch.from_numpy(dataSet[:, 0]).float()
        dataSet = torch.utils.data.TensorDataset(dataSet)

        # Define sampler
        if mode == 'train':
            sampler = CSS(1, dataSet.tensors[0].size()[0], self.seqLen, self.batchSize, trainIterInEpoch, train=True)
        else:
            sampler = CSS(1, dataSet.tensors[0].size()[0], self.seqLen, self.batchSize, 1, train=False)

        # Define Dataloader
        dataLoader = torch.utils.data.DataLoader(dataSet, sampler=sampler)

        # Sample batches and save them
        print("Creating Dataset")
        for iIter, x_batch in enumerate(dataLoader):
            x_batch = x_batch[0].squeeze()
            torch.save(x_batch, saveDir + os.sep + samplePrefix + str(iIter) + fileSuffix)
        print("Dataset Created!")

        # Save dataset descriptor
        dTrajTemplate['targetKLD'] = kldEstimator
        dTrajTemplate['nBatchedSamples'] = len(sampler)
        dTrajTemplate['timeFactor'] = T
        torch.save(dTrajTemplate, saveDir + os.sep + dsDescriptorName + fileSuffix)

        # Save The whole coarse grained trajectory for future modification of the samples
        torch.save(dataSet, saveDir + os.sep + trajFileName + fileSuffix)

        # Clear cache
        del dataLoader
        del dataSet

        return kldEstimator, len(sampler), T


    # Overriding existing batches with different batches created from the trajectory.
    # This function used in order to not generate full trajectories every time we want to check different input size.
    def ChangeBatchedSamples(self, batchSize=-1, seqLen=-1):
        # Get common params for dataset
        samplePrefix, fileSuffix, trainIterInEpoch, dsDescriptorName, trajFileName = DataSetCreationParams()

        if batchSize != -1:
            self.batchSize = batchSize

        if seqLen != -1:
            self.seqLen = seqLen

        # Read the trajectory in order to generate new samples
        dataSet = torch.load(self.DataSetDir + os.sep + trajFileName + fileSuffix)

        # Delete old samples
        for iSample in range(self.nBatchedSamples):
            os.remove(self.DataSetDir + os.sep + samplePrefix + str(iSample) + fileSuffix)

        ### Create New samples
        # Define sampler
        if os.path.split(self.DataSetDir)[-1] == 'train':
            sampler = CSS(1, dataSet.tensors[0].size()[0], self.seqLen, self.batchSize, trainIterInEpoch, train=True)
        else:
            sampler = CSS(1, dataSet.tensors[0].size()[0], self.seqLen, self.batchSize, 1, train=False)

        # Define Dataloader
        dataLoader = torch.utils.data.DataLoader(dataSet, sampler=sampler)


        # Sample batches and save them
        print("Modifying Dataset")
        for iIter, x_batch in enumerate(dataLoader):
            x_batch = x_batch[0].squeeze()
            torch.save(x_batch, self.DataSetDir + os.sep + samplePrefix + str(iIter) + fileSuffix)
        print("Dataset Modified!")
        self.nBatchedSamples = iIter + 1

        # Update Dataset descriptor
        dTraj = torch.load(self.DataSetDir + os.sep + dsDescriptorName + fileSuffix)
        dTraj['seqLen'] = self.seqLen
        dTraj['batchSize'] = self.batchSize
        dTraj['nBatchedSamples'] = self.nBatchedSamples
        torch.save(dTraj, self.DataSetDir + os.sep + dsDescriptorName + fileSuffix)

        return True

if __name__ == '__main__':
    # Define estimated size of trajectories(this will be the size of full trajectory which will be coarse grained)
    nTimeStamps = 1e8 #int(4096 * 128 * 1e2)  # batchSize X maxBatchSize X minNumBatches

    # Init Dataset - including dataset dumping

    trainDataSet = CGTrajectoryDataSet(lenTrajFull=nTimeStamps)
    trainDataSet.ChangeBatchedSamples(seqLen=128)
