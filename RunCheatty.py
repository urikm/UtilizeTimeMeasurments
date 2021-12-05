# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:04:45 2020

@title: KLD entropy production rate estimator(Reproduction of inferred broken detailed balance paper from 2019)

@author: Uri Kapustin

@description: This is the main script to run

"""
import argparse

import numpy as np
import time
import os

from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
import PhysicalModels.PartialTrajCheatty as pt
import LearningModels.Neep as neep
from RNeepSampler import CartesianSeqSampler as CSS

import torch
import torch.utils.data as dat
from torch.optim import Adam, SGD, Adagrad, RMSprop, Rprop

import pickle

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# device = 'cpu'

# %% Parse arguments
def parse_args():
    """Parse script arguments.

    Get training hyper-parameters such as: learning rate, momentum,
    batch size, number of training epochs and optimizer.
    Get training dataset and the model name.
    """
    parser = argparse.ArgumentParser(description="Hidden Markov EPR estimation using NEEP")
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--batch_size', '-b', default=4096, type=int,
                        help='Training batch size')
    parser.add_argument('--epochs', '-e', default=10, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--seq_list', '-l', default='3,16,32,64,128', type=str,
                        help='Input sequence size to check')
    parser.add_argument('--ext_forces', '-f', choices=('coarse', 'nearSt', 'zoomed', 'extended'), default='coarse',
                        help='Define grid of external forces')
    parser.add_argument("--save-path",
                        default="", type=str,
                        metavar="PATH", help="path to save result (default: none)")

    return parser.parse_args()


# %% Comparing KLD estimator to previous
if __name__ == '__main__':
    ## Handle parsing arguments
    opt = parse_args()
    ## UI
    #
    loadDbFlag = False  # True - read dataset from file; False - create new(very slow)
    rneeptFlag = False  # True - use time data ; False - only states data
    plotFlag = True

    plotDir = opt.save_path
    try:
        os.mkdir(plotDir)
    except:
        pass

    # TODO : need to make rewrite it if we want use pre-built DB
    dbName = 'RneepDbCoarse'
    dbPath = 'StoredDataSets' + os.sep + dbName
    dbFileName = 'InitRateMatAsGilis'
    logFile = 'log.txt'

    vSeqSize = np.array([int(item) for item in opt.seq_list.split(',')])
    maxSeqSize = np.max(vSeqSize)
    nTrainIterPerEpoch = 5000

    nDim = 4  # dimension of the problem
    nTimeStamps = int(maxSeqSize * opt.batch_size * 1e2)  # how much time stamps will be saved
    vHiddenStates = np.array([2, 3])  # states 3 and 4 for 4-D state sytem

    ## Define base dynamics
    if 0:
        mW = pt.GenRateMat(nDim)  # transition matrix
        timeRes = 1
    else:
        mW = np.array([[-11., 2., 0., 1.], [3., -52.2, 2., 35.], [0., 50., -77., 0.7], [8., 0.2, 75., -36.7]])
        timeRes = 0.001

    # Calculate Stalling data
    vPiSt, xSt, r01, r10 = pt.CalcStallingData(mW)
    # Init vectors for plotting
    vGrid = np.concatenate((np.arange(-1., xSt, 1), np.arange(xSt, 3., 1)))
    # This used for running with different grid, dont change the upper version - its the defualt
    if opt.ext_forces == 'coarse':
        vGrid = np.concatenate((np.arange(-1., xSt, 1), np.arange(xSt, 3., 1)))
    elif opt.ext_forces == 'nearSt':
        vGrid = np.concatenate((np.arange(xSt - 0.02, xSt - 0.005, 0.01), np.arange(xSt, xSt + 0.02, 0.01)))
    elif opt.ext_forces == 'zoomed':
        vGrid = np.arange(-1., 0., 0.25)
    elif opt.ext_forces == 'extended':
        vGrid = np.arange(-2., 0., 0.25)

    vInformed = np.zeros(np.size(vGrid))
    vPassive = np.zeros(np.size(vGrid))
    vKld = np.zeros(np.size(vGrid))
    vKldValid = np.zeros(np.size(vGrid))
    vFull = np.zeros(np.size(vGrid))
    mNeep = np.zeros([np.size(vSeqSize), np.size(vGrid)])

    print("Used device is:" + device)

    # For each driving force in vGrid, estimate the ERP
    i = 0
    for idx, x in enumerate(vGrid):
        mWx = pt.CalcW4DrivingForce(mW, x)  # Calculate W matrix after applying force
        # Passive partial entropy production rate
        vP0 = np.random.uniform(size=(nDim))
        vP0 = vP0 / sum(vP0)
        n, vPiX, mWx, vWPn = MESolver(nDim, vP0, mWx, timeRes)
        vPassive[i] = pt.CalcPassivePartialEntropyProdRate(mWx, vPiX)
        # Informed partial entropy production rate
        vInformed[i] = pt.CalcInformedPartialEntropyProdRate(mWx, vPiX, vPiSt)
        # The full entropy rate
        vFull[i] = pt.EntropyRateCalculation(nDim, mWx, vPiX)
        # KLD bound
        if loadDbFlag:
            # TODO : support for using Time Data??
            # Choose the wanted trajectory according to x
            with open(dbPath + os.sep + 'MappingVector' + '.pickle', 'rb') as handle:
                vX = pickle.load(handle)
                wantedIdx = (np.abs(vX - x)).argmin()
            with open(dbPath + os.sep + dbFileName + '_' + str(wantedIdx) + '.pickle', 'rb') as handle:
                dDataTraj = pickle.load(handle)

            trainDataSet = dDataTraj.pop('vStates')
            trainDataSet = np.array([trainDataSet, dDataTraj.pop('vTimeStamps')]).T
            vKld[i] = dDataTraj['kldBound']
            validDataSet = trainDataSet
            vKldValid[i] = vKld[i]
            T = dDataTraj['timeFactor']
        else:
            trainDataSet, nCgDim = pt.CreateCoarseGrainedTraj(nDim, nTimeStamps, mWx, vHiddenStates, timeRes)
            sigmaDotKld, T, sigmaDotAff, sigmaWtd, dd1H2, dd2H1 = pt.CalcKLDPartialEntropyProdRate(trainDataSet, nCgDim)
            vKld[i] = sigmaDotKld
            validDataSet, _ = pt.CreateCoarseGrainedTraj(nDim, nTimeStamps, mWx, vHiddenStates, timeRes)
            sigmaDotKld, T, sigmaDotAff, sigmaWtd, dd1H2, dd2H1 = pt.CalcKLDPartialEntropyProdRate(validDataSet, nCgDim)
            vKldValid[i] = sigmaDotKld

        k = 0
        if rneeptFlag == False:
            # ==============================================
            # # NEEP entropy rate
            trainDataSet = torch.from_numpy(trainDataSet[:, 0])
            vTrainL = np.kron(vKld[i] * T, np.ones(len(trainDataSet)))
            vTrainL = torch.from_numpy(vTrainL).type(torch.FloatTensor)
            trainDataSet = torch.utils.data.TensorDataset(trainDataSet, vTrainL)
            # trainLoader =  torch.utils.data.DataLoader(trainDataSet, batch_size=opt.batch_size, shuffle=True)

            # mValid = validDataSet[:int(np.floor(validDataSet.shape[0]/iSeqSize)*iSeqSize),0].reshape(iSeqSize,-1,order='F').transpose()
            validDataSet = torch.from_numpy(validDataSet[:, 0])
            vValidL = np.kron(vKldValid[i] * T, np.ones(len(validDataSet)))
            vValidL = torch.from_numpy(vValidL).type(torch.FloatTensor)
            validDataSet = torch.utils.data.TensorDataset(validDataSet, vValidL)
            validLoader = torch.utils.data.DataLoader(validDataSet, batch_size=opt.batch_size, shuffle=False)
        else:  # TODO : update this section
            # ==============================================
            # NEEP entropy rate using time
            # Obsolete Naive sampler - TODO : implement more "standard" sampler
            tmpStates = trainDataSet[:int(np.floor(trainDataSet.shape[0] / iSeqSize) * iSeqSize), 0].reshape(iSeqSize,
                                                                                                             -1,
                                                                                                             order='F').transpose()
            tmpWtd = trainDataSet[:int(np.floor(trainDataSet.shape[0] / iSeqSize) * iSeqSize), 1].reshape(iSeqSize,
                                                                                                          -1,
                                                                                                          order='F').transpose()

            trainDataSet = np.concatenate((np.expand_dims(tmpStates, 2), np.expand_dims(tmpWtd, 2)), axis=2)
            trainDataSet = torch.from_numpy(trainDataSet).float()
            vTrainL = np.kron(vKld[i] * T, np.ones(int(np.floor(trainDataSet.shape[0] / iSeqSize))))
            vTrainL = torch.from_numpy(vTrainL).float()
            trainDataSet = torch.utils.data.TensorDataset(trainDataSet, vTrainL)
            # trainLoader =  torch.utils.data.DataLoader(trainDataSet, batch_size=opt.batch_size, shuffle=True)

            tmpSValid = validDataSet[:int(np.floor(validDataSet.shape[0] / iSeqSize) * iSeqSize), 0].reshape(iSeqSize,
                                                                                                             -1,
                                                                                                             order='F').transpose()
            tmpWValid = validDataSet[:int(np.floor(validDataSet.shape[0] / iSeqSize) * iSeqSize), 1].reshape(iSeqSize,
                                                                                                             -1,
                                                                                                             order='F').transpose()

            validDataSet = np.concatenate((np.expand_dims(tmpSValid, 2), np.expand_dims(tmpWValid, 2)), axis=2)
            validDataSet = torch.from_numpy(validDataSet).float()
            vValidL = np.kron(vKldValid[i] * T, np.ones(int(np.floor(validDataSet.shape[0] / iSeqSize))))
            vValidL = torch.from_numpy(vValidL).float()
            validDataSet = torch.utils.data.TensorDataset(validDataSet, vValidL)
            validLoader = torch.utils.data.DataLoader(validDataSet, batch_size=opt.batch_size, shuffle=False)
        # ==============================================

        for iSeqSize in vSeqSize:
            print('Calculating estimator for x = ' + str(x) + ' ; Sequence size: ' + str(iSeqSize) + " ; KLD: " + str(
                vKldValid[i]))

            # define RNN model
            if rneeptFlag == False:
                model = neep.RNEEP()
                outFileadd = ''
            else:
                model = neep.RNEEPT()
                outFileadd = 'T_'
            if device == 'cuda:0':
                model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            model.to(device)
            # defining the optimizer
            # optimizer = SGD(model.parameters(),lr=vLrate[k])
            optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
            trainRnn = neep.make_trainRnn(model, optimizer, iSeqSize, device)
            bestLoss = 1e3

            # Define sampler - train and validation
            for epoch in range(int(opt.epoches)):
                validLoader = torch.utils.data.DataLoader(validDataSet,
                                                          sampler=CSS(1, validDataSet.tensors[0].size()[0], iSeqSize,
                                                                      opt.batch_size, nTrainIterPerEpoch, train=False),
                                                          pin_memory=False)
                trainLoader = torch.utils.data.DataLoader(trainDataSet,
                                                          sampler=CSS(1, trainDataSet.tensors[0].size()[0], iSeqSize,
                                                                      opt.batch_size, nTrainIterPerEpoch, train=True),
                                                          pin_memory=False)
                tic = time.time()
                bestLossEpoch, bestEpRate, bestEpErr = trainRnn(trainLoader, validLoader, epoch)
                toc = time.time()
                print('Elapsed time of Epoch ' + str(epoch + 1) + ' is: ' + str(toc - tic) + " ; KLD est: " + str(
                    bestEpRate / T))
                if bestLossEpoch < bestLoss:
                    mNeep[k, i] = bestEpRate / T
                    bestLoss = bestLossEpoch
                    # Save best model for specific external force
            k += 1

        i += 1

        # %% Save results
        print("DB mNeep:" + str(mNeep))
        with open(plotDir + os.sep + 'vInformed_x_' + outFileadd + str(i - 1) + '.pickle', 'wb') as handle:
            pickle.dump(vInformed, handle)
        with open(plotDir + os.sep + 'vPassive_x_' + outFileadd + str(i - 1) + '.pickle', 'wb') as handle:
            pickle.dump(vPassive, handle)
        with open(plotDir + os.sep + 'vKld_x_' + outFileadd + str(i - 1) + '.pickle', 'wb') as handle:
            pickle.dump(vKld, handle)
        with open(plotDir + os.sep + 'mNeep_x_' + outFileadd + str(i - 1) + '.pickle', 'wb') as handle:
            pickle.dump(mNeep, handle)


