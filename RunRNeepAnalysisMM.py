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
import PhysicalModels.PartialTrajectories as pt
import LearningModels.Neep as neep
from Dataset import CGTrajectoryDataSet
from Utility.Params import BaseSystem, ExtForcesGrid, MolecularMotor

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
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--wd', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--batch_size', '-b', default=4096, type=int,
                        help='Training batch size')
    parser.add_argument('--length', '-n', default=5e7, type=int,
                        help='Length of trajectory')
    parser.add_argument('--epochs', '-e', default=20, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--seq_list', '-l', default='3,16,32,64,128', type=str,
                        help='Input sequence size to check')
    parser.add_argument('--ext_forces', '-f',
                        choices=('coarse', 'full', 'nearSt', 'nearSt', 'zoomed', 'extended', 'RNEEPadd'),
                        default='coarse',
                        help='Define grid of external forces')
    parser.add_argument("--save-path",
                        default=".", type=str,
                        metavar="PATH", help="path to save result (default: none)")
    parser.add_argument("--dump-ds",
                        default="StoredDataSets", type=str,
                        metavar="PATH", help="path to dump dataset (default: 'StoredDataSets')")
    parser.add_argument("--semiCG", action="store_true", help="Perform semi coarse grain")
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

    vSeqSize = np.array([int(item) for item in opt.seq_list.split(',')])
    maxSeqSize = np.max(vSeqSize)

    ## Define base dynamics
    mW, nDim, vHiddenStates, timeRes = BaseSystem()
    nTimeStamps = int(opt.length)  # how many time stamps will be saved


    # Fetch external forces grid
    # -----------Grid----------------
    res = 0.3
    resInterp = 0.03
    vMu = np.array([[3]])
    vFl = np.expand_dims(np.arange(0.5 + res / 2, 1 + res / 2, res), 1)
    #vFlinterp = np.expand_dims(np.arange(-1 + resInterp / 2, 1 + resInterp / 2, resInterp), 1)
    # vFlinterp = vFlinterp[1:]  # TODO remove line
    nMu = vMu.size
    nFl = vFl.size

    vGrid = np.reshape((vMu.repeat(nFl, 0) + vFl.repeat(nMu, 1)).T, -1)
    #vGridInterp = np.reshape((vMu.repeat(vFlinterp.size, 0) + vFlinterp.repeat(nMu, 1)).T, -1)

    # Init vectors for plotting
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
        mu = vMu[0][idx // vFl.size]
        F = x
        mWx, nDim, vHiddenStates, timeRes = MolecularMotor(mu, F)  # Calculate W matrix after applying force
        # Passive partial entropy production rate
        vP0 = np.random.uniform(size=(nDim))
        vP0 = vP0 / sum(vP0)
        n, vPiX, mWx, vWPn = MESolver(nDim, vP0, mWx, timeRes)
        vPassive[idx] = pt.CalcPassivePartialEntropyProdRate(mWx, vPiX)
        # The full entropy rate
        vFull[i] = pt.EntropyRateCalculation(nDim, mWx, vPiX)

        # Create Datasets
        trainDataSet = CGTrajectoryDataSet(seqLen=vSeqSize[0], batchSize=opt.batch_size, lenTrajFull=nTimeStamps,
                                           extForce=x, rootDir=opt.dump_ds, semiCG=opt.semiCG)
        validDataSet = CGTrajectoryDataSet(seqLen=vSeqSize[0], batchSize=opt.batch_size, lenTrajFull=nTimeStamps,
                                           extForce=x, mode='valid', rootDir=opt.dump_ds, semiCG=opt.semiCG)
        # KLD bound
        vKldValid[i] = validDataSet.targetKLD
        T = validDataSet.timeFactor

        k = 0
        for iSeqSize in vSeqSize:
            print('Calculating estimator for x = ' + str(x) + ' ; Sequence size: ' + str(iSeqSize) + " ; KLD: " + str(
                vKldValid[i]))
            validLoader = torch.utils.data.DataLoader(validDataSet)
            trainLoader = torch.utils.data.DataLoader(trainDataSet)

            # define RNN model
            if rneeptFlag == False:
                model = neep.RNEEP()
                outFileadd = ''
            else:
                model = neep.RNEEPT()
                outFileadd = 'T_'
            # if device == 'cuda:0':
            #     model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
            model.to(device)
            # defining the optimizer
            # optimizeurikr = SGD(model.parameters(),lr=vLrate[k])
            optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
            trainRnn = neep.make_trainRnn(model, optimizer, iSeqSize, device)
            bestLoss = 1e3

            # Define sampler - train and validation
            for epoch in range(int(opt.epochs)):
                tic = time.time()
                bestLossEpoch, bestEpRate, bestEpErr = trainRnn(trainLoader, validLoader, epoch)
                toc = time.time()
                if bestLossEpoch < bestLoss:
                    mNeep[k, i] = bestEpRate / T
                    bestLoss = bestLossEpoch
                    # Save best model for specific external force
                    state = {
                        'model': model.state_dict(),
                        'test_epr': mNeep[k, i],
                        'test_loss': bestLoss,
                        'epoch': epoch,
                    }
                    torch.save(state,
                               plotDir + os.sep + 'model_forceIdx' + str(idx) + 'seqSize' + str(iSeqSize) + '.pt')
                print('Elapsed time of Epoch ' + str(epoch + 1) + ' is: ' + str(toc - tic) + " ; KLD est: " + str(
                    bestEpRate / T))

            k += 1
            # Modify the batches to represent he next seqLen input
            if k < len(vSeqSize):
                trainDataSet.ChangeBatchedSamples(seqLen=vSeqSize[k])
                validDataSet.ChangeBatchedSamples(seqLen=vSeqSize[k])
        i += 1

        # %% Save results
        print("DB mNeep:" + str(mNeep))
        with open(plotDir + os.sep + 'vInformed_x_' + outFileadd + str(i - 1) + '.pickle', 'wb') as handle:
            pickle.dump(vInformed, handle)
        with open(plotDir + os.sep + 'vPassive_x_' + outFileadd + str(i - 1) + '.pickle', 'wb') as handle:
            pickle.dump(vPassive, handle)
        with open(plotDir + os.sep + 'vKld_x_' + outFileadd + str(i - 1) + '.pickle', 'wb') as handle:
            pickle.dump(vKldValid, handle)
        with open(plotDir + os.sep + 'mNeep_x_' + outFileadd + str(i - 1) + '.pickle', 'wb') as handle:
            pickle.dump(mNeep, handle)


