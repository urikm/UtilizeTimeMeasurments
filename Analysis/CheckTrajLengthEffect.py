"""
@title : Analyse the effect of changing the trajectory length on the estimated EPR

@author: Uri Kapustin
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import PhysicalModels.PartialTrajectories as pt
from Utility.Params import BaseSystem
from LearningModels import Neep
from Dataset import CGTrajectoryDataSet
from RNeepSampler import CartesianSeqSampler as CSS

def EvaluateModel(model, length, x): # Evaluating as done in 'RunRNeepAnalysis.py'
    # TODO : make script more flexible
    #nTrainIterPerEpoch = 5000
    batch_size = 128
    iSeqSize = 12
    #validDataSet = torch.from_numpy(Data[:, 0])
    #vValidL = np.kron(0.5, np.ones(validDataSet.size()))
    #vValidL = torch.from_numpy(vValidL).type(torch.FloatTensor)
    #validDataSet = torch.utils.data.TensorDataset(validDataSet, vValidL)
    validDataSet = CGTrajectoryDataSet(seqLen=iSeqSize, batchSize=batch_size, lenTrajFull=length,
                                       extForce=x, mode='valid')
    validLoader = torch.utils.data.DataLoader(validDataSet)

    avgValLosses = 0
    avgValScores = []
    with torch.no_grad():
        for x_val, _, _ in validLoader:
            x_val = x_val.squeeze().long().to(device)
            model.eval()
            entropy_val = model(x_val)
            # print("Out Model Val: Input size " + str(x_val.size()) + " ; Output size: " + str(entropy_val.size()))
            val_loss = (-entropy_val + torch.exp(-entropy_val)).mean()
            avgValLosses += val_loss
            avgValScores.append(entropy_val)

        avgValScores = torch.cat(avgValScores).squeeze()
        predEntRate = torch.mean(avgValScores) / (iSeqSize - 1)
        avgValLoss = avgValLosses/validLoader.__len__()
    return predEntRate, avgValLoss

# %% Define parameters
modelCPs = ['model_forceIdx0seqSize32.pt', 'model_forceIdx1seqSize32.pt', 'model_forceIdx15seqSize32.pt'] #, 'model_forceIdx16seqSize32.pt']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
vTrajLengths = np.array([5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7])
mW, nDim, vHiddenStates, timeRes = BaseSystem()
extForce = [-2., -1.7, 2.47002] #, 2.77002]
nIters = 10 # Number of iterations for collecting statistics


# %% Load model
mEffectiveLen = np.zeros([len(extForce), vTrajLengths.shape[0], nIters])
mEstimatedEPr = np.zeros([len(extForce), vTrajLengths.shape[0], nIters])
for iForce, modelCP in enumerate(modelCPs):
    model = Neep.RNEEP()
    print('DBG')
    if device == 'cpu':
        model.load_state_dict(torch.load(modelCP, map_location=torch.device('cpu'))['model'])
    else:
        model.load_state_dict(torch.load(modelCP)['model'])
    # %% Create CG trajectories in different lengths and calculate the EPR result of model
    for iIter in range(nIters):
        for iLen, len in enumerate(vTrajLengths):
            mWx = pt.CalcW4DrivingForce(mW, extForce[iForce])
            mCgTrajectory, nCgDim, vHiddenStates = pt.CreateCoarseGrainedTraj(nDim, int(len), mWx, vHiddenStates, timeRes)
            sigmaDotKld, T, _, _ = pt.CalcKLDPartialEntropyProdRate(mCgTrajectory, vHiddenStates)
            mEffectiveLen[iForce, iLen, iIter] = mCgTrajectory.shape[0]
            # print('Traj length: ' + str(mEffectiveLen[iLen]) + ' | T: ' + str(T))
            predEntRate, avgValLoss = EvaluateModel(model, int(len), extForce[iForce])
            mEstimatedEPr[iForce, iLen, iIter] = predEntRate / T / sigmaDotKld # normalized
            #print('Traj length: ' + str(mEffectiveLen[iForce, iLen, iIter]) + ' | EPR: ' + str(mEstimatedEPr[iForce, iLen, iIter]))
    print('Ended Iteration ' + str(iIter) + ' from total of ' + str(nIters) + 'iteration')
# %% Plot analysis

resFig = plt.figure(0)
for iForce, _ in enumerate(modelCPs):
    plt.errorbar(mEffectiveLen[iForce].mean(axis=1), mEstimatedEPr[iForce].mean(axis=1)/mEstimatedEPr[iForce].mean(axis=1)[-1], xerr=mEffectiveLen[iForce].std(axis=1)/2, yerr=mEstimatedEPr[iForce].std(axis=1)/2, fmt=':o', label='x='+str(extForce[iForce]))

plt.xscale('log')
plt.xlabel('Effective trajectory length [jumps]', fontsize='x-large')
plt.ylabel('Normalized EPR [per jump]', fontsize='x-large')
# plt.title('EPR vs Trajectory length')
plt.legend(prop={'size': 10})
plt.show()
resFig.set_size_inches((8, 8))
resFig.savefig(f'PlotTrajLengthAnalysis.svg')
