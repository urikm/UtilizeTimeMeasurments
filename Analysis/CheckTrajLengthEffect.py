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
from RNeepSampler import CartesianSeqSampler as CSS

def EvaluateModel(model, Data): # Evaluating as done in 'RunRNeepAnalysis.py'
    # TODO : make script more flexible
    nTrainIterPerEpoch = 5000
    batch_size = 4096
    iSeqSize = 32
    validDataSet = torch.from_numpy(Data[:, 0])
    vValidL = np.kron(0.5, np.ones(validDataSet.size()))
    vValidL = torch.from_numpy(vValidL).type(torch.FloatTensor)
    validDataSet = torch.utils.data.TensorDataset(validDataSet, vValidL)
    validLoader =  torch.utils.data.DataLoader(validDataSet, sampler = CSS(1,validDataSet.tensors[0].size()[0],iSeqSize,batch_size,nTrainIterPerEpoch,train=False), pin_memory=False)

    avgValLosses = 0
    avgValScores = []
    with torch.no_grad():
        for x_val, y_val in validLoader:
            x_val = x_val.squeeze().long().to(device)
            model.eval()
            entropy_val = model(x_val)
            # print("Out Model Val: Input size " + str(x_val.size()) + " ; Output size: " + str(entropy_val.size()))
            val_loss = (-entropy_val + torch.exp(-entropy_val)).mean()
            avgValLosses += val_loss
            avgValScores.append(entropy_val)

        avgValScores = torch.cat(avgValScores).squeeze()
        predEntRate = torch.mean(avgValScores) / (iSeqSize - 1)
        avgValLoss = avgValLosses / validLoader.sampler.size
    return predEntRate, avgValLoss

# %% Define parameters
modelCPs = ['model_forceIdx0seqSize32.pt', 'model_forceIdx1seqSize32.pt', 'model_forceIdx15seqSize32.pt', 'model_forceIdx16seqSize32.pt']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
vTrajLengths = np.array([5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7])
mW, nDim, vHiddenStates, timeRes = BaseSystem()
extForce = [-2., -1.7, 2.47002, 2.77002]
nIters = 2 # Number of iterations for collecting statistics
resFig = plt.figure(0)

# %% Load model
for iForce, modelCP in enumerate(modelCPs):
    model = Neep.RNEEP()
    print('DBG')
    if device == 'cpu':
        model.load_state_dict(torch.load(modelCP, map_location=torch.device('cpu'))['model'])
    else:
        model.load_state_dict(torch.load(modelCP)['model'])

    # %% Create CG trajectories in different lengths and calculate the EPR result of model
    mEffectiveLen = np.zeros([vTrajLengths.shape[0], nIters])
    mEstimatedEPr = np.zeros([vTrajLengths.shape[0], nIters])
    for iIter in range(nIters):
        for iLen, len in enumerate(vTrajLengths):
            mWx = pt.CalcW4DrivingForce(mW, extForce[iForce])
            mCgTrajectory, nCgDim = pt.CreateCoarseGrainedTraj(nDim, int(len), mWx, vHiddenStates, timeRes)
            sigmaDotKld, T, _, _, _, _ = pt.CalcKLDPartialEntropyProdRate(mCgTrajectory, nCgDim)
            mEffectiveLen[iLen, iIter] = mCgTrajectory.shape[0]
            # print('Traj length: ' + str(mEffectiveLen[iLen]) + ' | T: ' + str(T))
            predEntRate, avgValLoss = EvaluateModel(model, mCgTrajectory)
            mEstimatedEPr[iLen, iIter] = predEntRate / T
            print('Traj length: ' + str(mEffectiveLen[iLen, iIter]) + ' | EPR: ' + str(mEstimatedEPr[iLen, iIter]))

# %% Plot analysis
    plt.errorbar(mEffectiveLen.mean(axis=1), mEstimatedEPr.mean(axis=1), xerr=mEffectiveLen.std(axis=1)/2, yerr=mEstimatedEPr.std(axis=1)/2, fmt=':o', label=str(extForce[iForce]))

plt.xscale('log')
plt.xlabel('Effective trajectory length[jumps]')
plt.ylabel('Entropy Production rate[per jump]')
plt.title('EPR vs Trajectory length')
plt.legend()
plt.show()
resFig.set_size_inches((16, 16))
resFig.savefig(f'PlotTrajLengthAnalysis.png')