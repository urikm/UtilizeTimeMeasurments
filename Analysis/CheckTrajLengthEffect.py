"""
@title : Analyse the effect of changing the trajectory length on the estimated EPR

@author: Uri Kapustin

@Note: This script can be adsjusted to analyse convergence, vs traj length, of different estimators
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

import PhysicalModels.PartialTrajectories as pt
from Utility.Params import BaseSystem
from LearningModels import Neep
from Dataset import CGTrajectoryDataSet
import Utility.FindPluginInfEPR as infEPR

def EvaluateModel(model, length, x): # Evaluating as done in 'RunRNeepAnalysis.py'
    # TODO : make script more flexible
    #nTrainIterPerEpoch = 5000
    batch_size = 128
    iSeqSize = 128
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
modelCP = 'model_forceIdx3seqSize128.pt' #['model_forceIdx0seqSize32.pt', 'model_forceIdx1seqSize32.pt', 'model_forceIdx15seqSize32.pt'] #, 'model_forceIdx16seqSize32.pt']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
vTrajLengths = np.array([5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7])
mW, nDim, vHiddenStates, timeRes = BaseSystem()

extForce = np.array([0.15]) #[-2., -1.7, 2.47002] #, 2.77002]
# NOTE! : RNEEP supported only if script used for 1 external force, and not several!

nIters = 5 # Number of iterations for collecting statistics


# %% Load model
mEffectiveLen = np.zeros([len(extForce), vTrajLengths.shape[0], nIters])
mEstimatedEPr = np.zeros([len(extForce), vTrajLengths.shape[0], nIters])
mEstimatedWtd = np.zeros([len(extForce), vTrajLengths.shape[0], nIters])
mEstimatedAff = np.zeros([len(extForce), vTrajLengths.shape[0], nIters])
mEstimatedPlg = np.zeros([len(extForce), vTrajLengths.shape[0], nIters])
mEstimatedRnp = np.zeros([len(extForce), vTrajLengths.shape[0], nIters])

for iForce, force in enumerate(extForce): #for iForce, modelCP in enumerate(modelCPs):
    model = Neep.RNEEP()
    print()
    if device == 'cpu':
        model.load_state_dict(torch.load(modelCP, map_location=torch.device('cpu'))['model'])
    else:
        model.load_state_dict(torch.load(modelCP)['model'])
    #Create CG trajectories in different lengths and calculate the EPR result of model
    for iIter in range(nIters):
        for iLen, len in enumerate(vTrajLengths):
            mWx = pt.CalcW4DrivingForce(mW, extForce[iForce])
            mCgTrajectory, _, _ = pt.CreateCoarseGrainedTraj(nDim, int(len), mWx, vHiddenStates, timeRes)
            sigmaDotKld, T, sigmaDotAff, sigmaDotWtd = pt.CalcKLDPartialEntropyProdRate(mCgTrajectory, vHiddenStates)
            mEffectiveLen[iForce, iLen, iIter] = mCgTrajectory.shape[0]
            mEstimatedEPr[iForce, iLen, iIter] = sigmaDotKld #predEntRate / T / sigmaDotKld # normalized
            mEstimatedWtd[iForce, iLen, iIter] = sigmaDotWtd
            mEstimatedAff[iForce, iLen, iIter] = sigmaDotAff
            mEstimatedPlg[iForce, iLen, iIter] = infEPR.EstimatePluginInf(mCgTrajectory[:, 0], gamma=1) / T
            mEstimatedRnp[iForce, iLen, iIter], _ = EvaluateModel(model, int(len), extForce[iForce])
            mEstimatedRnp[iForce, iLen, iIter] /= T
        print('Ended Iteration ' + str(iIter) + ' from total of ' + str(nIters) + 'iteration')
# %% Plot analysis

resFig = plt.figure(0)
for iForce, _ in enumerate(extForce): #enumerate(modelCPs):
    plt.errorbar(mEffectiveLen[iForce].mean(axis=1), mEstimatedEPr[iForce].mean(axis=1),  yerr=mEstimatedEPr[iForce].std(axis=1)/2, fmt='d-', lw=0.5, color=(0.3010, 0.7450, 0.9330), label='$\sigma_{\mathrm{KLD}}$')#'x='+str(extForce[iForce]))
    plt.errorbar(mEffectiveLen[iForce].mean(axis=1), mEstimatedPlg[iForce].mean(axis=1), yerr=mEstimatedPlg[iForce].std(axis=1) / 2, fmt='.-', lw=0.5, color=(0.9290, 0.6940, 0.1250), label='$\sigma_{\mathrm{plug}}$')  # 'x='+str(extForce[iForce]))
    plt.errorbar(mEffectiveLen[iForce].mean(axis=1), mEstimatedRnp[iForce].mean(axis=1), yerr=mEstimatedRnp[iForce].std(axis=1) / 2, fmt='s-', lw=0.5, color=(0.1660, 0.3740, 0.0880), label='$\sigma_{\mathrm{RNEEP}}$')  # 'x='+str(extForce[iForce]))
    plt.errorbar(mEffectiveLen[iForce].mean(axis=1), mEstimatedAff[iForce].mean(axis=1),  yerr=mEstimatedAff[iForce].std(axis=1)/2, fmt='^-', lw=0.5, color=(0.6350, 0.0780, 0.1840), label='$\sigma_{\mathrm{aff}}$')#'x='+str(extForce[iForce]))
    plt.errorbar(mEffectiveLen[iForce].mean(axis=1), mEstimatedWtd[iForce].mean(axis=1),  yerr=mEstimatedWtd[iForce].std(axis=1)/2, fmt='v-', lw=0.5, color=(0.4020, 0.545, 0.470), label='$\sigma_{\mathrm{WTD}}$')#'x='+str(extForce[iForce]))

plt.xscale('log')
plt.xlabel('Effective trajectory length', fontsize='small')
plt.ylabel('Entropy Production rate $[s^{-1}]$', fontsize='small')
# plt.title('EPR vs Trajectory length')
plt.tick_params(axis="both", labelsize=6)
plt.legend(prop={'size': 5}, loc=1, title='Full-CG', title_fontsize='xx-small')
plt.show()
resFig.set_size_inches((3.38582677*2, 3.38582677))
resFig.savefig(f'PlotTrajLengthAnalysis.pdf')



