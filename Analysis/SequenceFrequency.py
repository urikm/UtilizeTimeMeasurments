"""
@title : Visualize the appearance of possible sequences

@author: Uri Kapustin
"""

import numpy as np
import matplotlib.pyplot as plt

import PhysicalModels.ratchet as rt
import PhysicalModels.PartialTrajectories as pt
import Utility.FindPluginInfEPR as infEPR
from Utility.Params import ExtForcesGrid, BaseSystem


# %% Test as Uri - Check the portion of observed sequence that obey ratio of appearance higher than T~(0,1)
def TAU(vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2, thresh=0.001):
    detectT = 0 # hardcoded of the algorithm - shouldn't be controlled externally
    countB = 0
    countF = 0
    countSeq = 0
    countNon = 0
    scoreU = -1
    vBuffer = []
    while countF < len(vProbOfSeq1) and countB < len(vProbOfSeq2):
        if vSeqs1[countF] == vSeqs2[countB]:
            countSeq += 1
            countF += 1
            countB += 1
        elif countF >= len(vProbOfSeq1) or vSeqs1[countF] > vSeqs2[countB]:
            if countB < len(vProbOfSeq2):
                countNon += 1
                countB += 1
        else:
            if countF < len(vProbOfSeq1):
                countNon += 1
                countF += 1
    # Now calculate score
    scoreU = (countNon)/(countNon+countSeq)
    return scoreU

# %% Visualization params
vSeqSize = np.linspace(2, maxSeq, maxSeq - 2 + 1, dtype=np.intc)#np.array([3, 16, 32, 64, 128], dtype=np.float) #np.array([2, 3, 4, 5, 6, 7], dtype=np.float) #
threshold = 0
naiveTrajLen = int(5e6)
gamma = 1e-9
# %% Visualize flatching ratchet
start = 0.5
end = 2.
step = 0.25
vPots = np.linspace(start,end,int(np.floor((end-start)/step)+1))
mW, nDim, vHiddenStates, timeRes = BaseSystem()
mFRSys = np.zeros((len(vSeqSize), len(vPots)))
vEPRfr = np.zeros((len(vPots),))
for ix, x in enumerate(vPots):
    mCgTraj = infEPR.CreateNEEPTrajectory(nTimeStamps=naiveTrajLen, x=x, fullCg=False)
    print(mCgTraj.shape)
    for iSeq, seqLen in enumerate(vSeqSize):
        seqEpr, (vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2) = infEPR.EstimatePluginM(mCgTraj.T, int(seqLen), gamma=gamma)
        mFRSys[iSeq, ix] = seqEpr #TAU(vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2)
    vEPRfr[ix] = infEPR.EstimatePluginInf(mCgTraj.T, gamma=gamma)

# %% Visualize Gili's system
vGrid, _, _ = ExtForcesGrid('full',interpRes=1e-3)
mW, nDim, vHiddenStates, timeRes = BaseSystem()
mGilisSys = np.zeros((len(vSeqSize), len(vGrid)))
vEPRgs = np.zeros((len(vGrid),))
for ix, x in enumerate(vGrid):
    mWx = pt.CalcW4DrivingForce(mW, x)
    mCgTraj, nCgDim = pt.CreateCoarseGrainedTraj(nDim, naiveTrajLen, mWx, vHiddenStates, timeRes)
    for iSeq, seqLen in enumerate(vSeqSize):
        seqEpr, (vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2) = infEPR.EstimatePluginM(mCgTraj[:, 0], int(seqLen), gamma=gamma)
        mGilisSys[iSeq, ix] = seqEpr #TAU(vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2)
    vEPRgs[ix] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=gamma)

# %% Visualize Gili's system - semi CG
vGrid, _, _ = ExtForcesGrid('full',interpRes=1e-3)
mW, nDim, vHiddenStates, timeRes = BaseSystem()
mGilisSys2 = np.zeros((len(vSeqSize), len(vGrid)))
vEPRscg = np.zeros((len(vGrid),))
for ix, x in enumerate(vGrid):
    mWx = pt.CalcW4DrivingForce(mW, x)
    mCgTraj, nCgDim = pt.CreateCoarseGrainedTraj(nDim, naiveTrajLen, mWx, vHiddenStates, timeRes, semiCG=True)
    for iSeq, seqLen in enumerate(vSeqSize):
        seqEpr, (vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2) = infEPR.EstimatePluginM(mCgTraj[:, 0], int(seqLen), gamma=gamma)
        mGilisSys2[iSeq, ix] = seqEpr #TAU(vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2)
    vEPRscg[ix] = infEPR.EstimatePluginInf(mCgTraj[:, 0], gamma=gamma)
# %% Plotting
fig1 = plt.figure()
# extent = np.min(vSeqSize), np.max(vSeqSize), np.min(vPots), np.max(vPots)
plt.imshow(mFRSys, cmap="jet")
plt.xlabel("External force")
plt.ylabel("Input sequence length")
plt.title("Flashing Ratchet")
plt.colorbar()
plt.show()
fig1.set_size_inches((16, 16))
fig1.savefig(f'seqsFR.png')

fig2 = plt.figure()
# extent = np.min(vSeqSize), np.max(vSeqSize), np.min(vGrid), np.max(vGrid)
plt.imshow(mGilisSys, cmap="jet")
plt.xlabel("External force")
plt.ylabel("Input sequence length")
plt.title("Gili's System")
plt.colorbar()
plt.show()
fig2.set_size_inches((16, 16))
fig2.savefig(f'seqsGilis.png')

fig3 = plt.figure()
# extent = np.min(vSeqSize), np.max(vSeqSize), np.min(vGrid), np.max(vGrid)
plt.imshow(mGilisSys2, cmap="jet")
plt.xlabel("External force")
plt.ylabel("Input sequence length")
plt.title("Gili's System - semi CG")
plt.colorbar()
plt.show()
fig2.set_size_inches((16, 16))
fig2.savefig(f'seqsGilis2.png')

figT = plt.figure()
plt.plot(vPots, vEPRfr)
plt.title('FR-EPR')
plt.show()

figT2 = plt.figure()
plt.plot(vGrid, vEPRgs, label='FullCG')
plt.plot(vGrid, vEPRscg, label='SemiCG')
plt.title('Gilis-EPR')
plt.legend()
plt.show()



