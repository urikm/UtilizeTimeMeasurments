# Compare EPR bounds :  KLD, FCG, with time data  VS Plugin, SCG, without time data

import numpy as np
import matplotlib.pyplot as plt
import PhysicalModels.PartialTrajectories as pt
import Utility.FindPluginInfEPR as infEPR
from Utility.Params import ExtForcesGrid, BaseSystem, HiddenControl



def ModifyRates(mW, ): # simulate system with hidden controlable

    return True

# %% Comparison params
checkHidden = True
maxSeq = 9
threshold = 0
naiveTrajLen = int(1e7)
gamma = 1e-16

# %% Define system
if not checkHidden:
    mW, nDim, vHiddenStates, timeRes = BaseSystem()
    vGrid, _, _ = ExtForcesGrid('converege',interpRes=1e-3)
else:
    vGrid = np.array([0, 0.005, 0.05, 0.5, 5])
vEPRfcg = np.zeros((len(vGrid),))
vEPRscg = np.zeros((len(vGrid),))

# %% Calculate KLD for FCG with time data
for ix, x in enumerate(vGrid):
    if checkHidden:
        mWx, nDim, vHiddenStates, timeRes = HiddenControl(hidBond=x)
    else:
        mWx = pt.CalcW4DrivingForce(mW, x)
    mCgTraj, nCgDim = pt.CreateCoarseGrainedTraj(nDim, naiveTrajLen, mWx, vHiddenStates, timeRes, semiCG=False)
    vEPRfcg[ix], T, _, _, _, _ = pt.CalcKLDPartialEntropyProdRate(mCgTraj, nCgDim)
    vEPRfcg[ix] = vEPRfcg[ix]*T

# %% Calculate Plugin for SCG without time data
for ix, x in enumerate(vGrid):
    if checkHidden:
        mWx, nDim, vHiddenStates, timeRes = HiddenControl(hidBond=x)
    else:
        mWx = pt.CalcW4DrivingForce(mW, x)
    mCgTraj, nCgDim = pt.CreateCoarseGrainedTraj(nDim, naiveTrajLen, mWx, vHiddenStates, timeRes, semiCG=True)
    vEPRscg[ix] = infEPR.EstimatePluginInf(mCgTraj[:, 0], maxSeq=maxSeq, gamma=gamma)

# %% Plot
figT2 = plt.figure()
plt.plot(np.flip(-vGrid), np.flip(vEPRfcg), label='FullCGwTime')
plt.plot(np.flip(-vGrid), np.flip(vEPRscg), label='SemiCGwoTime')
plt.title('Gilis-EPR')
plt.legend()
plt.show()
# figT2.set_size_inches((16, 16))
# figT2.savefig(f'CompareKLDnPLGIN.png')