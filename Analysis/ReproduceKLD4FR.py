"""
@title : Reproduce KLD estimators for Flatching Ratchet model

@author: Uri Kapustin
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import Utility.FindPluginInfEPR as eprinf
import PhysicalModels.ratchet as rt

# %% Recreate different estimators
vPotentials = np.linspace(0.5,2.,int(np.floor(1.5/0.25)+1))
nTimeStamps = int(1e7)

vKldInf = np.zeros(vPotentials.shape)
vKldInfFullCg = np.zeros(vPotentials.shape)

for iPot, pot in enumerate(vPotentials):
    mCgTrajectory = eprinf.CreateNEEPTrajectory(nTimeStamps=nTimeStamps, x=pot, fullCg=False)
    print(mCgTrajectory.shape)
    vKldInf[iPot] = eprinf.EstimatePluginInf(mCgTrajectory)
    print('For Potential: '+str(pot)+' we get Inf estimator of: '+str(vKldInf[iPot]))

for iPot, pot in enumerate(vPotentials):
    mCgTrajectory = eprinf.CreateNEEPTrajectory(nTimeStamps=int(5*nTimeStamps), x=pot, fullCg=True)
    print(mCgTrajectory.shape)
    vKldInfFullCg[iPot] = eprinf.EstimatePluginInf(mCgTrajectory)
    print('For Potential: '+str(pot)+' we get Inf estimator of: '+str(vKldInf[iPot]))

# %% Plotting
resFig = plt.figure(0)
plt.plot(np.flip(vPotentials), np.flip(vKldInf), ':m', label='InfinityExt_SemiCg')
plt.plot(np.flip(vPotentials), np.flip(vKldInfFullCg), ':k', label='InfinityExt_FullCg')
plt.xlabel('x - Driving Force')
plt.ylabel('Entropy Production rate')
plt.legend()
plt.show()

resFig2 = plt.figure(1)
plt.plot(np.flip(vPotentials), np.flip(vKldInf), ':m', label='InfinityExt_SemiCg')
plt.plot(np.flip(vPotentials), np.flip(vKldInfFullCg), ':k', label='InfinityExt_FullCg')
plt.yscale('log')
plt.xlabel('x - Driving Force')
plt.ylabel('Entropy Production rate')
plt.legend()
plt.show()

resFig.set_size_inches((16, 16))
resFig.savefig(
    os.path.join('InfExtrapolatorKLD.png'))
resFig2.set_size_inches((16, 16))
resFig2.savefig(
    os.path.join('InfExtrapolatorKLDlog.png'))