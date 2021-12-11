"""
@title : Reproduce KLD estimators for Flatching Ratchet model

@author: Uri Kapustin
"""

import numpy as np
import matplotlib.pyplot as plt
import Utility.FindPluginInfEPR as eprinf
import PhysicalModels.ratchet as rt

# %% Recreate different estimators
vPotentials = np.linspace(0.5,2.,int(np.floor(1.5/0.25)+1))
nTimeStamps = int(1e7)

vKldInf = np.zeros(vPotentials.shape)

for iPot, pot in enumerate(vPotentials):
    mCgTrajectory = eprinf.CreateNEEPTrajectory(nTimeStamps=nTimeStamps, x=pot, fullCg=True)
    vKldInf[iPot] = eprinf.EstimatePluginInf(mCgTrajectory)
    print('For Potential: '+str(pot)+' we get Inf estimator of: '+str(vKldInf[iPot]))

# %% Plotting
resFig = plt.figure(0)
plt.plot(np.flip(vPotentials), np.flip(vKldInf),':m')
plt.yscale('log')
plt.xlabel('x - Driving Force')
plt.ylabel('Entropy Production rate')
plt.legend(['InfinityExt'])

plt.show()