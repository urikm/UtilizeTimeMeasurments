"""
@title : Reproduce KLD estimators for Flatching Ratchet model : TODO - DELETE, obsolete there is a script in "Results"

@author: Uri Kapustin
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import Utility.FindPluginInfEPR as eprinf
import PhysicalModels.ratchet as rt
from scipy.io import loadmat


# %% Find the ratio of same subsequent states
def SubsequentRatio(vTrajectory):
    totLength = vTrajectory.size
    vSubseqJumps = vTrajectory[1:] == vTrajectory[:-1]
    ratio = vSubseqJumps.sum()/totLength
    return ratio

# %% Recreate different estimators
vPotentials = np.array([0.5, 1, 1.5, 2])#np.linspace(0.5,2.,int(np.floor(1.5/0.25)+1))
nTimeStamps = int(1e7)

vKldInf = np.zeros(vPotentials.shape)
vKldInfFullCg = np.zeros(vPotentials.shape)
vFull = np.zeros(vPotentials.shape)
vRatio = np.zeros(vPotentials.shape)

for iPot, pot in enumerate(vPotentials):
    mCgTrajectory = rt.CreateNEEPTrajectory(nTimeStamps=nTimeStamps, x=pot, fullCg=False)
    print(mCgTrajectory.shape)
    vKldInf[iPot] = eprinf.EstimatePluginInf(mCgTrajectory)
    print('For Potential: '+str(pot)+' we get Inf estimator of: '+str(vKldInf[iPot]))
    vFull[iPot] = rt.ep_per_step(pot)
    # Check the ratio of subsequent jumps
    vRatio[iPot] = SubsequentRatio(mCgTrajectory)
# %% Full cg
for iPot, pot in enumerate(vPotentials):
    mCgTrajectory = rt.CreateNEEPTrajectory(nTimeStamps=int(3*nTimeStamps), x=pot, fullCg=True)
    print(mCgTrajectory.shape)
    vKldInfFullCg[iPot] = eprinf.EstimatePluginInf(mCgTrajectory)
    print('For Potential: '+str(pot)+' we get Inf estimator of: '+str(vKldInfFullCg[iPot]))

# %% Load recorded data from NEEP
mMeans = loadmat('mMeans.mat')
mStds = loadmat('mStds.mat')
mMeans = mMeans['mMeanCheat']
mStds = mStds['mStdCheat']
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

resFig3 = plt.figure(2)
plt.plot(np.flip(vPotentials), 1-np.flip(vRatio), ':m', label='NonSubsequentRatio')
plt.plot(np.flip(vPotentials), np.flip(np.divide(vKldInf, vFull)), ':k', label='SCG-EPR/Full-EPR')
plt.xlabel('x - Driving Force')
plt.ylabel('ratio[a.u]')
plt.legend()
plt.show()

resFig4 = plt.figure(3)
plt.plot(np.flip(vPotentials), np.flip(vKldInf), ':m', label='Plugin')
plt.errorbar(np.flip(vPotentials),np.flip(mMeans[0,:]),yerr=np.flip(mStds[0,:]), fmt='xm', label='RNEEP-seq2')
plt.errorbar(np.flip(vPotentials),np.flip(mMeans[1,:]),yerr=np.flip(mStds[1,:]), fmt='xc', label='RNEEP-seq8')
plt.errorbar(np.flip(vPotentials),np.flip(mMeans[2,:]),yerr=np.flip(mStds[2,:]), fmt='xr', label='RNEEP-seq16')
plt.errorbar(np.flip(vPotentials),np.flip(mMeans[3,:]),yerr=np.flip(mStds[3,:]), fmt='xy', label='RNEEP-seq32')
plt.errorbar(np.flip(vPotentials),np.flip(mMeans[4,:]),yerr=np.flip(mStds[4,:]), fmt='xk', label='RNEEP-seq64')
plt.errorbar(np.flip(vPotentials),np.flip(mMeans[5,:]),yerr=np.flip(mStds[5,:]), fmt='xb', label='RNEEP-seq128')
plt.xlabel('x - Driving Force')
plt.ylabel('Entropy Production rate')
plt.legend()
plt.show()

resFig.set_size_inches((16, 16))
resFig.savefig(
    os.path.join('InfExtrapolatorKLDFR.svg'))
resFig2.set_size_inches((16, 16))
resFig2.savefig(
    os.path.join('InfExtrapolatorKLDlogFR.svg'))
resFig2.set_size_inches((16, 16))
resFig3.savefig(
    os.path.join('FR_ratios.svg'))
resFig4.savefig(
    os.path.join('RNEEPreproduction.svg'))
resFig4.set_size_inches((16, 16))