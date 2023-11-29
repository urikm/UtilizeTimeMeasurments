# Import
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utility.Params import GenRateMat
from PhysicalModels.PartialTrajectories import CreateCoarseGrainedTraj, CalcKLDPartialEntropyProdRate, RemapStates
from PhysicalModels.UtilityTraj import EntropyRateCalculation
from PhysicalModels.MasterEqSim import MasterEqSolver as MESolver
import Utility.FindPluginInfEPR as infEPR



mResults = pd.read_csv("..\StatisticalCompare.csv")

vDiffResult = mResults.NeepScg - mResults.TrnsKLD
vMask = mResults.NeepScg < mResults.FullEpr
vMask = vMask.to_numpy()
vDiffResult = vDiffResult.to_numpy()

fig = plt.figure(4)
plt.hist(vDiffResult[vMask], 29, density=True, cumulative=True, histtype='bar')
plt.yticks(np.arange(0, 1, 0.1))
plt.grid()
plt.xlabel('$\sigma_{RNEEP} - \sigma_{KLD}$', fontsize='small')
plt.ylabel('Cumulative Distribution Function', fontsize='small')
plt.tick_params(axis="both", labelsize=6)
plt.show()

fig.set_size_inches((3.38582677,  3.38582677))
fig.savefig(f'NeepVsKLD_SCG_CDF.pdf')