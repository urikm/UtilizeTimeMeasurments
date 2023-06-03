import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')

aa = torch.load('..\Results\dResults_40k_gamma_1e-4_allRandom0_50.pt')  #torch.load('..\Results\dResults_4k_gamma_1e-3.pt')  #torch.load('..\Results\dResults_40k')  #torch.load('..\Results\dResults_g0_40k')  #

fEPR = np.array(aa['F-EPR'])
sEPR = np.array(aa['S-EPR'])
fullEPR = np.array(aa['Full-EPR'])

validInd = np.logical_and(sEPR >= 0, sEPR <= 1000)
sEPR = sEPR[validInd]
fEPR = fEPR[validInd]
fullEPR = fullEPR[validInd]
vv = np.maximum(sEPR, fEPR)/fullEPR * 100
# don't show outliers - extreme outliers, this situation happen because some numerical issues
validInd2 = vv <= 100  # fullEPR >= 25
sEPR = sEPR[validInd2]
fEPR = fEPR[validInd2]
fullEPR = fullEPR[validInd2]
vv = vv[validInd2]
maxGrid = np.max([fEPR.max(), sEPR.max()])

resFig1, ax1 = plt.subplots()
a = ax1.scatter(fEPR, sEPR, c=vv, cmap='jet')
plt.title('SCG vs FCG EPR')
plt.xlabel('EPR-FCG', fontsize='x-large')
plt.ylabel('EPR-SCG', fontsize='x-large')
ax1.plot([0, maxGrid], [0, maxGrid])
resFig1.colorbar(a, label='S-EPR/Full-EPR [%]')
plt.show()
resFig1.set_size_inches((64, 64))
resFig1.savefig(f'SCGvsFCG_ratioOfFullEPR.svg')

resFig2, ax2 = plt.subplots()
a2 = ax2.scatter(fEPR, sEPR, c=fullEPR, cmap='jet')
plt.title('SCG vs FCG EPR')
plt.xlabel('EPR-FCG', fontsize='x-large')
plt.ylabel('EPR-SCG', fontsize='x-large')

ax2.plot([0, maxGrid], [0, maxGrid])
resFig2.colorbar(a2, label='Full-EPR')
plt.show()
resFig2.set_size_inches((64, 64))
resFig2.savefig(f'SCGvsFCG_FullEPR.svg')

resFig3, ax3 = plt.subplots()
a3 = ax3.hist(vv, ec="black", density=1)
# plt.title('Histogram of ')
plt.xlabel('S-EPR/Full-EPR [%]', fontsize='x-large')
plt.ylabel('pdf', fontsize='x-large')
plt.show()
resFig3.set_size_inches((8, 8))
resFig3.savefig(f'SCGvsFCG_histRatios.svg')

