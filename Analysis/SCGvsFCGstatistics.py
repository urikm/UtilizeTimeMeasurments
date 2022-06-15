import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')

aa = torch.load('..\Results\dResults_4k_1e-4.pt')  #torch.load('..\Results\dResults_4k_gamma_1e-3.pt')  #torch.load('..\Results\dResults_40k')  #torch.load('..\Results\dResults_g0_40k')  #

fEPR = np.array(aa['F-EPR'])
sEPR = np.array(aa['S-EPR'])
fullEPR = np.array(aa['Full-EPR'])

validInd = np.logical_and(sEPR >= 0, sEPR <= 50)
sEPR = sEPR[validInd]
fEPR = fEPR[validInd]
fullEPR = fullEPR[validInd]
vv = np.maximum(sEPR, fEPR)/fullEPR * 100
# don't show outliers
validInd2 = vv <= 100
sEPR = sEPR[validInd2]
fEPR = fEPR[validInd2]
fullEPR = fullEPR[validInd2]
vv = vv[validInd2]

resFig1, ax1 = plt.subplots()
a = ax1.scatter(fEPR, sEPR, c=vv, cmap='jet')
plt.title('SCG vs FCG EPR')
plt.xlabel('EPR-FCG', fontsize='x-large')
plt.ylabel('EPR-SCG', fontsize='x-large')
ax1.plot([0, 5], [0, 5])
resFig1.colorbar(a, label='max(S-EPR,F-EPR)/Full-EPR [%]')
plt.show()
resFig1.set_size_inches((64, 64))
resFig1.savefig(f'SCGvsFCG_ratioOfFullEPR.svg')

resFig2, ax2 = plt.subplots()
a2 = ax2.scatter(fEPR, sEPR, c=fullEPR, cmap='jet')
plt.title('SCG vs FCG EPR')
plt.xlabel('EPR-FCG', fontsize='x-large')
plt.ylabel('EPR-SCG', fontsize='x-large')
ax2.plot([0, 5], [0, 5])
resFig2.colorbar(a2, label='Full-EPR')
plt.show()
resFig2.set_size_inches((64, 64))
resFig2.savefig(f'SCGvsFCG_FullEPR.svg')