import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')

aa = torch.load('dResults.pt')  #torch.load('..\Results\dResults_40k')  #torch.load('..\Results\dResults_g0_40k')  #

fEPR = np.array(aa['F-EPR'])
sEPR = np.array(aa['S-EPR'])
fullEPR = np.array(aa['Full-EPR'])

validInd = np.logical_and(sEPR >= 0, sEPR <= 50)
sEPR = sEPR[validInd]
fEPR= fEPR[validInd]
fullEPR = fullEPR[validInd]
vv=np.maximum(sEPR,fEPR)/fullEPR
# don't show outliers
validInd2 = vv <= 1
sEPR = sEPR[validInd2]
fEPR = fEPR[validInd2]
fullEPR = fullEPR[validInd2]
vv = vv[validInd2]
plt.scatter(fEPR, sEPR, c=vv, cmap='jet')
plt.xlabel('FCG')
plt.ylabel('SCG')
plt.plot([0, 5], [0, 5])
plt.show()

plt.scatter(fEPR, sEPR, c=np.log(fullEPR), cmap='jet')
plt.xlabel('FCG')
plt.ylabel('SCG')
plt.plot([0, 5], [0, 5])
plt.show()