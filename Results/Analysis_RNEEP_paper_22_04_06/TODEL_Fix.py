# A nasty merge between differenet sequence runs
import pickle
import numpy as np

for iRun in [17,18,19]:
    with open('AnalysisPaper64_'+str(iRun)+'\\mNeep_x_10.pickle', 'rb') as handle:
        vAdd = pickle.load(handle)
    with open('AnalysisPaper_' + str(iRun) + '\\mNeep_x_10.pickle', 'rb') as handle:
        mRead = pickle.load(handle)

    mWrite = np.concatenate((mRead[:-1], vAdd, np.expand_dims(mRead[-1],0)), axis=0)

    with open('AnalysisPaper_' + str(iRun) + '\\mNeep_x_10.pickle', 'wb') as handle:
        pickle.dump(mWrite, handle)
