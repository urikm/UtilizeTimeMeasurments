import numpy as np

def BaseSystem():
    nDim = 4  # dimension of the problem
    mW = np.array([[-11., 2., 0., 1.], [3., -52.2, 2., 35.], [0., 50., -77., 0.7], [8., 0.2, 75., -36.7]])
    vHiddenStates = np.array([2, 3])  # states 3 and 4 for 4-D state sytem
    timeRes = 0.001

    return mW, nDim, vHiddenStates, timeRes

def DataSetCreationParams():
    samplePrefix = 'BatchedSample_'
    fileSuffix = '.pt'
    trainIterInEpoch = 5000
    dsDescriptorName = 'DataSetDescriptor'
    trajFileName = 'CoarseGrainedTrajectory'

    return samplePrefix, fileSuffix, trainIterInEpoch, dsDescriptorName, trajFileName
