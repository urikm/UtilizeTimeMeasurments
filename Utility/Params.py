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

def ExtForcesGrid(chooseGrid, interpRes=5e-3):
    xSt = 0.6700199322178189  # Calculated using Master Equation Solver
    if chooseGrid == 'coarse':
        vGrid = np.concatenate((np.arange(-1., xSt, 1), np.arange(xSt, 3., 1)))
        vGridInterp = np.concatenate((np.arange(-1., xSt, interpRes), np.arange(xSt, 3., interpRes)))
    elif chooseGrid == 'aroundSt':
        vGrid = np.concatenate((np.arange(xSt-0.02,xSt-0.005,0.01),np.arange(xSt,xSt+0.02,0.01)))
        vGridInterp = np.concatenate((np.arange(xSt-0.02, xSt-0.005, interpRes), np.arange(xSt, xSt+0.02, interpRes)))
    elif chooseGrid == 'nearSt':
        vGrid = np.arange(0, xSt,0.15)
        vGridInterp = np.arange(0, xSt, interpRes)
    elif chooseGrid == 'zoomed':
        vGrid = np.arange(-1., 0., 0.25)
        vGridInterp = np.arange(-1., 0., interpRes)
    elif chooseGrid == 'extended':
        vGrid = np.arange(-2., 1., 0.25)
        vGridInterp = np.arange(-2., 1., interpRes)
    else:
        raise Exception("Wrong grid chosen!")

    return vGrid, vGridInterp