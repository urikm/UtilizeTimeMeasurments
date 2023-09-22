import numpy as np
import PhysicalModels.UtilityTraj as ut

def GenRateMat(nDim, limit):
    mW = np.random.uniform(limit, size=(nDim, nDim))
    #mW[3, 2] = np.random.uniform(limit / 10)
    #mW[2, 3] = np.random.uniform(limit / 10)
    for k in range(nDim):
        mW[k,k] = mW[k,k] - np.sum(mW[:,k])

    vHiddenStates = np.array([2, 3])  # states 3 and 4 for 4-D state sytem
    timeRes = 0.01
    return mW, nDim, vHiddenStates, timeRes

def BaseSystem():
    nDim = 4  # dimension of the problem
    mW = np.array([[-11., 2., 0., 1.], [3., -52.2, 2., 35.], [0., 50., -77., 0.7], [8., 0.2, 75., -36.7]])  #, dtype=np.float32)
    vHiddenStates = np.array([2, 3])  # states 3 and 4 for 4-D state sytem
    timeRes = 0.001

    return mW, nDim, vHiddenStates, timeRes

def HiddenControl(hid2to3=0, hid3to2=0, rate0to2=22, rate2to0=11): # hidBond=0 means that hidden states 3-4 are disconnected
    mW, nDim, vHiddenStates, timeRes = BaseSystem()
    mW[2, 2] = mW[2, 2] + mW[3, 2] - hid2to3
    mW[3, 3] = mW[3, 3] + mW[2, 3] - hid3to2
    mW[3, 2] = hid2to3
    mW[2, 3] = hid3to2


    mW[0, 0] = mW[0, 0] + mW[2, 0] - rate0to2
    mW[2, 2] = mW[2, 2] - mW[0, 2] - rate2to0
    mW[2, 0] = rate0to2
    mW[0, 2] = rate2to0

    return mW, nDim, vHiddenStates, timeRes

def RingSystem():
    nDim = 4  # dimension of the problem
    vHiddenStates = np.array([2, 3])  # states 3 and 4 for 4-D state sytem
    timeRes = 1
    mW = ut.GenRateMat(nDim)

    mW[0, 0] = mW[0, 0] - mW[3, 0]
    mW[3, 0] = 0

    mW[1, 1] = mW[1, 1] - mW[1, 2]
    mW[1, 2] = 0

    mW[2, 2] = mW[2, 2] - mW[2, 1]
    mW[2, 1] = 0

    mW[3, 3] = mW[3,3] - mW[0, 3]
    mW[0, 3] = 0

    return mW, nDim, vHiddenStates, timeRes

def MolecularMotor(mu, F):
    l = 1 # 0.1
    r = 1 # 0.1
    u1 = 0.01 # 0.8
    d1 = 0.01 # 0.8
    u2 = 1
    d2 = 1

    l_tmp = l
    r_tmp = r
    u1_tmp = u1*np.exp(-F/2)
    d1_tmp = d1*np.exp(F/2)
    u2_tmp = u2*np.exp(-(F-mu)/2)
    d2_tmp = d2*np.exp((F-mu)/2)
    mW = np.array([[0, r_tmp, d1_tmp, 0, u1_tmp, u2_tmp],
                    [l_tmp, 0, d2_tmp, 0, 0, 0],
                    [u1_tmp, u2_tmp, 0, r_tmp, d1_tmp, 0],
                    [0, 0, l_tmp, 0, d2_tmp, 0],
                    [d1_tmp, 0, u1_tmp, u2_tmp, 0, r_tmp],
                    [d2_tmp, 0, 0, 0, l_tmp, 0]], dtype=float)

    np.fill_diagonal(mW, -np.sum(mW, axis=0).squeeze())

    nDim = mW.shape[0]  # dimension of the problem
    timeRes = 0.05

    vHiddenStates = np.array([0, 1, 2])

    return mW, nDim, vHiddenStates, timeRes
def DataSetCreationParams():
    samplePrefix = 'BatchedSample_'
    fileSuffix = '.pt'
    trainIterInEpoch = 2500 # TODO: make it as parameter of dataset creation(specifically, case of train set creation)
    dsDescriptorName = 'DataSetDescriptor'
    trajFileName = 'CoarseGrainedTrajectory'

    return samplePrefix, fileSuffix, trainIterInEpoch, dsDescriptorName, trajFileName

def ExtForcesGrid(chooseGrid, interpRes=5e-3):
    xSt = 0.6700199322178189  # 0.5703983252253606 #Calculated using Master Equation Solver
    if chooseGrid == 'full':
        vGrid = np.concatenate((np.arange(-2., xSt, 0.3), np.arange(xSt, 3., 0.3)))
        vGridInterp = np.concatenate((np.arange(-2., xSt, interpRes), np.arange(xSt, 3., interpRes)))
        subFolder = 'AnalysisFull_'
    elif chooseGrid == 'coarse':
        vGrid = np.concatenate((np.arange(-1., xSt, 1), np.arange(xSt, 3., 1)))
        vGridInterp = np.concatenate((np.arange(-1., xSt, interpRes), np.arange(xSt, 3., interpRes)))
        subFolder = 'Analysis_'
    elif chooseGrid == 'aroundSt':
        vGrid = np.concatenate((np.arange(xSt-0.02,xSt-1e-8,0.01),np.arange(xSt,xSt+0.02,0.01)))
        vGridInterp = np.concatenate((np.arange(xSt-0.02, xSt-1e-8, interpRes), np.arange(xSt, xSt+0.02, interpRes)))
        subFolder = 'AnalysisSt_'
    elif chooseGrid == 'nearSt':
        vGrid = np.arange(0, xSt,0.15)
        vGridInterp = np.arange(0, xSt, interpRes)
        subFolder = 'AnalysisNearSt_'
        subFolder = 'AnalysisSt_'
    elif chooseGrid == 'zoomed':
        vGrid = np.arange(-1., 0., 0.25)
        vGridInterp = np.arange(-1., 0., interpRes)
        subFolder = 'AnalysisZoomed_'
    elif chooseGrid == 'extended':
        vGrid = np.arange(-2., 1., 0.25)
        vGridInterp = np.arange(-2., 1., interpRes)
        subFolder = 'AnalysisExt_'
    elif chooseGrid == 'converege':
        vGrid = np.concatenate((np.arange(-0.75, xSt, 0.3), np.arange(xSt, 2.25, 0.3)))
        vGridInterp = np.concatenate((np.arange(-0.75, xSt, interpRes), np.arange(xSt, 2.25, interpRes)))
        subFolder = 'AnalysisPaper_'
    elif chooseGrid == 'RNEEPadd': # Also used in the paper
        vGrid = np.array([xSt-0.1, xSt+0.1])
        vGridInterp = np.array([xSt-0.1 ,xSt+0.1]) # DONT CARE
        subFolder = 'AnalysisPaper_'
    else:
        raise Exception("Wrong grid chosen!")

    return vGrid, vGridInterp, subFolder