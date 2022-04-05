"""
@title : Visualize the appearance of possible sequences

@author: Uri Kapustin
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

import PhysicalModels.MasterEqSim as mstr
import PhysicalModels.PartialTrajectories as pt
import Utility.FindPluginInfEPR as infEPR
from Utility.Params import ExtForcesGrid, BaseSystem


# %% Test as Uri - Check the portion of observed 1-directional sequences of all the observed sequences(including pairs of observed sequences)
def TAU(vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2, thresh=0.001):
    detectT = 0 # hardcoded of the algorithm - shouldn't be controlled externally
    countB = 0
    countF = 0
    countSeq = 0
    countNon = 0

    while countF < len(vProbOfSeq1) or countB < len(vProbOfSeq2):
        if (countF < len(vProbOfSeq1) and countB < len(vProbOfSeq2)) and vSeqs1[countF] == vSeqs2[countB]:
            countSeq += 1
            countF += 1
            countB += 1
        elif countF >= len(vProbOfSeq1) or (countB < len(vProbOfSeq2) and vSeqs1[countF] > vSeqs2[countB]):
            countNon += 1
            countB += 1
        else:
            countNon += 1
            countF += 1
    # Now calculate score
    scoreU = (countNon)/(countNon+countSeq)
    return scoreU

# Validate TAU criterion by checking that sequences that not seen have lower probability that those that seen
def ValidateTAU(validationDict):
    # Init
    validatedFlag = False
    n1DwithRepeat = 0 # TODO : add logic to count these sequences

    vProb1DSeen = []
    vProb1DUnseen = []
    vProb2DSeen1 = []
    vProb2DSeen2 = []

    vSeq1DSeen = []
    vSeq1DUnseen = []
    vSeq2DSeen1 = []
    vSeq2DSeen2 = []

    countB = 0
    countF = 0
    while countF < len(validationDict['vSeq1']) or countB < len(validationDict['vSeq2']):
        if validationDict['vSeq1'][countF] == validationDict['vSeq2'][countB]:
            # Check probability of sequence and it's reverse
            bwdSeq = FlipDecimal(validationDict['vSeq2'][countB], validationDict['seqLen'])
            theoryProbSeq1 = ProbablityOfSeq(validationDict['vSeq1'][countF], validationDict['seqLen'], validationDict['force'])
            theoryProbSeq2 = ProbablityOfSeq(bwdSeq, validationDict['seqLen'], validationDict['force'])

            vProb2DSeen1.append(theoryProbSeq1)
            vProb2DSeen2.append(theoryProbSeq2)
            vSeq2DSeen1.append(validationDict['vSeq1'][countF])
            vSeq2DSeen2.append(bwdSeq)

            countF += 1
            countB += 1
        elif countF >= len(validationDict['vSeq1'])-1 or validationDict['vSeq1'][countF] > validationDict['vSeq2'][countB]:
            # If only backward sequences observed - it classified as unseen
            bwdSeq = FlipDecimal(validationDict['vSeq2'][countB], validationDict['seqLen'])
            theoryProbUnseen = ProbablityOfSeq(bwdSeq, validationDict['seqLen'], validationDict['force'])

            vProb1DUnseen.append(theoryProbUnseen)
            vSeq1DUnseen.append(bwdSeq)

            countB += 1
        else:
            # the forward sequences are those that defined as "seen", if it's backward doesnt observed - it's classified as unseen
            bwdSeq = FlipDecimal(validationDict['vSeq1'][countF], validationDict['seqLen'])
            theoryProbSSeen = ProbablityOfSeq(validationDict['vSeq1'][countF], validationDict['seqLen'], validationDict['force'])
            theoryProbUnseen = ProbablityOfSeq(bwdSeq, validationDict['seqLen'], validationDict['force'])

            vProb1DSeen.append(theoryProbSSeen)
            vProb1DUnseen.append(theoryProbUnseen)
            vSeq1DSeen.append(validationDict['vSeq1'][countF])
            vSeq1DUnseen.append(bwdSeq)

            countF += 1

        # Check if the probability of unseen is really lower than seen

    debugValidation = {'seq2DSeen1':vSeq2DSeen1, 'seq2DProb1':vProb2DSeen1, 'seq2DSeen2':vSeq2DSeen2, 'seq2DProb2':vProb2DSeen2,
                       'seq1DSeen':vSeq1DSeen, 'seq1DProbS':vProb1DSeen, 'seq1DUnseen':vSeq1DUnseen, 'seq1DProbUn':vProb1DUnseen,
                       'num1DseqWithRepeat': n1DwithRepeat}
    return validatedFlag, debugValidation

# Estimate the probability of a sequence from rate matrix. NOTE: relevant only for "Gili's" basic 4-states systems
def ProbablityOfSeq(seq, seqLen, Force):
    # Retrieve the rate matrix of the relevant system Full system
    mW, nDim, vHiddenStates, timeRes = BaseSystem()
    mWx = pt.CalcW4DrivingForce(mW, Force)

    # Create transition probability matrix between states
    mDiag = np.multiply(np.eye(nDim), mWx) #For competability with jit
    mP = (mWx-mDiag)/np.abs(np.dot(mDiag, np.ones(4))) # calculate discrete PDF for states jumps

    # Get steady state distribution
    vP0 = np.random.uniform(size=(nDim))
    vP0 = vP0 / sum(vP0)
    _, vPiX, _, _ = mstr.MasterEqSolver(nDim, vP0, mWx, timeRes)

    # For loop over sequence to calculate probability of the input sequence.
    prevState = int(seq // 10**(seqLen - 1)) # Extract first state

    if prevState == 2:
        prob = vPiX[2] + vPiX[3]
    else:
        prob = vPiX[prevState]

    seq = seq % 10 ** (seqLen - 1)
    for iStep in range(seqLen - 1):
        currState = int(seq // 10**(seqLen - 2 - iStep))
        # Handle case of jumping to hidden state TODO : make it suitable for more generic system
        if currState == 2 and prevState != 2:
            prob *= (mP[3, prevState]+mP[2, prevState])
        # Handle case of jumping from hidden state TODO : make it suitable for more generic system
        elif currState != 2 and prevState == 2:
            prob *= max(mP[currState, 2], mP[currState, 3]) # approximation, not a lower bound
        # Handle case of jumping between hidden state, for semi CS TODO : make it suitable for more generic system
        elif currState == 2 and prevState == 2:
            prob *= max(mP[3, 2], mP[2, 3]) # approximation, not a lower bound
        # Handle case of jumping between 0<->1
        else:
            prob *= mP[currState, prevState]

        # Update previous state and the residual sequence
        prevState = currState
        seq = seq % 10**(seqLen - 2 - iStep)

    return prob


# Flip decimal number - used to flip the encoded sequences in decimal base
def FlipDecimal(number, nDigits):
    ret = 0
    for iStep in range(nDigits):
        ret = ret * 10
        ret = ret + number % 10
        number = number // 10
    return ret


# %% Systems to analyze
frFlag = False # Perform analysis on flatching ratchet fully CG
fcgFlag = True # Perform analysis on Gili's system fully CG
scgFlag = True # Perform analysis on Gili's system fully CG

# %% Complementary buffers for TAU criterion validation
threshRec = 0.1 # if encountering this value or higher of TAU - record data for validation
frValidate = {'vSeq1':[], 'vSeq2':[], 'TAU':100000, 'force':100000, 'seqLen':100000}
gsValidate = {'vSeq1':[], 'vSeq2':[], 'TAU':100000, 'force':100000, 'seqLen':100000}
gsSCGValidate = {'vSeq1':[], 'vSeq2':[], 'TAU':100000, 'force':100000, 'seqLen':100000}

# %% Visualization params
maxSeq = 9
vSeqSize = np.linspace(2, maxSeq, maxSeq - 2 + 1, dtype=np.intc)#np.array([3, 16, 32, 64, 128], dtype=np.float) #np.array([2, 3, 4, 5, 6, 7], dtype=np.float) #
threshold = 0
naiveTrajLen = int(1e7)
gamma = 1e-17

# %% Visualize flatching ratchet
if frFlag:
    start = 0.5
    end = 2.
    step = 0.25
    vPots = np.linspace(start,end,int(np.floor((end-start)/step)+1))
    mW, nDim, vHiddenStates, timeRes = BaseSystem()
    mFRSys = np.zeros((len(vSeqSize), len(vPots)))
    mEprFr = np.zeros((len(vSeqSize), len(vPots)))
    vEPRfr = np.zeros((len(vPots),))
    for ix, x in enumerate(vPots):
        mCgTraj = infEPR.CreateNEEPTrajectory(nTimeStamps=naiveTrajLen, x=x, fullCg=False)
        for iSeq, seqLen in enumerate(vSeqSize):
            seqEpr, (vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2) = infEPR.EstimatePluginM(mCgTraj.T, int(seqLen), gamma=gamma)
            score = TAU(vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2)
            mFRSys[iSeq, ix] = score
            mEprFr[iSeq, ix] = seqEpr
            if score > threshRec and seqLen < frValidate['seqLen']:
                frValidate['vSeq1'] = vSeqs1
                frValidate['vSeq2'] = vSeqs2
                frValidate['TAU'] = score
                frValidate['force'] = x
                frValidate['seqLen'] = seqLen

        vEPRfr[ix] = infEPR.EstimatePluginInf(mCgTraj.T, maxSeq=maxSeq, gamma=gamma)

# %% Visualize Gili's system
if fcgFlag:
    vGrid, _, _ = ExtForcesGrid('converege',interpRes=1e-3)
    mW, nDim, vHiddenStates, timeRes = BaseSystem()
    mGilisSys = np.zeros((len(vSeqSize), len(vGrid)))
    mEprFcg = np.zeros((len(vSeqSize), len(vGrid))) # hold the plugin estimator for finite sequences
    vEPRgs = np.zeros((len(vGrid),)) # Hold the plugin estimator fit
    for ix, x in enumerate(vGrid):
        mWx = pt.CalcW4DrivingForce(mW, x)
        mCgTraj, nCgDim = pt.CreateCoarseGrainedTraj(nDim, naiveTrajLen, mWx, vHiddenStates, timeRes)
        for iSeq, seqLen in enumerate(vSeqSize):
            seqEpr, (vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2) = infEPR.EstimatePluginM(mCgTraj[:, 0], int(seqLen), gamma=gamma)
            score = TAU(vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2)
            mGilisSys[iSeq, ix] = score
            mEprFcg[iSeq, ix] = seqEpr
            if score > threshRec and seqLen < gsValidate['seqLen']:
                gsValidate['vSeq1'] = vSeqs1
                gsValidate['vSeq2'] = vSeqs2
                gsValidate['TAU'] = score
                gsValidate['force'] = x
                gsValidate['seqLen'] = seqLen

        vEPRgs[ix] = infEPR.EstimatePluginInf(mCgTraj[:, 0], maxSeq=maxSeq, gamma=gamma)

# %% Visualize Gili's system - semi CG
if scgFlag:
    vGrid, _, _ = ExtForcesGrid('converege',interpRes=1e-3)
    mW, nDim, vHiddenStates, timeRes = BaseSystem()
    mGilisSys2 = np.zeros((len(vSeqSize), len(vGrid)))
    mEprScg = np.zeros((len(vSeqSize), len(vGrid))) # hold the plugin estimator for finite sequences
    vEPRscg = np.zeros((len(vGrid),)) # Hold the plugin estimator fit
    for ix, x in enumerate(vGrid):
        mWx = pt.CalcW4DrivingForce(mW, x)
        mCgTraj, nCgDim = pt.CreateCoarseGrainedTraj(nDim, naiveTrajLen, mWx, vHiddenStates, timeRes, semiCG=True)
        for iSeq, seqLen in enumerate(vSeqSize):
            seqEpr, (vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2) = infEPR.EstimatePluginM(mCgTraj[:, 0], int(seqLen), gamma=gamma)
            score = TAU(vSeqs1, vProbOfSeq1, vSeqs2, vProbOfSeq2)
            mGilisSys2[iSeq, ix] = score
            mEprScg[iSeq, ix] = seqEpr
            if score > threshRec and seqLen < gsSCGValidate['seqLen']:
                gsSCGValidate['vSeq1'] = vSeqs1
                gsSCGValidate['vSeq2'] = vSeqs2
                gsSCGValidate['TAU'] = score
                gsSCGValidate['force'] = x
                gsSCGValidate['seqLen'] = seqLen

        vEPRscg[ix] = infEPR.EstimatePluginInf(mCgTraj[:, 0], maxSeq=maxSeq, gamma=gamma)

# %% Plotting
label_format = '{:,.0f}'

if frFlag:
    fig1 = plt.figure()
    # extent = np.min(vSeqSize), np.max(vSeqSize), np.min(vPots), np.max(vPots)
    X, Y = np.meshgrid(vPots, vSeqSize)
    plt.pcolormesh(X, Y, mFRSys, cmap="jet", shading='auto')
    plt.xlabel("External force")
    plt.ylabel("Input sequence length")
    plt.title("Flashing Ratchet")
    plt.colorbar()
    plt.show()
    fig1.set_size_inches((16, 16))
    fig1.savefig(f'seqsFR.png')

if fcgFlag:
    fig2 = plt.figure()
    # extent = np.min(vSeqSize), np.max(vSeqSize), np.min(vGrid), np.max(vGrid)
    X, Y = np.meshgrid(vGrid, vSeqSize)
    plt.pcolormesh(X, Y, mGilisSys, cmap="jet", shading='auto')
    plt.xlabel("External force")
    plt.ylabel("Input sequence length")
    plt.title("Gili's System")
    plt.colorbar()
    plt.show()
    fig2.set_size_inches((16, 16))
    fig2.savefig(f'seqsGilis.png')

if scgFlag:
    fig3 = plt.figure()
    # extent = np.min(vSeqSize), np.max(vSeqSize), np.min(vGrid), np.max(vGrid)
    X, Y = np.meshgrid(np.flip(-vGrid), vSeqSize)
    plt.pcolormesh(X, Y, mGilisSys2, cmap="jet", shading='auto')
    plt.xlabel("External force")
    plt.ylabel("Input sequence length")
    plt.title("Gili's System - semi CG")
    plt.colorbar()
    plt.show()
    fig3.set_size_inches((16, 16))
    fig3.savefig(f'seqsGilis2.png')

if frFlag:
    figT = plt.figure()
    plt.plot(vPots, vEPRfr, label='plugin')
    for idx, iSeq in enumerate(vSeqSize):
        plt.plot(vPots, mEprFr[idx, :]/iSeq, ':', label='Semi_'+str(iSeq))
    plt.title('FR-EPR')
    plt.legend()
    plt.show()
    figT.set_size_inches((16, 16))
    figT.savefig(f'PluginFlatching.png')

if fcgFlag and scgFlag:
    figT2 = plt.figure()
    plt.plot(np.flip(-vGrid), np.flip(vEPRgs), label='FullCG')
    for idx, iSeq in enumerate(vSeqSize):
        plt.plot(np.flip(-vGrid), np.flip(mEprFcg[idx, :]/iSeq), ':', label='Full_'+str(iSeq))
    plt.plot(np.flip(-vGrid), np.flip(vEPRscg), label='SemiCG')
    for idx, iSeq in enumerate(vSeqSize):
        plt.plot(np.flip(-vGrid), np.flip(mEprScg[idx, :])/iSeq, '+', label='Semi_'+str(iSeq))
    plt.title('Gilis-EPR')
    plt.legend()
    plt.show()
    figT2.set_size_inches((16, 16))
    figT2.savefig(f'PluginGili.png')

# %% Save criterion validation data
if frFlag:
    with open('frValidate_TAU', 'wb') as f:
        pickle.dump(frValidate, f)
if fcgFlag:
    with open('gsSCGValidate_TAU', 'wb') as f:
        pickle.dump(gsSCGValidate, f)
if scgFlag:
    with open('gsValidate_TAU', 'wb') as f:
        pickle.dump(gsValidate, f)

# %% Validate automatically - check for each 1-direction sequence it's analytical probability same for 2-direction observation
# for 1-direction we will calculate also the the analytical probability of the unseen direction.
if fcgFlag and scgFlag:
    semiTAUvalidFlag, dbgSemi = ValidateTAU(gsSCGValidate)
    fullTAUvalidFlag, dbgFull = ValidateTAU(gsValidate)

    print("Full CG")
    print("Theoretical statistics of sequences that observed in 2 directions:")
    print('Forward - mean: ', np.mean(dbgFull['seq2DProb1']),' ; std:', np.std(dbgFull['seq2DProb1']), ' ; median: ', np.median(dbgFull['seq2DProb1']))
    print('Backward - mean: ', np.mean(dbgFull['seq2DProb2']),' ; std:', np.std(dbgFull['seq2DProb2']), ' ; median: ', np.median(dbgFull['seq2DProb2']))
    print("Theoretical statistics of sequences that observed in 1 directions:")
    print('Seen direction - mean: ', np.mean(dbgFull['seq1DProbS']),' ; std:', np.std(dbgFull['seq1DProbS']), ' ; median: ', np.median(dbgFull['seq1DProbS']))
    print('Unseen direction - mean: ', np.mean(dbgFull['seq1DProbUn']),' ; std:', np.std(dbgFull['seq1DProbUn']), ' ; median: ', np.median(dbgFull['seq1DProbUn']))
    print("\n")
    print("Semi CG")
    print("Theoretical statistics of sequences that observed in 2 directions:")
    print('Forward - mean: ', np.mean(dbgSemi['seq2DProb1']),' ; std:', np.std(dbgFull['seq2DProb1']), ' ; median: ', np.median(dbgSemi['seq2DProb1']))
    print('Backward - mean: ', np.mean(dbgSemi['seq2DProb2']),' ; std:', np.std(dbgFull['seq2DProb2']), ' ; median: ', np.median(dbgSemi['seq2DProb2']))
    print("Theoretical statistics of sequences that observed in 1 directions:")
    print('Seen direction - mean: ', np.mean(dbgSemi['seq1DProbS']),' ; std:', np.std(dbgFull['seq1DProbS']), ' ; median: ', np.median(dbgSemi['seq1DProbS']))
    print('Unseen direction - mean: ', np.mean(dbgSemi['seq1DProbUn']),' ; std:', np.std(dbgFull['seq1DProbUn']), ' ; median: ', np.median(dbgSemi['seq1DProbUn']))








