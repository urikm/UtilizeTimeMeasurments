## Copied from neep github
import numpy as np
from PhysicalModels.TrajectoryCreation import CreateTrajectory
import PhysicalModels.PartialTrajectories as pt

def transition_matrix(V, T=1, r=1):
    """Transition matrix of discrete flashing ratchet model.
    
    Args:
        V : potential value
        T : Temperature of the heat bath (default: 1)
        r : switching rate (default: 1)

    Returns:
        probability matrix
    """

    sum0 = np.exp(-V / (2 * T)) + np.exp(-V / T) + r
    sum1 = np.exp(V / (2 * T)) + np.exp(-V / (2 * T)) + r
    sum2 = np.exp(V / T) + np.exp(V / (2 * T)) + r
    sum3 = 2 + r
    sum4 = 2 + r
    sum5 = 2 + r

    P = np.array(
        [
            [0, np.exp(-V / (2 * T)) / sum0, np.exp(-V / T) / sum0, r / sum0, 0, 0],
            [
                np.exp(V / (2 * T)) / sum1,
                0,
                np.exp(-V / (2 * T)) / sum1,
                0,
                r / sum1,
                0,
            ],
            [np.exp(V / T) / sum2, np.exp(V / (2 * T)) / sum2, 0, 0, 0, r / sum2],
            [r / sum3, 0, 0, 0, 1 / sum3, 1 / sum3],
            [0, r / sum4, 0, 1 / sum4, 0, 1 / sum4],
            [0, 0, r / sum5, 1 / sum5, 1 / sum5, 0],
        ]
    )
    return P

####### Uri's addition
def rate_matrix(V, T=1, r=1):
    """Rate matrix of discrete flashing ratchet model.

    Args:
        V : potential value
        T : Temperature of the heat bath (default: 1)
        r : switching rate (default: 1)

    Returns:
        rate matrix (rows are output states and colomns input, opposite from transition matrix)
    """

    sum0 = np.exp(-V / (2 * T)) + np.exp(-V / T) + r
    sum1 = np.exp(V / (2 * T)) + np.exp(-V / (2 * T)) + r
    sum2 = np.exp(V / T) + np.exp(V / (2 * T)) + r
    sum3 = 2 + r
    sum4 = 2 + r
    sum5 = 2 + r

    vSum = np.array([sum0, sum1, sum2, sum3, sum4, sum5])
    mP = transition_matrix(V, T=T, r=r)
    mW = mP.T * vSum

    for k in range(6): # 6 is the dimension of states in this FR system
        mW[k, k] = mW[k, k] - np.sum(mW[:, k])
    return mW
#######


def simulation(num_trjs, trj_len, V, T=1, r=1, seed=0):
    """Simulation of a discrete flashing ratchet.
    
    Args:
        num_trjs : Number of trajectories you want
        trj_len : length of trajectories
        V : potential value
        T : Temperature of the heat bath (default: 1)
        r : switching rate (default: 1)
        seed : seed of random generator (default: 0)

    Returns:
        trajectories of a discrete flashing ratchet model.
        So, its shape is (num_trjs, trj_len)
    """
    P = transition_matrix(V, T, r)
    trajs = []
    states = np.random.choice(6, size=(num_trjs,), p=p_ss(V, T, r))
    trajs.append(states)

    for i in range(trj_len - 1):
        mc = np.random.uniform(0.0, 1.0, size=(num_trjs, 1))
        intervals = np.cumsum(P[states], axis=1)
        next_state = np.sum(intervals < mc, axis=1)
        trajs.append(next_state)
        states = next_state
    return np.array(trajs).T


def ep_per_step(V, T=1, r=1):
    """Analytic average entropy production per step
    
    Args:
        V : potential value
        T : Temperature of the heat bath (default: 1)
        r : switching rate (default: 1)

    Returns:
        analytic average entropy production per step
    """
    P = transition_matrix(V, T, r)
    stationary = p_ss(V, T, r)
    entropy_per_step = (
        (-P[0][1] * V * stationary[0])
        + (-P[0][2] * 2 * V * stationary[0])
        + (P[1][0] * V * stationary[1])
        + (-P[1][2] * V * stationary[1])
        + (P[2][0] * 2 * V * stationary[2])
        + (P[2][1] * V * stationary[2])
    )
    return entropy_per_step


def analytic_etpy(trj, V, T=1, r=1):
    """Analytic stochastic entropy production over a single trajectory.
    
    Args:
        trj : a single trajectory
        V : potential value
        T : Temperature of the heat bath (default: 1)
        r : switching rate (default: 1)

    Returns:
        analytic stochastic entropy production trajectory
    """
    etpy = []
    tran = transition_matrix(V, T, r)
    p = p_ss(V, T, r)

    for it in range(len(trj) - 1):
        temp = (
            p[trj[it + 1]]
            * tran[trj[it], trj[it + 1]]
            / (p[trj[it]] * tran[trj[it + 1], trj[it]])
        )
        etpy.append(np.log(temp))

    return np.array(etpy)


def p_ss(V, T=1, r=1):
    """Array of probability density in steady state.
    
    Args:
        V : potential value
        T : Temperature of the heat bath (default: 1)
        r : switching rate (default: 1)

    Returns:
        array of steady state probability density
    """
    steady = np.array(
        [
            (
                (1 + np.exp(1) ** (V / (2 * T)) + np.exp(1) ** (V / T) * r)
                * (
                    3 * np.exp(1) ** (V / (2 * T)) * r ** 2
                    + r * (3 + r)
                    + np.exp(1) ** ((2 * V) / T) * (3 + r) ** 2
                    + 3 * np.exp(1) ** ((3 * V) / (2 * T)) * (3 + 4 * r + r ** 2)
                    + np.exp(1) ** (V / T) * (9 + 15 * r + 4 * r ** 2)
                )
            )
            / (
                (2 + r)
                * (
                    3
                    + 4 * r
                    + r ** 2
                    + np.exp(1) ** ((3 * V) / T) * (3 + r)
                    + np.exp(1) ** ((5 * V) / (2 * T)) * (3 + 4 * r)
                    + np.exp(1) ** ((2 * V) / T) * (6 + 8 * r + r ** 2)
                    + np.exp(1) ** ((3 * V) / (2 * T)) * (3 + r + 3 * r ** 2)
                    + np.exp(1) ** (V / (2 * T)) * (3 + 7 * r + 3 * r ** 2)
                    + np.exp(1) ** (V / T) * (6 + 11 * r + 4 * r ** 2)
                )
            ),
            (
                (1 + np.exp(1) ** (V / T) + np.exp(1) ** (V / (2 * T)) * r)
                * (
                    r * (3 + r)
                    + np.exp(1) ** ((2 * V) / T) * r * (3 + r)
                    + 3 * np.exp(1) ** (V / (2 * T)) * (3 + 4 * r + r ** 2)
                    + 3 * np.exp(1) ** ((3 * V) / (2 * T)) * (3 + 4 * r + r ** 2)
                    + np.exp(1) ** (V / T) * (9 + 6 * r + 4 * r ** 2)
                )
            )
            / (
                (2 + r)
                * (
                    3
                    + 4 * r
                    + r ** 2
                    + np.exp(1) ** ((3 * V) / T) * (3 + r)
                    + np.exp(1) ** ((5 * V) / (2 * T)) * (3 + 4 * r)
                    + np.exp(1) ** ((2 * V) / T) * (6 + 8 * r + r ** 2)
                    + np.exp(1) ** ((3 * V) / (2 * T)) * (3 + r + 3 * r ** 2)
                    + np.exp(1) ** (V / (2 * T)) * (3 + 7 * r + 3 * r ** 2)
                    + np.exp(1) ** (V / T) * (6 + 11 * r + 4 * r ** 2)
                )
            ),
            (
                (np.exp(1) ** (V / (2 * T)) + np.exp(1) ** (V / T) + r)
                * (
                    3 * np.exp(1) ** ((3 * V) / (2 * T)) * r ** 2
                    + np.exp(1) ** ((2 * V) / T) * r * (3 + r)
                    + (3 + r) ** 2
                    + 3 * np.exp(1) ** (V / (2 * T)) * (3 + 4 * r + r ** 2)
                    + np.exp(1) ** (V / T) * (9 + 15 * r + 4 * r ** 2)
                )
            )
            / (
                (2 + r)
                * (
                    3
                    + 4 * r
                    + r ** 2
                    + np.exp(1) ** ((3 * V) / T) * (3 + r)
                    + np.exp(1) ** ((5 * V) / (2 * T)) * (3 + 4 * r)
                    + np.exp(1) ** ((2 * V) / T) * (6 + 8 * r + r ** 2)
                    + np.exp(1) ** ((3 * V) / (2 * T)) * (3 + r + 3 * r ** 2)
                    + np.exp(1) ** (V / (2 * T)) * (3 + 7 * r + 3 * r ** 2)
                    + np.exp(1) ** (V / T) * (6 + 11 * r + 4 * r ** 2)
                )
            ),
            (
                3
                + r
                + np.exp(1) ** (V / (2 * T)) * (3 + 4 * r)
                + np.exp(1) ** ((3 * V) / T) * (3 + 4 * r + r ** 2)
                + np.exp(1) ** (V / T) * (6 + 8 * r + r ** 2)
                + np.exp(1) ** ((3 * V) / (2 * T)) * (3 + r + 3 * r ** 2)
                + np.exp(1) ** ((5 * V) / (2 * T)) * (3 + 7 * r + 3 * r ** 2)
                + np.exp(1) ** ((2 * V) / T) * (6 + 11 * r + 4 * r ** 2)
            )
            / (
                3
                + 4 * r
                + r ** 2
                + np.exp(1) ** ((3 * V) / T) * (3 + r)
                + np.exp(1) ** ((5 * V) / (2 * T)) * (3 + 4 * r)
                + np.exp(1) ** ((2 * V) / T) * (6 + 8 * r + r ** 2)
                + np.exp(1) ** ((3 * V) / (2 * T)) * (3 + r + 3 * r ** 2)
                + np.exp(1) ** (V / (2 * T)) * (3 + 7 * r + 3 * r ** 2)
                + np.exp(1) ** (V / T) * (6 + 11 * r + 4 * r ** 2)
            ),
            (
                3
                + r
                + np.exp(1) ** ((3 * V) / T) * (3 + r)
                + np.exp(1) ** (V / (2 * T)) * (3 + 4 * r + r ** 2)
                + np.exp(1) ** ((5 * V) / (2 * T)) * (3 + 4 * r + r ** 2)
                + np.exp(1) ** (V / T) * (6 + 11 * r + 3 * r ** 2)
                + np.exp(1) ** ((2 * V) / T) * (6 + 11 * r + 3 * r ** 2)
                + np.exp(1) ** ((3 * V) / (2 * T)) * (3 + 4 * r + 4 * r ** 2)
            )
            / (
                3
                + 4 * r
                + r ** 2
                + np.exp(1) ** ((3 * V) / T) * (3 + r)
                + np.exp(1) ** ((5 * V) / (2 * T)) * (3 + 4 * r)
                + np.exp(1) ** ((2 * V) / T) * (6 + 8 * r + r ** 2)
                + np.exp(1) ** ((3 * V) / (2 * T)) * (3 + r + 3 * r ** 2)
                + np.exp(1) ** (V / (2 * T)) * (3 + 7 * r + 3 * r ** 2)
                + np.exp(1) ** (V / T) * (6 + 11 * r + 4 * r ** 2)
            ),
            1,
        ]
    )
    steady = steady / steady.sum()
    return steady


# %% Create Flashing Ratchet CG trajectory  ;   Note: deprecates use of simulation
def CreateNEEPTrajectory(nTimeStamps=5e4, V=0., fullCg=False, isCG=True, remap=False):
    T=1
    r=1
    nTimeStamps = int(nTimeStamps)
    nDim = 6
    tauFactorBeforeRemap = -1

    initState = np.random.choice(nDim, 1, p=p_ss(V, T=T, r=r)).item()
    mW = rate_matrix(V, T=T, r=r)

    mCgTrajectory, _ = CreateTrajectory(nDim, nTimeStamps, initState, mW) # Create FR trajectory
    vHiddenStates = np.array([])
    if isCG:
        mCgTrajectory[:, 0] = mCgTrajectory[:, 0] % 3  # Semi-CG

        if not fullCg: # in case of semi-CG we want to calculate Tau factor before remaping
            tauFactorBeforeRemap = mCgTrajectory[:, 1].mean()
        if fullCg or (not fullCg and remap):
            # prepare the trajectory -if semiCG make pseudo CG for than remapping. in FullCG it does al the CG
            trainBuffer = np.ones(mCgTrajectory[:, 0].size)
            for iStep in range(mCgTrajectory[:, 0].size):
            # now decimate train and test set
                # handle special case of first element
                if iStep == 0:
                    continue
                # create mask for train set
                if mCgTrajectory[iStep-1, 0] == mCgTrajectory[iStep, 0]:
                    trainBuffer[iStep-1] = 0
                    mCgTrajectory[iStep, 1] += mCgTrajectory[iStep - 1, 1]

        if fullCg:
            mCgTrajectory = mCgTrajectory[trainBuffer == 1, :]
            tauFactorBeforeRemap = mCgTrajectory[:, 1].mean()

        vStates = np.unique(mCgTrajectory[:, 0])
        nDim = vStates.size
        vHiddenStates = vStates
        if remap and not fullCg:  # for each state we need to repeat the remap process
            vHiddenStates = np.array([])
            for iHid in vStates:
                mCgTrajectory, nDim, vNewHidden, countAdded = pt.RemapStates(mCgTrajectory, iHid, baseState=int((iHid + 1) * 100))
                # Add the new vHiddenStates
                vHiddenStates = np.concatenate((vHiddenStates, vNewHidden[:countAdded]), axis=0)
            nDim = np.unique(mCgTrajectory[:, 0]).size
    return mCgTrajectory, nDim, vHiddenStates, tauFactorBeforeRemap
