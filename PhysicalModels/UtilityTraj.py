# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 20:36:13 2020

@ Utility Trajectory simulation
@author: Uri
"""
import numpy as np
import random
from itertools import product

import torch

# %% Generate Rate matrix 
def GenRateMat(nDim):
    mW = np.random.uniform(size=(nDim,nDim))
    mW = mW/nDim # Normalize
    for k in range(nDim):
        mW[k,k] = mW[k,k] - np.sum(mW[:,k])
    return mW

# Coarse grain rate matrix by combining hidden states to single state.
# Note: there is no importance for dweling rate for the hidden state because its not Poisson distributed
# TODO : write it more general by using the chosen hidden states
def CgRateMatrix(mW,vHidden):
    # Inits
    nDim = np.size(mW,0) # We can take the '1' dimension, doesnt matter for rate matrix
    nDimCg = nDim-(np.size(vHidden)-1)
    mWCg = np.zeros([nDimCg,nDimCg])
    mWCg[0:2,0:2] = mW[0:2,0:2] 
    mWCg[2,0] = mW[2,0] + mW[3,0] 
    mWCg[2,1] = mW[2,1] + mW[3,1]  
    mWCg[0,2] = mW[0,2] + mW[0,3] 
    mWCg[1,2] = mW[1,2] + mW[1,3]      
    # for iState in range(nDim):
    #     if iState in vHidden:
    #     else:
    #         mWCg[:,]
    return mWCg
           
# %% Proceed step of Master equation defined by 'timeJump'
def MasterEqStep(mW,vP,timeJump):
    vPdot = np.dot(mW,vP) # Pdot = W*P
    return vP + vPdot*timeJump


# %% Calculate Entropy Rate from Steady-State probabilities and rate matrix
def EntropyRateCalculation(nDim,mW,vPi):
    entropyRate = 0 
    for iRow in range(nDim):
        for iCol in range(iRow+1,nDim):
            Jij,J_j2i,J_i2j = CalcSteadyStateCurrent(mW,vPi,iCol,iRow)
            if np.abs(Jij) > 0:
                entropyRate += Jij*np.log(J_j2i/J_i2j)
    return entropyRate

def CalcSteadyStateCurrent(mW,vPi,iState,jState):
    # iState is inbound state and jState is outbound state
    # vPi - steady state distribution
    J_j2i = mW[iState,jState]*vPi[jState]
    J_i2j = mW[jState,iState]*vPi[iState]    
    Jij = J_j2i-J_i2j
    return Jij,J_j2i,J_i2j

# Calculate Kintec bound as mentioned in Rahav's 2021 "Thermodynamic uncertainty relation for first-passage times on Markov chains"
def CalcKineticBoundEntProdRate(mW,vPi):
    nDim = np.size(vPi)
    kineticBound = 0 # Init
    for iState in range(1):#range(nDim): 
        # kineticBound += np.sum(np.dot(np.delete(mW[:,iState],iState),vPi[iState]))
        kineticBound = mW[iState+1,iState]*vPi[iState]+mW[iState,iState+1]*vPi[iState+1]
    return kineticBound

class CartesianSampler(object):
    """Random subset sampling from {0, 1, ..., M-1} X {0, 1, ..., L-2} where X is Cartesian product.

    Attributes:
        M: number of trajectories.
        L: trajectory length.
        batch_size: input batch size for training the NEEP.
        train: if True randomly sample a subset else ordered sample. (default: True)

    Examples::

        >>> # 10 trajectories, trajectory length 100, batch size 32 for training
        >>> sampler = CartesianSampler(10, 100, 32) 
        >>> batch, next_batch = next(sampler)

        >>> # 5 trajectories, trajectory length 50, batch size 32 for test
        >>> test_sampler = CartesianSampler(5, 50, 32, train=False) 
        >>> batch, next_batch = next(test_sampler)
        >>> for batch, next_batch in test_sampler:
        >>>     print(batch, next_batch)
    """

    def __init__(self, M, L, batch_size, device="cpu", train=True):
        self.size = M * (L - 1)
        self.M = M
        self.L = L
        self.batch_size = batch_size
        self.device = device
        self.training = train
        self.index = 0

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.training:
            ens_idx = torch.randint(self.M, (self.batch_size,), device=self.device)
            traj_idx = torch.randint(
                0, (self.L - 1), (self.batch_size,), device=self.device
            )
            batch = (ens_idx, traj_idx)
            next_batch = (ens_idx, traj_idx + 1)
            return batch, next_batch
        else:
            prev_idx = self.index * self.batch_size
            next_idx = (self.index + 1) * self.batch_size
            if prev_idx >= self.size:
                raise StopIteration
            elif next_idx >= self.size:
                next_idx = self.size
            ens_idx = torch.arange(prev_idx, next_idx, device=self.device) // (
                self.L - 1
            )
            traj_idx = torch.arange(prev_idx, next_idx, device=self.device) % (
                self.L - 1
            )
            self.index += 1
            batch = (ens_idx, traj_idx)
            next_batch = (ens_idx, traj_idx + 1)
            return batch, next_batch


class CartesianSeqSampler(CartesianSampler):
    def __init__(self, M, L, n, batch_size, device="cpu", train=True):
        """Random subset with sequence length of n sampling from 
           {0, 1, ..., M-1} X {0, 1, ..., L-n} where X is Cartesian product.

        Attributes:
            M: number of trajectories.
            L: trajectory length.
            n: sequence length.
            batch_size: input batch size for training the RNEEP.
            train: if True randomly sample a subset else ordered sample. (default: True)

        Examples::

            >>> # 10 trajectories, trajectory length 100, sequence length 64, batch size 32 for training
            >>> sampler = CartesianSeqSampler(10, 100, 64, 32) 
            >>> batch = next(sampler)

            >>> # 5 trajectories, trajectory length 50, sequence length 32, batch size 64 for test
            >>> test_sampler = CartesianSeqSampler(5, 50, 32, 64, train=False) 
            >>> batch = next(test_sampler)
            >>> for batch in test_sampler:
            >>>     print(batch)
        """

        super().__init__(
            M, L, batch_size, device=device, train=train
        )
        self.size = M * (L - n + 1)
        self.dropL = torch.arange(0, L, n - 1)[-1].item()
        self.test_size = self.dropL * M
        self.seq_length = n
        self.trj_idx = torch.ones(
            self.seq_length, self.batch_size, device=self.device, dtype=torch.long
        )
        self.trj_idx = torch.cumsum(self.trj_idx, 0) - 1

    def __next__(self):
        if self.training:
            ens_idx = torch.randint(self.M, (self.batch_size,), device=self.device)
            traj_idx = torch.randint(
                0,
                (self.L - self.seq_length + 1),
                (self.batch_size,),
                device=self.device,
            )
            traj_idx = self.trj_idx + traj_idx
            return (ens_idx, traj_idx)
        else:
            prev_idx = self.index * self.batch_size * (self.seq_length - 1)
            if prev_idx >= self.test_size:
                raise StopIteration

            next_idx = prev_idx + self.batch_size * (self.seq_length - 1)
            isLast = next_idx > self.test_size
            if isLast:
                batch_size = (self.test_size - prev_idx) // (self.seq_length - 1)
                next_idx = prev_idx + batch_size * (self.seq_length - 1)

            ens_idx = (
                torch.arange(
                    prev_idx, next_idx, self.seq_length - 1, device=self.device
                )
                // self.dropL
            )
            traj_idx = (
                torch.arange(
                    prev_idx, next_idx, self.seq_length - 1, device=self.device
                )
                % self.dropL
            )

            if isLast:
                traj_idx = self.trj_idx[:, : len(traj_idx)] + traj_idx
            else:
                traj_idx = self.trj_idx + traj_idx
            self.index += 1
            batch = (ens_idx, traj_idx)
            return batch
