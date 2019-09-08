#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#################################
# This script includes all functions we used in the main script
#################################
import random
import numpy as np
import time
import os
import pandas as pd
import glob



# This function does the data splitting step given any dataset
# Dat: n by d dataset
# Prob: probability of testdata

def SplitData(Dat, Prob = 0.5):
    n = np.size(Dat, 0)
    nn = int(n * Prob)
    a = random.sample(range(n), nn)
    TrainDat = Dat[a,:]
    TestDat = Dat[list(set(range(n)) - set(a)), :]
    return TrainDat, TestDat

    


# This function implements our proposed algorithm in Stage I: initialization
# hatSigmaX: sample covariance in one group
# hatSigmaY: sample covariance in one group
# alphahat, shat, rhat: tuning parameter
# n: sample size

def OurInitial(hatSigmaX, hatSigmaY, alphahat, shat, rhat, n):
    d = np.size(hatSigmaX, 0)
    TildeSigmaX = (n-1)/(n-d+2)*hatSigmaX
    TildeSigmaY = (n-1)/(n-d+2)*hatSigmaY
    Deltahat = np.linalg.inv(TildeSigmaX) - np.linalg.inv(TildeSigmaY)
    JsDeltahat = HardTrun(Deltahat, shat)
    S0 = PropTrun(JsDeltahat, alphahat)
    if rhat >= 1:
        R0 = Deltahat - S0
        U0, r1hat = LtoU(R0, rhat, 0)
        L0 = UtoL(U0, r1hat)
        return S0, U0, L0, r1hat
    else:
        return S0, np.zeros((d, 0)), np.zeros((d,d)), 0


# This function defines the hard truncation on the symmetric matrix given a threshold index
# (We only truncate the off-diagonal entries)
# M: matrix
# s: threshold index (>=dimension d)

def HardTrun(M, s):
    d = np.size(M, 0)
    ThresIndex = min((s - d)//2, d*(d-1)//2)
    DiagM = np.diag(np.diag(M))
    UOffM = np.triu(M) - DiagM
    if ThresIndex == 0:
        return DiagM
    else:
        Thres = np.sort(np.abs(np.ravel(UOffM)))[::-1][ThresIndex-1]
        UOffM[np.abs(UOffM)<Thres] = 0
        return UOffM + UOffM.T + DiagM




# This function defines the proportion truncation on a symmetric matrix given a threshold proportion
# (We only truncate the off-diagonal entries)
# M: matrix
# alpha: threshold proportion (0<alpha<1)

def PropTrun(M, alpha):
    d = np.size(M, 0)
    ThresIndex = int(alpha*d)
    DiagM = np.diag(np.diag(M))
    if ThresIndex == 0:
        return DiagM
    else:
        OffM = M - DiagM
        ThresM = np.flip(np.sort(np.abs(OffM)),axis = 1)[:,ThresIndex-1].reshape((d,1))
        IdM = (np.abs(OffM)>=ThresM)
        return DiagM + OffM*IdM*IdM.T
        

# This function outputs U by given L, L = ULambdaU^T
# L: low-rank component
# rhatThres: either rhat or threshold
# Id: Id = 0: given rhat
#     Id = 1: given thres

def LtoU(L, rhatThres, Id):
    EigVal, EigVec = np.linalg.eig(L)
    if Id == 0:
        ThreseigVal = np.sort(np.abs(EigVal))[::-1][rhatThres-1]
        IdEig = (np.abs(EigVal)>=ThreseigVal)
        EigVal = EigVal[IdEig]
        EigVec = EigVec[:,IdEig]
        IdEig = EigVal.argsort()[::-1]
        EigVal = EigVal[IdEig]
        EigVec = EigVec[:,IdEig]
        EigVec = EigVec.real
        U = np.matmul(EigVec, np.diag((np.abs(EigVal))**(1/2)))
        r1 = sum(EigVal>0)
        return U, r1
    elif Id == 1:
        IdEig = (np.abs(EigVal)>=rhatThres)
        EigVal = EigVal[IdEig]
        EigVec = EigVec[:,IdEig]
        IdEig = EigVal.argsort()[::-1]
        EigVal = EigVal[IdEig]
        EigVec = EigVec[:,IdEig]
        EigVec = EigVec.real
        U = np.matmul(EigVec, np.diag((np.abs(EigVal))**(1/2)))
        r = len(EigVal)
        r1 = sum(EigVal>0)
        return U, r, r1




# This function calculates the lowrank matrix by given U and r1
# U: low-rank factor
# r1: positive index of inertia

def UtoL(U, r1):
    r = np.size(U, 1)
    Lambda = np.concatenate((np.concatenate((np.eye(r1),np.zeros((r1,r-r1))), axis = 1),\
                    np.concatenate((np.zeros((r-r1,r1)),-1*np.eye(r-r1)), axis = 1)),axis = 0)
    L = np.matmul(U, np.matmul(Lambda, U.T))
    return L



class SimuResult:
    def __init__(self): 
        self.Param = [0,0,0,0,0]  # parameter setup: betahat, alphahat, shatprob, rhat, r1hat
        self.CostTime =  0        # cost time
        self.Idend = 1            # indicator of convergence or not
        self.TrueError = []       # true error iteration
        self.ObjLoss = []         # objective loss
        self.Sparse = np.zeros(0) # output: sparse component
        self.Lowrank = np.zeros(0)# output: lowrank component
        self.tSparse = np.zeros(0)  # true sparse component
        self.tLowrank = np.zeros(0) # true lowrank component
        self.Iternum = 0          # iteration number
        self.Gencomb = [0,0,0]    # combination of n, d, r,
        self.Rank = [0, 0]        # true rank and positive index of inertia





# This function calculates the true error
# TrueError: list to save error
# S0, L0: one pair
# tSparse, tLowrank: true pair

def CalTrueErr(TrueError, S0, L0, tSparse, tLowrank):
    Err1 = np.linalg.norm(S0 - tSparse, 'fro')
    Err2 = np.linalg.norm(L0 - tLowrank,'fro')
    Err3 = np.linalg.norm(tSparse + tLowrank - (S0 + L0), 'fro')
    if np.linalg.norm(tLowrank) < 1e-6:
        Scale = 1
    else:
        Scale = np.linalg.norm(tLowrank,2)**(1/2)
    
    TrueError.append([Err1, Err2, Err3/Scale])
    return TrueError


# This function calculates the loss given sample covariance of two groups
# ObjLoss: list to save loss
# hatSigmaX, hatSigmaY: sample covariance of two groups
# S0, U0, L0, r1hat: sparse component and lowrank component

def LossSampCov(ObjLoss, hatSigmaX, hatSigmaY, S0, U0, L0, r1hat):
    DeltaHat = S0 + L0
    hatDiff = hatSigmaY - hatSigmaX
    A = np.linalg.norm(np.matmul(U0[:,0:r1hat].T,  U0[:,r1hat:]),'fro')**2/2
    B = np.matmul(DeltaHat, np.matmul(hatSigmaX, np.matmul(DeltaHat, hatSigmaY)))
    C = np.matmul(DeltaHat, hatDiff)
    D = 0.5 * np.trace(B) - np.trace(C) + A
    ObjLoss.append(D)
    return ObjLoss




# This function generates Lambda by given r1 and r
# r: rank
# r1: positive index of inertia (r1<r0

def genLambda(r, r1):
    Lambda = np.concatenate((np.concatenate((np.eye(r1),np.zeros((r1,r-r1))), axis = 1),\
                    np.concatenate((np.zeros((r-r1,r1)),-1*np.eye(r-r1)), axis = 1)),axis = 0)
    return Lambda





# This function does the projection for low-rank matrix to satisfy incoherence condition
# M: d by r matrix
# B: incoherence condition bound

def ProjInco(M, B):
    RowNorm = np.linalg.norm(M, axis = 1)
    Idnorm = RowNorm > B
    RowNorm[Idnorm] = B/RowNorm[Idnorm]
    RowNorm[(1-Idnorm).astype(bool)] = 1
    return np.matmul(np.diag(RowNorm), M)



# This function defines the total error distance 
# S0, U0: one pair
# S1, U1: one pair
# r1hat: positive index of inertia
# rhat: rank

def MaxDist(S0, U0, S1, U1, r1hat, rhat):
    if rhat == 0:
        Scale = 1
    else:
        Scale = np.linalg.norm(U1, 2)**2
            
    D1 = np.linalg.norm(S0 - S1,'fro')**2/Scale
    U01 = U0[:,0:r1hat]
    U02 = U0[:,r1hat:]
    U11 = U1[:,0:r1hat]
    U12 = U1[:,r1hat:]
    UU1 = np.matmul(U01.T, U11)
    UU2 = np.matmul(U02.T, U12)
    if r1hat > 0:
        D21 = np.linalg.norm(U01)**2 + np.linalg.norm(U11)**2 - 2*np.linalg.norm(UU1,'nuc')
    else:
        D21 = 0
    
    if rhat - r1hat > 0:
        D22 = np.linalg.norm(U02)**2 + np.linalg.norm(U12)**2 - 2*np.linalg.norm(UU2,'nuc')
    else:
        D22 = 0
    
    D2 = D21 + D22
    return max(D1, D2)



 
# This function implements our proposed algorithm in Stage II: convergence
# S0, U0: initial value
# hatSigmaX: sample covariance of X
# hatSigmaY: sample covariance of Y
# tSparse, tLowrank: true sparse, true lowrank
# d, rhat, r1hat: sample size, dimension, rank, positive inertia
# betahat, alphahat, shat: tuning parameter
# eta1, eta2: step size
# TrueError: vector to save error
# ObjLoss: objective loss
# Result: class to save result
# maxIter: maximum iteration
# eps: precision error

        
def OurConverg(S0, U0, hatSigmaX, hatSigmaY, tSparse, tLowrank, d, rhat, r1hat, betahat,\
               alphahat, shat, eta1, eta2, TrueError, ObjLoss, Result, maxIter, eps):
    hatDiff = hatSigmaY - hatSigmaX
    # Iteration
    k = 0
    Idend = 1
    Lambda0 = genLambda(rhat, r1hat)
    if rhat > 0:
        Const = np.linalg.norm(U0,2)

    TimeStart = time.time()
    while k <= maxIter:
        # calculate some common terms
        Term1 = S0 + UtoL(U0, r1hat)
        Term2 = np.matmul(np.matmul(hatSigmaX, Term1), hatSigmaY)
        Term3 = np.matmul(np.matmul(hatSigmaY, Term1), hatSigmaX)
        Term4 = np.matmul(U0, Lambda0)
        Term5 = np.matmul(Term2, Term4)
        Term6 = np.matmul(Term3, Term4)
        Term7 = np.matmul(hatDiff, Term4)
        Term8 = np.matmul(U0.T, U0)

        # iteration S
        S12 = S0 - eta1 * (0.5*Term2 + 0.5*Term3 - hatDiff)
        S13 = HardTrun(S12, shat)
        S1 = PropTrun(S13, alphahat)
        # iteration U
        if rhat>0:
            U12 = U0 - eta2 * (Term5 + Term6 - 2 * Term7) - eta2/2 * (np.matmul(U0, Term8) \
                           - np.matmul(np.matmul(Term4, Term8), Lambda0))
            Incobound = 2 * Const * (betahat*rhat/d)**(1/2)
            U1 = ProjInco(U12, Incobound)
        else:
            U1 = U0    
        # calculate error
        succeps = MaxDist(S0, U0, S1, U1, r1hat, rhat)
        L1 = UtoL(U1, r1hat)
        TrueError = CalTrueErr(TrueError, S1, L1, tSparse, tLowrank)

        ObjLoss = LossSampCov(ObjLoss, hatSigmaX, hatSigmaY, S1, U1, L1, r1hat)
        if ObjLoss[-1] > ObjLoss[-2]:  # if loss increase, we need decrease step size
            eta1 = eta1/(k+1)
            eta2 = eta2/(k+1)        
        if succeps <= eps:
            Idend = 0
            break
        
        S0 = S1.copy()
        U0 = U1.copy()
        k = k + 1
    
    TimeEnd = time.time()
    # save results
    Result.CostTime =  TimeEnd - TimeStart
    Result.Idend = Idend
    Result.TrueError = TrueError
    Result.ObjLoss = ObjLoss
    Result.Sparse = S1
    Result.Lowrank = L1
    Result.tSparse = tSparse
    Result.tLowrank = tLowrank
    Result.Iternum = len(TrueError)
    
    return Result




# This function implements our two-stage algorithm

# hatSigmaX, hatSigmaY: sample covariance matrix
# tSparse, tLowrank: true result
# Result: class that saves the result
# rhat, betahat, shatprob, alphahat: tuning parameter
# eta1: step size
# maxIter: maximum iteration
# eps: thresholding error


def OurTwoStageMethod(hatSigmaX, hatSigmaY, tSparse, tLowrank, n, betahat = 1, alphahat = 0.5,\
                       shatprob = 5, rhat = 2, eta1 = 0.1, maxIter = 50000, eps = 1e-5):
    Result = SimuResult()
    d = np.size(hatSigmaX,1)
    shat = int(d*(1 + shatprob))  # sparsity
    # Run Stage-I algorithm
    S0, U0, L0, r1hat = OurInitial(hatSigmaX,hatSigmaY,alphahat,shat,rhat,n)    
    # save parameter we use
    Result.Param = [betahat, alphahat, shatprob, rhat, r1hat]
    # initialize error matrix
    TrueError, ObjLoss = [], []
    TrueError = CalTrueErr(TrueError, S0, L0, tSparse, tLowrank)
    ObjLoss = LossSampCov(ObjLoss, hatSigmaX, hatSigmaY, S0, U0, L0, r1hat)

    # Run Stage-II algorithm
    # specify eta2 such that 0.2<= eta2<= 0.7
    if rhat > 0:
        if 3*eta1 <= np.linalg.norm(U0, 2)**2 and np.linalg.norm(U0, 2)**2 <= 5*eta1:
            eta2 = eta1/np.linalg.norm(U0, 2)**2/10
        else:
            eta2 = eta1/2
    else:
        eta2 = 1
    
    Result = OurConverg(S0, U0, hatSigmaX, hatSigmaY, tSparse, tLowrank, d, rhat, r1hat,\
                        betahat, alphahat, shat, eta1, eta2, TrueError, ObjLoss, Result, \
                        maxIter, eps)
    return Result
 

    
# This function calculates the loss value on test set
# TestX, TestY: dataset from two groups
# Result: class

def Loss(TestX, TestY, Result):
    CovTestX = np.cov(TestX,rowvar = False)
    CovTestY = np.cov(TestY,rowvar = False)
    DiffCov = CovTestY - CovTestX
    DeltaHat = Result.Sparse + Result.Lowrank
    A = np.matmul(DeltaHat, np.matmul(CovTestX, np.matmul(DeltaHat, CovTestY)))
    B = np.matmul(DeltaHat, DiffCov)
    return 0.5 * np.trace(A) - np.trace(B)


   

# This function slelcts the combination of parameters that can minimize the loss on test set
# X, Y: data of two groups
# Betahat: incoherence condition parameter
# Alphahat: sparsity proportion parameter
# Shatprob: sparsity parameter
# Rhat: rank parameter

def ChooseComb(X, Y, Betahat = [1,3], Alphahat = [0.1, 0.3,0.5,0.8],\
               Shatprob = [5,10,30], Rhat = [42, 45, 48]):
    d = np.size(X,1)
    # separate into training set and test set
    TrainX, TestX = SplitData(X)
    TrainY, TestY = SplitData(Y)
    n = np.size(TrainX, 0)
    # calculate covariance matrix for two groups
    hatSigmaX = np.cov(TrainX,rowvar = False)
    hatSigmaY = np.cov(TrainY,rowvar = False)

    # for loop to choose combination
    Param = np.zeros((0, 5))
    for betahat in Betahat:
        for alphahat in Alphahat:
            for shatprob in Shatprob:
                for rhat in Rhat:
                    print('Choose Param', [betahat, alphahat, shatprob, rhat])
                    Result = OurTwoStageMethod(hatSigmaX, hatSigmaY, np.zeros((d,d)),\
                                        np.zeros((d,d)), n, betahat, alphahat, shatprob, rhat)
                    print('Conv', Result.Idend)
                    
                    if Result.Idend == 0:
                        testloss = Loss(TestX, TestY, Result)
                        print('Loss', testloss)
                        Param = np.concatenate((Param, np.array([[betahat, alphahat, shatprob,\
                                                                  rhat, testloss]])), axis=0) 
    
    if len(Param) > 0:
        SelectParam = Param[np.argmin(Param[:,4]), 0:4]
    else:
        SelectParam = np.array([4, 0.15, 0.3, 4])
    
    return SelectParam




# This  function implements our two-stage algorithm
# X, Y: data
# SelectParam: selected parameter

def OurApproach(X,Y,SelectParam):
    n, d = np.size(X,0), np.size(X,1)
    hatSigmaX = np.cov(X,rowvar = False)
    hatSigmaY = np.cov(Y,rowvar = False)
    # calculate true value
    tSparse = np.zeros((d,d))
    tLowrank = np.zeros((d,d))
    # run our algorithm
    Result = OurTwoStageMethod(hatSigmaX,hatSigmaY,tSparse,tLowrank,n,SelectParam[0],\
                                   SelectParam[1],SelectParam[2],int(SelectParam[3]))
    # calculate true rank
    tU, tr, tr1 =  LtoU(tLowrank, 1e-9, 1)
    Result.Rank = [tr, tr1]

    return Result






def MyMain(DataDir = os.getcwd() + '/../data/Groupdata/', OutputDir = os.getcwd() + '/ResultMatrix'):
    # load data
    X = pd.read_csv(DataDir + 'Control.csv', index_col=0).values
    Y = pd.read_csv(DataDir + 'Test.csv', index_col=0).values
    nx = np.size(X,0)
    ny = np.size(Y,0)
    if nx<ny:
        Y = Y[random.sample(range(ny), nx), :]
    else:
        X = X[random.sample(range(nx), ny), :]
    
    # remove file
    filelist = glob.glob(OutputDir + '/*')
    for f in filelist:
        os.remove(f)
    print('Remove result file. Done!')
    # select parameter
    SelectParam = ChooseComb(X, Y)
              
    # Use the selected parameters to calculate the error
    Result = OurApproach(X, Y, SelectParam)

    # save result to txt file
    SS = pd.DataFrame(Result.Sparse)
    print(Result.ObjLoss[-1])
    SS.to_csv(OutputDir + '/mySparse.csv')
            


















