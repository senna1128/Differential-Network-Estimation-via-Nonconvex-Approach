#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:54:30 2020

@author: senna
"""

import matplotlib.pyplot as plt
import random
import numpy as np
import time
import pickle
import os
import re
import fnmatch

# This function generates covariance matrices for four models in the paper, each model 
# corresponds to one pattern of covariance matrix
# d: dimension
# r: number of latent factor
# Idm: index of generation method
# ParIdm: parameter of Idm method
# Prob: nonzero probability for off-diagonal block matrix
# Drange: parameter for explosion

def CreateData(d,r,Idm,Par1= [0.6, 0.3], Par2 = [0.5], Par3 = [0.8, 0.1], Par4 = [0.5, 0.5],\
               Prob = 0.9, Drange = [0.5, 2.5]):
    if Idm == 1:
        OmegaOO = np.eye(d) + Par1[0]*np.diag(np.ones(d-1),1) + Par1[0]*np.diag(np.ones(d-1),-1)\
                  + Par1[1]*np.diag(np.ones(d-2),2) + Par1[1]*np.diag(np.ones(d-2),-2)
    elif Idm == 2:
        OmegaOO = np.eye(d)
        for k in range(1, d//10+1):
            OmegaOO[10*(k-1),(10*k-7):(10*k)],OmegaOO[(10*k-7):(10*k),10*(k-1)]=Par2[0],Par2[0]
    elif Idm == 3:
        OmegaOO = np.eye(d)
        for i in range(d-4):
            for j in range(i+1,i+4):
                a = Par3[0] * np.random.binomial(1, Par3[1])
                OmegaOO[i, j], OmegaOO[j, i] = a, a        
    elif Idm == 4:
        OmegaOO = np.eye(d)
        for k in range(1, d//2+1):
            a = Par4[0] * np.random.binomial(1, Par4[1])
            OmegaOO[2*(k-1),(2*k-1):min(2*(k+1),d-1)],OmegaOO[(2*k-1):min(2*(k+1),d-1),2*(k-1)]=a,a
                  
    OmegaOH = np.zeros([d,r])
    for i in range(d):
        for j in range(r):
            if np.random.rand(1) < Prob:
                OmegaOH[i, j] = random.uniform(0.5,1)
    OmegaHH = np.eye(r)
    Omega = np.concatenate((np.concatenate((OmegaOO,OmegaOH),axis = 1),\
                            np.concatenate((OmegaOH.T,OmegaHH),axis = 1)), axis = 0)    
    Omega = Omega + (np.abs(min(np.linalg.eig(Omega)[0])) + 1) * np.eye(d+r)
    D = np.diag(np.random.uniform(Drange[0],Drange[1],d+r))
    Omega = np.matmul(D**(1/2), np.matmul(Omega, D**(1/2)))

    Sigma = np.linalg.inv(Omega)
    
    SigmaOO = Sigma[0:d, 0:d]
    OmegaOO = Omega[0:d, 0:d]
    OmegaOH = Omega[0:d, d:]
    OmegaHH = Omega[d:, d:]
    LowrankOO = np.matmul(np.matmul(OmegaOH, np.linalg.inv(OmegaHH)), OmegaOH.T)
    return SigmaOO, OmegaOO, LowrankOO



# This function generates dense positive definite matrices (which is fixed by seed)
# d: dimension
# Par: parameters
# seed: random seed

def GenPosDefMat(d, Par = [0, 1], seed = 2020):
    np.random.seed(seed)
    A = np.random.uniform(Par[0], Par[1], (d,d))
    A = (A + A.T)/2
    A += d * (Par[1] - Par[0]) * np.eye(d)
    return A



# This function generates covariance matrices in supplementary material
# d, r: dimension and number of latent factor
# Idm: index of generation method
# Par: parameters for sparse precision matrix

def CreateSupData(d, r, Idm, Par = [0.6, 0.3]):
    if Idm <= 4:
        SigmaOO, OmegaOO, LowrankOO = CreateData(d, r, Idm)
        Perturb = GenPosDefMat(d)
        OmegaOOnew = OmegaOO + Perturb
        SigmaOOnew = np.linalg.inv(OmegaOOnew - LowrankOO)
        return SigmaOOnew, OmegaOOnew, LowrankOO
    else:
        OmegaOO = np.eye(d) + Par[0]*np.fliplr(np.diag(np.ones(d-1),1)) +  Par[0]*np.fliplr(np.diag(np.ones(d-1),-1)) \
                  + Par[1]*np.fliplr(np.diag(np.ones(d-2),2)) + Par[1]*np.fliplr(np.diag(np.ones(d-2),-2)) + np.fliplr(np.diag(np.ones(d)))
        OmegaOO = OmegaOO + (np.abs(min(np.linalg.eig(OmegaOO)[0])) + 1) * np.eye(d)
        SigmaOO = np.linalg.inv(OmegaOO)
        return SigmaOO, OmegaOO, np.zeros((d,d))
    


# This function generates the data and save to Data folder
# n: sample size
# d: dimension
# r: rank
# IdConver: indicator of plotting convergence or random initialization 
# (only for main paper dataset)
    # IdConver = 1: only show final result without convergence and random initialization
    # IdConver = 2: show both final result and convergence, but not random initialization
    # IdConver = 3: show convergence only, not random initialization
    # IdConver = 4: show final result for both with and without random initialization
    # IdConver = 5: show both final result and convergence for both with and without random initialization
    # IdConver = 6: show only convergence for both with and without random initialization
    # IdConver = 7: only show final result with random initialization
    # IdConver = 8: show both final result and convergence for random initialization
    # IdConver = 9: show convergence for random initialization
# IdRun: indicator of running which result
    # IdRun = 1: run examples in the main paper
    # IdRun = 2: run examples in both main paper and supplementary material
    # IdRun = 3: run examples in the supplementary material

def GenData(n, d, r, IdConver = 5, IdRun = 2):
    # check what to run
    if IdRun <= 2:   # generate data for main paper
        if os.path.exists(os.getcwd()+'/Data/MainNDR.txt'):
            append_write = 'a+' # append if already exists
        else:
            append_write = 'w+' # make a new file if not
        with open(os.getcwd()+'/Data/MainNDR.txt', append_write) as f:
            print('N', n, 'D', d, 'R', r, 'IdConver', IdConver, file = f)
        
        for Idm in range(4):
            tSigmaX, tSparseX, tLowrankX = CreateData(d,r,Idm+1)
            SavePath = os.getcwd() + '/Data/MainPaper/Idm' + str(Idm+1) + 'N' + str(n) + 'D' + str(d) + 'R' + str(r)
            with open(SavePath, "wb") as f:
                pickle.dump(tSigmaX, f)
                pickle.dump(tSparseX, f)
                pickle.dump(tLowrankX, f)
    if IdRun >= 2:  # generate data for supplementary paper
        if os.path.exists(os.getcwd()+'/Data/SuppleNDR.txt'):
            append_write = 'a+' # append if already exists
        else:
            append_write = 'w+' # make a new file if not
        with open(os.getcwd()+'/Data/SuppleNDR.txt', append_write) as f:
            print('N', n, 'D', d, 'R', r, file = f)
        
        for Idm in range(5):
            tSigmaX, tSparseX, tLowrankX = CreateSupData(d,r,Idm+1)
            SavePath = os.getcwd() + '/Data/SuppleMaterial/Idm' + str(Idm+1) + 'N' + str(n) + 'D' + str(d) + 'R' + str(r)
            with open(SavePath, "wb") as f:
                pickle.dump(tSigmaX, f)
                pickle.dump(tSparseX, f)
                pickle.dump(tLowrankX, f)
        for Idm in range(3):
            tSigmaX, tSparseX, tLowrankX = CreateData(d,r,Idm+2)
            SavePath = os.getcwd() + '/Data/SuppleMaterial/SLIdm' + str(Idm+2) + 'N' + str(n) + 'D' + str(d) + 'R' + str(r)
            with open(SavePath, "wb") as f:
                pickle.dump(tSigmaX, f)
                pickle.dump(tSparseX, f)
                pickle.dump(tLowrankX, f)




##############################
#### Run Algorithm  ##########
##############################




# read ndr.txt file and process the data
# IdRun: indicator for runing which results
    # IdRun = 1: run examples in the main paper
    # IdRun = 2: run examples in both main paper and supplementary material
    # IdRun = 3: run examples in the supplementary material
# DataDir: data directory
# OutputDir: result directory

def MyMain(IdRun = 2, DataDir = os.getcwd()+'/Data/', OutputDir = os.getcwd() + '/Results/'):
    ###########################
    ### clear result file  ####
    ###########################
    for f in os.listdir(OutputDir):
        try:
            os.remove(OutputDir + f)
        except:
            subfilelist = [l for l in os.listdir(OutputDir + f)]
            for l in subfilelist:
                os.remove(OutputDir + f + '/' + l)
    print('remove result file, Done!')
    ##############################
    ### Run main paper result ####
    ##############################
    if IdRun <= 2:  # run main paper
        A = []         # read ndr data
        with open(DataDir + 'MainNDR.txt', 'r') as f:
            for line in f:
                match = re.findall(r'-?\d+-?\d*', line)
                if match:
                    A.append(list(map(int, match)))
        TotalCase = np.size(A, 0)
        # loop over all combination of n, d, r
        for i in range(TotalCase):
            n, d, r, IdConver = A[i][0], A[i][1], A[i][2], A[i][3]
            # check what to run for the main paper
            if IdConver >= 7:
                continue
            if IdConver != 3 and IdConver != 6:
                # define output file and write setup of n,d,r
                SaveFile = OutputDir + 'MainPaper/N' + str(n) + 'D' + str(d) + 'R' + str(r) +'.txt'
                with open(SaveFile, 'w+') as f:
                    print('Comb: [{}, {}, {}]'.format(n, d, r), file = f)
            # load covariance matrix and data for control group
            Filename  = DataDir + 'MainPaper/Idm1N' + str(n) + 'D' + str(d) + 'R' + str(r)
            with open(Filename, 'rb') as f:
                tSigmaX = pickle.load(f)
                tSparseX = pickle.load(f)
                tLowrankX = pickle.load(f)
            # consider the difference between control and each test group
            for j in range(3):
                # load covariance matrix for test group
                Filename  = DataDir + 'MainPaper/Idm'+str(j+2)+'N'+str(n)+'D'+str(d)+'R'+str(r)
                with open(Filename, 'rb') as f:
                    tSigmaY = pickle.load(f)
                    tSparseY = pickle.load(f)
                    tLowrankY = pickle.load(f)
                # Calculate true matrices
                tSparse = tSparseX - tSparseY
                tLowrank = tLowrankY - tLowrankX                
                # Calculate true rank and positive of inertia
                tU, tr, tr1 =  LtoU(tLowrank, 1e-9, 1)
                TrueRankIner = [tr, tr1]
                # Given tSigmaX, tSigmaY, n, truerankiner, idrndinitial, we choose tuning parameters
                SelectParam = ChooseComb(tSigmaX, tSigmaY, n, TrueRankIner)
                # Use selected parameters to run our algorithm
                ResSummary = OurApproach(tSigmaX,tSigmaY,tSparse,tLowrank,n,r,SelectParam,TrueRankIner)
                if IdConver != 3 and IdConver != 6:
                    # save result to txt file
                    with open(SaveFile, 'a+') as f:
                        print('Idm: [{}, {}]'.format(1, j+2), file = f)
                        print('RankRatio: {}, IndexRatio: {}'.format(ResSummary.trsum, ResSummary.tr1vec), file = f)
                        print('Time: {}'.format(ResSummary.CostTime), file = f)
                        print('Err: [{}, {}, {}]'.format(ResSummary.SparseError[0], ResSummary.LowrankError[0], ResSummary.TotalError[0]), file = f)
                        print('Std: [{}, {}, {}]'.format(ResSummary.SparseError[1], ResSummary.LowrankError[1], ResSummary.TotalError[1]), file = f)
                if IdConver != 1 and IdConver != 4:
                    # save error decay to file
                    IterErr = [ResSummary.SparseIterError, ResSummary.LowrankIterError, ResSummary.TotalIterError]
                    if IdConver <= 3:
                        IterPath = OutputDir + 'SuppleMaterial/MainIterN' + str(n) + 'D' + str(d) + 'R' + str(r)
                    else:
                        IterPath = OutputDir + 'SuppleMaterial/IterN' + str(n) + 'D' + str(d) + 'R' + str(r)
                    with open(IterPath, "ab") as f:
                        pickle.dump(IterErr, f)
    ###################################
    ### Run supplementary material ####
    ###################################
    if IdRun >= 2:  # run supplementary material
        ########################################################################  
        ##### Study Random Initialization (Use main paper data)   ##############
        ########################################################################
        A = []         # read ndr data
        with open(DataDir + 'MainNDR.txt', 'r') as f:
            for line in f:
                match = re.findall(r'-?\d+-?\d*', line)
                if match:
                    A.append(list(map(int, match)))
        TotalCase = np.size(A, 0)
        # loop over all combination of n, d, r
        for i in range(TotalCase):
            n, d, r, IdConver = A[i][0], A[i][1], A[i][2], A[i][3]
            # check what to run for the main paper
            if IdConver <= 3:
                continue
            if IdConver != 6 and IdConver != 9:
                # define output file and write setup of n,d,r
                SaveFile = OutputDir + 'SuppleMaterial/RandomN' + str(n) + 'D' + str(d) + 'R' + str(r) +'.txt'
                with open(SaveFile, 'w+') as f:
                    print('Comb: [{}, {}, {}]'.format(n, d, r), file = f)
            # load covariance matrix and data for control group
            Filename  = DataDir + 'MainPaper/Idm1N' + str(n) + 'D' + str(d) + 'R' + str(r)
            with open(Filename, 'rb') as f:
                tSigmaX = pickle.load(f)
                tSparseX = pickle.load(f)
                tLowrankX = pickle.load(f)
            # consider the difference between control and each test group
            for j in range(3):
                # load covariance matrix for test group
                Filename  = DataDir + 'MainPaper/Idm'+str(j+2)+'N'+str(n)+'D'+str(d)+'R'+str(r)
                with open(Filename, 'rb') as f:
                    tSigmaY = pickle.load(f)
                    tSparseY = pickle.load(f)
                    tLowrankY = pickle.load(f)
                # Calculate true matrices
                tSparse = tSparseX - tSparseY
                tLowrank = tLowrankY - tLowrankX                
                # Calculate true rank and positive of inertia
                tU, tr, tr1 =  LtoU(tLowrank, 1e-9, 1)
                TrueRankIner = [tr, tr1]
                # Given tSigmaX, tSigmaY, n, truerankiner, idrndinitial, we choose tuning parameters
                SelectParam = ChooseComb(tSigmaX, tSigmaY, n, TrueRankIner, 1)
                # Use selected parameters to run our algorithm
                ResSummary = OurApproach(tSigmaX,tSigmaY,tSparse,tLowrank,n,r,SelectParam,TrueRankIner,1)
                if IdConver != 6 and IdConver != 9:
                    # save result to txt file
                    with open(SaveFile, 'a+') as f:
                        print('Idm: [{}, {}]'.format(1, j+2), file = f)
                        print('RankRatio: {}, IndexRatio: {}'.format(ResSummary.trsum, ResSummary.tr1vec), file = f)
                        print('Time: {}'.format(ResSummary.CostTime), file = f)
                        print('Err: [{}, {}, {}]'.format(ResSummary.SparseError[0], ResSummary.LowrankError[0], ResSummary.TotalError[0]), file = f)
                        print('Std: [{}, {}, {}]'.format(ResSummary.SparseError[1], ResSummary.LowrankError[1], ResSummary.TotalError[1]), file = f)
                if IdConver != 4 and IdConver != 7:
                    # save error decay to file
                    IterErr = [ResSummary.SparseIterError, ResSummary.LowrankIterError, ResSummary.TotalIterError]
                    if IdConver >= 7:
                        IterPath = OutputDir + 'SuppleMaterial/SuppleIterN' + str(n) + 'D' + str(d) + 'R' + str(r)
                    else:
                        IterPath = OutputDir + 'SuppleMaterial/IterN' + str(n) + 'D' + str(d) + 'R' + str(r)
                    with open(IterPath, "ab") as f:
                        pickle.dump(IterErr, f)
        
        ############################################################    
        ##### Study more general precision matrices    #############
        ############################################################
        if not os.path.exists(DataDir + 'SuppleNDR.txt'):
            return
        A = []         # read ndr data
        with open(DataDir + 'SuppleNDR.txt', 'r') as f:
            for line in f:
                match = re.findall(r'-?\d+-?\d*', line)
                if match:
                    A.append(list(map(int, match)))
        TotalCase = np.size(A, 0)        
        # loop over all combination of n, d, r
        for i in range(TotalCase):
            n, d, r = A[i][0], A[i][1], A[i][2]
            # define output file and write setup of n,d,r
            SaveFile = OutputDir + 'SuppleMaterial/N' + str(n) + 'D' + str(d) + 'R' + str(r) +'.txt'
            with open(SaveFile, 'w+') as f:
                print('Comb: [{}, {}, {}]'.format(n, d, r), file = f)
            ###################
            ### Case 1: plus the same positive definite matrix
            # load covariance matrix and data for control group
            Filename  = DataDir + 'SuppleMaterial/Idm1N' + str(n) + 'D' + str(d) + 'R' + str(r)
            with open(Filename, 'rb') as f:
                tSigmaX = pickle.load(f)
                tSparseX = pickle.load(f)
                tLowrankX = pickle.load(f)            
            # consider the difference between control and each test group
            for j in range(3):
                # load covariance matrix for test group
                Filename  = DataDir + 'SuppleMaterial/Idm'+str(j+2)+'N'+str(n)+'D'+str(d)+'R'+str(r)
                with open(Filename, 'rb') as f:
                    tSigmaY = pickle.load(f)
                    tSparseY = pickle.load(f)
                    tLowrankY = pickle.load(f)
                # Calculate true matrices
                tSparse = tSparseX - tSparseY
                tLowrank = tLowrankY - tLowrankX
                # Calculate true rank and positive of inertia
                tU, tr, tr1 =  LtoU(tLowrank, 1e-9, 1)
                TrueRankIner = [tr, tr1]
                # Given tSigmaX, tSigmaY, n, truerankiner, idrndinitial, we choose tuning parameters
                SelectParam = ChooseComb(tSigmaX, tSigmaY, n, TrueRankIner)
                # Use selected parameters to run our algorithm
                ResSummary = OurApproach(tSigmaX,tSigmaY,tSparse,tLowrank,n,r,SelectParam,TrueRankIner)
                # save result to txt file
                with open(SaveFile, 'a+') as f:
                    print('Idm: [{}, {}]'.format(1, j+2), file = f)
                    print('RankRatio: {}, IndexRatio: {}'.format(ResSummary.trsum, ResSummary.tr1vec), file = f)
                    print('Time: {}'.format(ResSummary.CostTime), file = f)
                    print('Err: [{}, {}, {}]'.format(ResSummary.SparseError[0], ResSummary.LowrankError[0], ResSummary.TotalError[0]), file = f)
                    print('Std: [{}, {}, {}]'.format(ResSummary.SparseError[1], ResSummary.LowrankError[1], ResSummary.TotalError[1]), file = f)
            ###################
            ### Case 2: sparse minus sparse plus low-rank
            # load covariance matrix and data for control group
            Filename  = DataDir + 'SuppleMaterial/Idm5N' + str(n) + 'D' + str(d) + 'R' + str(r)
            with open(Filename, 'rb') as f:
                tSigmaX = pickle.load(f)
                tSparseX = pickle.load(f)
                tLowrankX = pickle.load(f)            
            # consider the difference between control and each test group
            for j in range(3):
                # load covariance matrix for test group
                Filename  = DataDir + 'SuppleMaterial/SLIdm'+str(j+2)+'N'+str(n)+'D'+str(d)+'R'+str(r)
                with open(Filename, 'rb') as f:
                    tSigmaY = pickle.load(f)
                    tSparseY = pickle.load(f)
                    tLowrankY = pickle.load(f) 
                # Calculate true matrices
                tSparse = tSparseX - tSparseY
                tLowrank = tLowrankY - tLowrankX
                # Calculate true rank and positive of inertia
                tU, tr, tr1 =  LtoU(tLowrank, 1e-9, 1)
                TrueRankIner = [tr, tr1]
                # Given tSigmaX, tSigmaY, n, truerankiner, idrndinitial, we choose tuning parameters
                SelectParam = ChooseComb(tSigmaX, tSigmaY, n, TrueRankIner)
                # Use selected parameters to run our algorithm
                ResSummary = OurApproach(tSigmaX,tSigmaY,tSparse,tLowrank,n,r,SelectParam,TrueRankIner)
                # save result to txt file
                with open(SaveFile, 'a+') as f:
                    print('Idm: [{}, {}]'.format(5, j+2), file = f)
                    print('RankRatio: {}, IndexRatio: {}'.format(ResSummary.trsum, ResSummary.tr1vec), file = f)
                    print('Time: {}'.format(ResSummary.CostTime), file = f)
                    print('Err: [{}, {}, {}]'.format(ResSummary.SparseError[0], ResSummary.LowrankError[0], ResSummary.TotalError[0]), file = f)
                    print('Std: [{}, {}, {}]'.format(ResSummary.SparseError[1], ResSummary.LowrankError[1], ResSummary.TotalError[1]), file = f)
                    

























def PlotlossDecay(ResDir = os.getcwd()+'/Results/SuppleMaterial', OutFigDir = os.getcwd() + '/Figures'):
    #######################################################################
    ###### Plot Convergence for with and without random initialization ####
    #######################################################################
    A = []
    for file in os.listdir(ResDir):
        if fnmatch.fnmatch(file, 'Iter*'):
            B = file.split('IterN')[-1]
            A.append(list(map(int, [B.split('D')[0]] + B.split('D')[-1].split('R'))))
    TotalCase = len(A)
    for i in range(TotalCase):
        n, d, r = A[i][0], A[i][1], A[i][2]
        # load iteration file
        Filename  = ResDir + '/IterN' + str(n) + 'D' + str(d) + 'R' + str(r)
        with open(Filename, 'rb') as f:
            Err12 = pickle.load(f)
            Err13 = pickle.load(f)
            Err14 = pickle.load(f)
            RndInitialErr12  = pickle.load(f)
            RndInitialErr13  = pickle.load(f)
            RndInitialErr14  = pickle.load(f)
        
        saveFigpath = OutFigDir+ '/BothN' + str(n) + 'D'+str(d) + 'R' + str(r) + '.png'
        f = plt.figure(figsize=(10,3))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        ax1.plot(range(len(Err12[2])), 2*np.log(Err12[2]), color='blue')
        ax1.plot(range(len(Err13[2])), 2*np.log(Err13[2]), color='green')
        ax1.plot(range(len(Err14[2])), 2*np.log(Err14[2]), color='red')
        ax1.set_xlabel('iteration k',fontsize=16)
        ax1.set_ylabel(r'$2\log(\frac{\|\|\hat{\Delta} - \Delta^*\|\|_F}{\sqrt{\sigma_{\max}(R^*)}})$', fontsize=16)
        LL = np.min([len(RndInitialErr12[2]), len(RndInitialErr13[2]), len(RndInitialErr14[2]),300])
        ax2.plot(range(LL), 2*np.log(RndInitialErr12[2][0:LL]), color='blue')
        ax2.plot(range(LL), 2*np.log(RndInitialErr13[2][0:LL]), color='green')
        ax2.plot(range(LL), 2*np.log(RndInitialErr14[2][0:LL]), color='red')
        ax2.set_xlabel('iteration k',fontsize=16)        
        f.savefig(saveFigpath,dpi=300,bbox_inches="tight")
        
        with open(Filename + '.txt', 'w+') as f:
            print('Comb: [{}, {}, {}]'.format(n, d, r), file = f)
            print('Idm: [{}, {}]'.format(1, 2), file = f)
            print('RegularErr: [{}, {}, {}]'.format(Err12[0][-1], Err12[1][-1], Err12[2][-1]), file = f)
            print('RndInitialErr: [{}, {}, {}]'.format(RndInitialErr12[0][-1], RndInitialErr12[1][-1], RndInitialErr12[2][-1]), file = f)
            print('Idm: [{}, {}]'.format(1, 3), file = f)
            print('RegularErr: [{}, {}, {}]'.format(Err13[0][-1], Err13[1][-1], Err13[2][-1]), file = f)
            print('RndInitialErr: [{}, {}, {}]'.format(RndInitialErr13[0][-1], RndInitialErr13[1][-1], RndInitialErr13[2][-1]), file = f)
            print('Idm: [{}, {}]'.format(1, 4), file = f)
            print('RegularErr: [{}, {}, {}]'.format(Err14[0][-1], Err14[1][-1], Err14[2][-1]), file = f)
            print('RndInitialErr: [{}, {}, {}]'.format(RndInitialErr14[0][-1], RndInitialErr14[1][-1], RndInitialErr14[2][-1]), file = f)

    #######################################################################
    ###### Plot Convergence for without random initialization #############
    #######################################################################
    A = []
    for file in os.listdir(ResDir):
        if fnmatch.fnmatch(file, 'MainIter*'):
            B = file.split('MainIterN')[-1]
            A.append(list(map(int, [B.split('D')[0]] + B.split('D')[-1].split('R'))))
    TotalCase = len(A)
    for i in range(TotalCase):
        n, d, r = A[i][0], A[i][1], A[i][2]
        # load iteration file
        Filename  = ResDir + '/MainIterN' + str(n) + 'D' + str(d) + 'R' + str(r)
        with open(Filename, 'rb') as f:
            Err12 = pickle.load(f)
            Err13 = pickle.load(f)
            Err14 = pickle.load(f)
        
        saveFigpath = OutFigDir+ '/MainN' + str(n) + 'D'+str(d) + 'R' + str(r) + '.png'
        f = plt.figure(figsize=(5,3))
        ax1 = f.add_subplot(111)
        ax1.plot(range(len(Err12[2])), 2*np.log(Err12[2]), color='blue')
        ax1.plot(range(len(Err13[2])), 2*np.log(Err13[2]), color='green')
        ax1.plot(range(len(Err14[2])), 2*np.log(Err14[2]), color='red')
        ax1.set_xlabel('iteration k',fontsize=16)
        ax1.set_ylabel(r'$2\log(\frac{\|\|\hat{\Delta} - \Delta^*\|\|_F}{\sqrt{\sigma_{\max}(R^*)}})$', fontsize=16)
        f.savefig(saveFigpath,dpi=300,bbox_inches="tight")
    #######################################################################
    ###### Plot Convergence for with random initialization ################
    #######################################################################
    A = []
    for file in os.listdir(ResDir):
        if fnmatch.fnmatch(file, 'SuppleIter*'):
            B = file.split('SuppleIterN')[-1]
            A.append(list(map(int, [B.split('D')[0]] + B.split('D')[-1].split('R'))))
    TotalCase = len(A)
    for i in range(TotalCase):
        n, d, r = A[i][0], A[i][1], A[i][2]
        # load iteration file
        Filename  = ResDir + '/SuppleIterN' + str(n) + 'D' + str(d) + 'R' + str(r)
        with open(Filename, 'rb') as f:
            RndInitialErr12 = pickle.load(f)
            RndInitialErr13 = pickle.load(f)
            RndInitialErr14 = pickle.load(f)
        
        saveFigpath = OutFigDir+ '/SuppleN' + str(n) + 'D'+str(d) + 'R' + str(r) + '.png'
        f = plt.figure(figsize=(5,3))
        ax1 = f.add_subplot(111)
        ax1.plot(range(len(RndInitialErr12[2])), 2*np.log(RndInitialErr12[2]), color='blue')
        ax1.plot(range(len(RndInitialErr13[2])), 2*np.log(RndInitialErr13[2]), color='green')
        ax1.plot(range(len(RndInitialErr14[2])), 2*np.log(RndInitialErr14[2]), color='red')
        ax1.set_xlabel('iteration k',fontsize=16)
        ax1.set_ylabel(r'$2\log(\frac{\|\|\hat{\Delta} - \Delta^*\|\|_F}{\sqrt{\sigma_{\max}(R^*)}})$', fontsize=16)
        f.savefig(saveFigpath,dpi=300,bbox_inches="tight")










# This function outputs U by given L, L = ULambdaU^T
# L: low-rank component
# rhatThres: either rhat or threshold
# Id: Id = 0: given rhat
#     Id = 1: given thres

def LtoU(L, rhatThres, Id):
    EigVal, EigVec = np.linalg.eig(L)
    if Id == 0:
        rhatThres = min(rhatThres, np.size(L,0))
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
# TrueRankIner: true rank and inertia
# IdRndInitial: indicator of random initialization
# alphahat, shat, rhat: tuning parameter
# n, d: sample size, dimension
# seed: random seed
# RandParm: parameter for normal initialization. (mean and std)

def OurInitial(hatSigmaX,hatSigmaY,TrueRankIner,IdRndInitial,alphahat,shat,rhat,n,d,\
               seed=2020,RandParm= [0,3]):
    # Random initialization
    if IdRndInitial:
#        np.random.seed(seed)
        S0 = np.zeros((d,d))
        U0 = np.random.normal(RandParm[0],RandParm[1],(d,TrueRankIner[0]))
        L0 = UtoL(U0, TrueRankIner[1])
        return S0, U0, L0, TrueRankIner[1]
    
    # Otherwise we use our stage-I algorithm
    TildeSigmaX = (n-1)/(n-d-2)*hatSigmaX
    TildeSigmaY = (n-1)/(n-d-2)*hatSigmaY
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





# This  function implements our two-stage algorithm
# tSigmaX: data for control group
# tSigmaY: data for test group
# tSparse, tLowrank: true matrices
# n, r: sample size, rank
# SelectParam: selected parameter
# TrueRankIner: true rank and inertial
# IdRndInitial: indicator of random initialization
# Idrep: replicate

def OurApproach(tSigmaX,tSigmaY,tSparse,tLowrank,n,r,SelectParam,TrueRankIner,IdRndInitial = 0,\
                Idrep = 40):
    FinalResult = []
    d = np.size(tSigmaX,0)
    # Run replicates
    for idrep in range(Idrep):
        print('Replicate',[idrep], [n,d,r], SelectParam)

        # for each replicate, we generate data
        X = np.random.multivariate_normal(np.zeros(d), tSigmaX, n)
        Y = np.random.multivariate_normal(np.zeros(d), tSigmaY, n)
        hatSigmaX = np.cov(X,rowvar = False)
        hatSigmaY = np.cov(Y,rowvar = False)
        
        # run our algorithm
        Result = OurTwoStageMethod(hatSigmaX,hatSigmaY,tSparse,tLowrank,n,TrueRankIner,IdRndInitial,\
                                   SelectParam[0],SelectParam[1],SelectParam[2],int(SelectParam[3]))        
        Result.Gencomb = [n, d, r]
        if Result.Idend == 0:
            FinalResult.append(Result)

    SumRes = SummarySimuResult()
    SumRes.Gencomb = [n, d, r]
    SparseErr, LowrankErr, TotalErr = [], [], []
    for result in FinalResult:
        SumRes.Iternumvec.append(result.Iternum)
        SumRes.CostTime.append(result.CostTime)
        ttr1 = (result.Rank[1] == int(result.Param[4]))
        SumRes.tr1vec.append(ttr1)
        ttr = (result.Rank[0] == int(result.Param[3]))
        SumRes.trsum.append(ttr)
        SparseErr.append(result.TrueError[-1][0])  # sparse true error
        LowrankErr.append(result.TrueError[-1][1])  # lowrank true error
        TotalErr.append(result.TrueError[-1][2])  # total true error
    
    SumRes.SparseError = [np.mean(SparseErr), np.std(SparseErr)]
    SumRes.LowrankError = [np.mean(LowrankErr), np.std(LowrankErr)]
    SumRes.TotalError = [np.mean(TotalErr), np.std(TotalErr)]
    SumRes.trsum = np.mean(SumRes.trsum)
    SumRes.tr1vec = np.mean(SumRes.tr1vec)
    SumRes.CostTime = np.mean(SumRes.CostTime)
    
    # summarize iteration error
    Idshortest = np.argmin(SumRes.Iternumvec) # choose the shortest iteration
    SelectRes = FinalResult[Idshortest]

    SumRes.Sparse = SelectRes.Sparse
    SumRes.Lowrank = SelectRes.Lowrank
    SumRes.tSparse = SelectRes.tSparse
    SumRes.tLowrank = SelectRes.tLowrank
    
    AllErr = SelectRes.TrueError
    SumRes.SparseIterError = [II[0] for II in AllErr]
    SumRes.LowrankIterError = [II[1] for II in AllErr]
    SumRes.TotalIterError = [II[2] for II in AllErr]
        
    return SumRes





class SummarySimuResult:
    def __init__(self):
        self.Iternumvec = []      # Iteration number
        self.CostTime = []        # cost time
        self.tr1vec = []          # correctly recover positive inertia
        self.trsum = []            # correctly select rank
        self.SparseError = []     # Final error for sparse with standard deviation
        self.LowrankError = []    # Final error for lowrank with standard deviation
        self.TotalError = []      # Final total error with standard deviation
        self.SparseIterError = [] # Iteration error (with shortest length)
        self.LowrankIterError = []# Iteration error 
        self.TotalIterError = []  # Iteration error
        self.Sparse = np.zeros(0) # output: sparse component
        self.Lowrank = np.zeros(0)# output: lowrank component
        self.tSparse = np.zeros(0)  # true sparse component
        self.tLowrank = np.zeros(0) # true lowrank component
        self.Gencomb = [0,0,0]    # combination of n, d, r,



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










# This function implements our two-stage algorithm
# hatSigmaX, hatSigmaY: sample covariance matrix
# tSparse, tLowrank: true result
# n: sample size
# TrueRankIner: true rank and inertia
# IdRndInitial: indicator of random initialization
# betahat, alphahat, shatprob, rhat: tuning parameter
# eta1: step size
# maxIter: maximum iteration
# eps: thresholding error

def OurTwoStageMethod(hatSigmaX,hatSigmaY,tSparse,tLowrank,n,TrueRankIner,IdRndInitial,\
                      betahat = 1,alphahat = 0.5,shatprob = 5, rhat = 2, eta1 = 0.5,\
                      maxIter = 5000, eps = 1e-5):
    Result = SimuResult()
    Result.Rank = TrueRankIner
    d = np.size(hatSigmaX,0)
    shat = int(d*(1+shatprob))  # sparsity
    # Run Stage-I algorithm
    S0, U0, L0, r1hat = OurInitial(hatSigmaX,hatSigmaY,TrueRankIner,IdRndInitial,alphahat,\
                                   shat,rhat,n,d)    
    # save parameter we use
    if IdRndInitial:
        Result.Param = [betahat, alphahat, shatprob, TrueRankIner[0], TrueRankIner[1]]
    else:
        Result.Param = [betahat, alphahat, shatprob, rhat, r1hat]
    # initialize error matrix
    TrueError, ObjLoss = [], []
    TrueError = CalTrueErr(TrueError, S0, L0, tSparse, tLowrank)
    ObjLoss = LossSampCov(ObjLoss, hatSigmaX, hatSigmaY, S0, U0, L0, r1hat)

    # Run Stage-II algorithm
    # specify eta2 such that 0.2<= eta2<= 0.7
    if (rhat > 0 and not IdRndInitial) or (IdRndInitial and TrueRankIner[0]):
        if 3*eta1 <= np.linalg.norm(U0, 2)**2 and np.linalg.norm(U0, 2)**2 <= 5*eta1:
            eta2 = eta1/np.linalg.norm(U0, 2)**2
        else:
            eta2 = eta1/2
    else:
        eta2 = 1
    
    Result = OurConverg(S0, U0, L0, hatSigmaX, hatSigmaY, tSparse, tLowrank, d, r1hat,\
                        betahat, alphahat, shat, eta1, eta2, TrueError, ObjLoss, Result,\
                        maxIter, eps)

    return Result




# This function implements our proposed algorithm in Stage II: convergence
# S0, U0, L0: initial value
# hatSigmaX: sample covariance of X
# hatSigmaY: sample covariance of Y
# tSparse, tLowrank: true sparse, true lowrank
# d, r1hat: sample size, dimension, positive inertia
# betahat, alphahat, shat: tuning parameter
# eta1, eta2: step size
# TrueError: vector to save error
# ObjLoss: objective loss
# Result: class to save result
# maxIter: maximum iteration
# eps: precision error

def OurConverg(S0, U0, L0, hatSigmaX, hatSigmaY, tSparse, tLowrank, d, r1hat, betahat,\
               alphahat, shat, eta1, eta2, TrueError, ObjLoss, Result, maxIter, eps):
    # Compute the difference and rhat
    hatDiff = hatSigmaY - hatSigmaX
    rhat = np.size(U0, 1)
    # Iteration
    k = 0
    Idend = 1
    Lambda0 = genLambda(rhat, r1hat)
    if rhat > 0:
        Const = np.linalg.norm(U0,2)

    TimeStart = time.time()
    while k <= maxIter:
        # calculate some common terms
        Term1 = S0 + L0
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
        L0 = L1.copy()
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






# This function generates Lambda by given r1 and r
# r: rank
# r1: positive index of inertia (r1<r0)

def genLambda(r, r1):
    Lambda = np.concatenate((np.concatenate((np.eye(r1),np.zeros((r1,r-r1))), axis = 1),\
                    np.concatenate((np.zeros((r-r1,r1)),-1*np.eye(r-r1)), axis = 1)),axis = 0)
    return Lambda


# This function calculates the lowrank matrix by given U and r1
# U: low-rank factor
# r1: positive index of inertia

def UtoL(U, r1):
    r = np.size(U, 1)
    Lambda = genLambda(r, r1)
    L = np.matmul(U, np.matmul(Lambda, U.T))
    return L







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





# This function runs a sequence of examples by given ratio, d, r
# ratio: rage of sqrt{dlog d/n}
# d, r: dimension, rank
# OutFigDir: output figure directory

def SequenceRun(ratio, d, r, OutFigDir = os.getcwd() + '/Figures'):    
    # try three models
    Idm = [[1, 2], [1, 3], [1, 4]]
    # get sample size
    NN = np.array(list(map(int,np.floor(d*np.log(d)/ratio**2))))
    SparseErr, SparseStd = [[] for i in range(3)], [[] for i in range(3)]
    LowrankErr, LowrankStd = [[] for i in range(3)], [[] for i in range(3)]
    TrueR, TrueR1 = [[] for i in range(3)], [[] for i in range(3)]

    for n in NN:
        SaveRes = []
        for i in range(3):
            # create control group
            tSigmaX, tSparseX, tLowrankX = CreateData(d,r,Idm[i][0])
            # create test group
            tSigmaY, tSparseY, tLowrankY = CreateData(d,r,Idm[i][1])
            # Calculate true matrices
            tSparse = tSparseX - tSparseY
            tLowrank = tLowrankY - tLowrankX
            # Calculate true rank and positive of inertia
            tU, tr, tr1 =  LtoU(tLowrank, 1e-9, 1)
            TrueRankIner = [tr, tr1]
            # Given tSigmaX, tSigmaY, n, truerankiner, idrndinitial, we choose tuning parameters
            SelectParam = ChooseComb(tSigmaX, tSigmaY, n, TrueRankIner)
            # Use selected parameters to run our algorithm
            ResSummary = OurApproach(tSigmaX,tSigmaY,tSparse,tLowrank,n,r,SelectParam,TrueRankIner,0, 40)
            SaveRes.append(ResSummary)
        for i in range(3):
            SparseErr[i].append(SaveRes[i].SparseError[0])
            SparseStd[i].append(SaveRes[i].SparseError[1])
            LowrankErr[i].append(SaveRes[i].LowrankError[0])
            LowrankStd[i].append(SaveRes[i].LowrankError[1])
            TrueR[i].append(SaveRes[i].trsum)
            TrueR1[i].append(SaveRes[i].tr1vec)
    
    # Plot
    saveFigpath = OutFigDir + '/Seq' + 'D' + str(d) + 'R' + str(r)
    # plot sparse
    Sseq = (d*np.log(d)/NN)**(1/2)
    SFigpath = saveFigpath + 'S.png'
    fig1, ax1 = plt.subplots()
    ax1.errorbar(Sseq, SparseErr[0], yerr = SparseStd[0], fmt='-o', color='blue',capsize = 7)
    ax1.errorbar(Sseq, SparseErr[1], yerr = SparseStd[1], fmt='-o', color='green',capsize = 7)
    ax1.errorbar(Sseq, SparseErr[2], yerr = SparseStd[2], fmt='-o', color='red',capsize = 7)
    plt.xlabel(r'$\sqrt{\frac{d\log d}{n}}$',fontsize=16)
    plt.ylabel(r'$\|\|\hat{S} - S^*\|\|_F$', fontsize=16)
    fig1.savefig(SFigpath,dpi=300,bbox_inches="tight")
    '''
    # plot rank recovery
    RFigpath = saveFigpath + 'R.png'
    fig2, ax2 = plt.subplots()
    ax2.plot(Sseq, TrueR[0], '-o', color='blue')
    ax2.plot(Sseq, TrueR[1], '-o', color='green')
    ax2.plot(Sseq, TrueR[2], '-o', color='red')
    plt.xlabel(r'$\sqrt{\frac{d\log d}{n}}$',fontsize=16)
    plt.ylabel(r'true recovery', fontsize=16)
    fig2.savefig(RFigpath,dpi=300,bbox_inches="tight")
    
    # plot positive recovery
    R1Figpath = saveFigpath + 'R1.png'
    fig3, ax3 = plt.subplots()
    ax3.plot(Sseq, TrueR1[0], '-o', color='blue')
    ax3.plot(Sseq, TrueR1[1], '-o', color='green')
    ax3.plot(Sseq, TrueR1[2], '-o', color='red')
    plt.xlabel(r'$\sqrt{\frac{d\log d}{n}}$',fontsize=16)
    plt.ylabel(r'true discovery rate', fontsize=16)
    fig3.savefig(R1Figpath,dpi=300,bbox_inches="tight")
    ''' and None
    # plot lowrank
    if r > 0:
        Lseq = (r/np.log(d))**(1/2)*Sseq
        LFigpath = saveFigpath + 'L.png'
        fig4, ax4 = plt.subplots()
        ax4.errorbar(Lseq, LowrankErr[0], yerr = LowrankStd[0], fmt='-o', color='blue',capsize = 7)
        ax4.errorbar(Lseq, LowrankErr[1], yerr = LowrankStd[1], fmt='-o', color='green',capsize = 7)
        ax4.errorbar(Lseq, LowrankErr[2], yerr = LowrankStd[2], fmt='-o', color='red',capsize = 7)
        plt.xlabel(r'$\sqrt{\frac{rd}{n}}$',fontsize=16)
        plt.ylabel(r'$\|\|\hat{R} - R^*\|\|_F$', fontsize=16)
        fig4.savefig(LFigpath,dpi=300,bbox_inches="tight")
  

# This function slelcts the combination of parameters that can minimize the loss on test set
# The loss function is defined in (5) in the paper 
# tSigmaX, tSigmaY: true covariance matrix
# n: sample size
# TrueRankInte: true rank and positive index of inertia
# IdRndInitial: inidicator of random initialization
    # IdRndInitial = 0: not random initialization
    # IdRndInitial = 1: use random initialization
# Betahat: incoherence condition parameter
# Alphahat: sparsity proportion parameter
# Shatprob: sparsity parameter
# Rhat: rank parameter

def ChooseComb(tSigmaX,tSigmaY,n,TrueRankIner,IdRndInitial = 0, Betahat = [1,3], \
               Alphahat=[0.01,0.03,0.05,0.5],Shatprob=[1,3,5],Rhat=[0,1,2,3,4]):
    # get dimension
    d = np.size(tSigmaX,0)
    # generate dataset
    X = np.random.multivariate_normal(np.zeros(d), tSigmaX, 2*n)
    Y = np.random.multivariate_normal(np.zeros(d), tSigmaY, 2*n)
    # separate into training set and test set
    TrainX, TestX = SplitData(X)
    TrainY, TestY = SplitData(Y)
    # calculate covariance matrix for two groups
    hatSigmaX = np.cov(TrainX,rowvar = False)
    hatSigmaY = np.cov(TrainY,rowvar = False)
    # check if one should run random initialization
    if IdRndInitial == 0:
        # for loop to choose combination
        Param = np.zeros((0, 5))
        for betahat in Betahat:
            for alphahat in Alphahat:
                for shatprob in Shatprob:
                    for rhat in Rhat:
                        print('Main Paper Choose Param', [n, d], [betahat, alphahat, shatprob, rhat])
                        Result = OurTwoStageMethod(hatSigmaX,hatSigmaY,np.zeros((d,d)),\
                                                   np.zeros((d,d)),n,TrueRankIner,IdRndInitial,\
                                                   betahat,alphahat,shatprob,rhat)
                        if Result.Idend == 0:
                            testloss = Loss(TestX, TestY, Result)
                            Param = np.concatenate((Param, np.array([[betahat, alphahat, shatprob,\
                                                                      rhat, testloss]])), axis=0)
        if len(Param) > 0:
            SelectParam = Param[np.argmin(Param[:,4]), 0:4]
        else:
            SelectParam = np.array([4, 0.15, 0.3, 4])
        return SelectParam
    else:
        # for loop to choose combination
        Param = np.zeros((0, 5))
        for betahat in Betahat:
            for alphahat in Alphahat:
                for shatprob in Shatprob:
                    print('Supple Choose Param', [n, d], [betahat, alphahat, shatprob, TrueRankIner[0]])
                    Result = OurTwoStageMethod(hatSigmaX,hatSigmaY,np.zeros((d,d)),np.zeros((d,d)),\
                                               n,TrueRankIner,IdRndInitial,betahat,alphahat,\
                                               shatprob)
                    if Result.Idend == 0:
                        testloss = Loss(TestX, TestY, Result)
                        Param = np.concatenate((Param, np.array([[betahat, alphahat, shatprob,\
                                                                      TrueRankIner[0], testloss]])), axis=0)
        if len(Param) > 0:
            SelectParam = Param[np.argmin(Param[:,4]), 0:4]
        else:
            SelectParam = np.array([4, 0.15, 0.3, TrueRankIner[0]])
        return SelectParam

