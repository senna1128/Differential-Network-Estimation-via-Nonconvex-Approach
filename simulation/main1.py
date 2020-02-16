#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:29:12 2020

@author: senna
"""

# Import package
import os
#import matplotlib as mpl
#if os.environ.get('DISPLAY','') == '':
#    print('no display found. Using non-interactive Agg backend')
#    mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import pickle
import time
import re


# Modify directory
os.chdir('/Users/senna')

from main2 import *


# remove file in Data folder
filelist = [ f for f in os.listdir(os.getcwd() + '/Data/')]
for f in filelist:
    try:
        os.remove(os.path.join(os.getcwd() + '/Data/', f))
    except:
        subfilelist = [l for l in os.listdir(os.getcwd() + '/Data/' + f)]
        for l in subfilelist:
            os.remove(os.getcwd() + '/Data/' + f + '/' + l)
print('remove data file, Done!')

# remove figures in Figures folder
for f in os.listdir(os.getcwd() + '/Figures/'):
    try:
        os.remove(os.getcwd() + '/Figures/' + f)
    except:
        print('there is a folder')
print('remove figures, Done!')
        
# remove results in Results folder
filelist = [ f for f in os.listdir(os.getcwd() + '/Results/')]
for f in filelist:
    try:
        os.remove(os.path.join(os.getcwd() + '/Results/', f))
    except:
        subfilelist = [l for l in os.listdir(os.getcwd() + '/Results/' + f)]
        for l in subfilelist:
            os.remove(os.getcwd() + '/Results/' + f + '/' + l)
print('remove results file, Done!')




#################################
## Some Specific Case     #######
#################################
random.seed(2019)
np.random.seed(2019)

# generate main dataset given (n, d, r), for running main paper result 
# and random initialization result
GenData(200,50,0,4,1)
GenData(1000,100,1,4,1)
GenData(10000,100,2,4,1)


# generate supplementary dataset given (n,d,r) for running more general precision structure
GenData(5000,30,0,1,3)
GenData(10000,30,1,1,3)
GenData(20000,30,2,1,3)

# generate dataset for showing convergence result only
GenData(150,100,1,6,1)
GenData(150,100,2,6,1)

MyMain()

# plot error decay trend
PlotlossDecay()





#################################
## Run  Ratio     ###############
#################################
random.seed(2019)
np.random.seed(2019)

ratio = np.linspace(0.09,0.4,8)

SequenceRun(ratio, 50, 0)

SequenceRun(ratio, 100, 1)
 
SequenceRun(ratio, 150, 2)



