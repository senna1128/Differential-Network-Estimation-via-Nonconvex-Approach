#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Import function
os.chdir('/Users/senna/course/Mladen/pro3/simulation/local/mymethod')

from main2 import *

random.seed(2019)
np.random.seed(2019)

#################################
## Some Specific Case     #######
#################################

### Generate Data and save to Data folder

# remove result file in Data folder
filelist = [ f for f in os.listdir(os.getcwd() + '/Data/')]
for f in filelist:
    os.remove(os.path.join(os.getcwd() + '/Data/', f))
print('remove result file, Done!')

# remove ndr.txt
try:
    os.remove(os.getcwd() + '/ndr.txt')
    print('remove ndr.txt file, Done!')
except:
    print('no need to delete ndr file')

# Generate three datasets
# n, d, r
GenData(200, 50, 0)

GenData(1000, 100, 1)

GenData(10000, 100, 2)


# run some specific algorithm
MyMain()





########################################
## Sequence of n by given d, r    ######
##############################3#########

ratio = np.linspace(0.09,0.4,8)

SequenceRun(ratio, 50, 0)

SequenceRun(ratio, 50, 1)
 
SequenceRun(ratio, 50, 2)


