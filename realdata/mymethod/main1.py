#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import package
import os
#import matplotlib as mpl
#if os.environ.get('DISPLAY','') == '':
#    print('no display found. Using non-interactive Agg backend')
#    mpl.use('Agg')

import numpy as np
#import matplotlib.pyplot as plt
import random
import sys
import pickle
import time
import re
import pandas as pd
import glob
from nilearn import plotting


# Import function
os.chdir('/Users/senna/')

# remove file
filelist = glob.glob(os.getcwd() + '/MyResult' + '/*')
for f in filelist:
    os.remove(f)
print('Remove result file. Done!')


from main2 import *


random.seed(2020)
np.random.seed(2020)

MyMain()




















