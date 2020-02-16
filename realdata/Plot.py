#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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
os.chdir('/Users/senna/course/Mladen/project/pro3/BiometrikaRevision/simu/realdata')
# import label and coordinate
Lab = pd.read_csv(os.getcwd() + '/data/Output/labels.csv', index_col=0)['0'].tolist()[1:]
f = open(os.getcwd()+'/data/atlas.txt', "r")
atlas = f.read().splitlines()[0]
Coor = plotting.find_parcellation_cut_coords(labels_img=atlas)
f.close()


# load our result
mySparse = pd.read_csv(os.getcwd()+'/mymethod/MyResult/mySparse.csv', index_col=0).values






#################################
########### Plot     ############
#################################

# show brain

# plot connectome with 80% edge strength in the connectivity
fig = plotting.plot_connectome(mySparse, Coor,edge_threshold="80%",colorbar = True)
fig.savefig('Figures/my.png', dpi = 250)

