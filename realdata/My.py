#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import numpy as np
import os
import glob


os.chdir('/Users/senna/course/Mladen/pro3/simulation/realdata')

# remove csv file in output folder
filelist = [ f for f in os.listdir(os.getcwd() + '/data/Output')]
for f in filelist:
    os.remove(os.path.join(os.getcwd() + '/data/Output/', f))
print('remove result file, Done!')


## download harvard data
dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', os.getcwd() + '/data/Atlas')
atlas_filename = dataset.maps
labels = dataset.labels
df = pd.DataFrame(labels)
df.to_csv('data/Output/labels.csv')
with open(os.getcwd()+'/data/atlas.txt', 'w+') as f:
    print(atlas_filename, file = f)

## download cobre data
data = datasets.fetch_cobre(n_subjects = 146, data_dir=os.getcwd() + '/data/COBRE')
fmri_filenames = data.func
confounders_filenames = data.confounds
# save data
nLength = len(fmri_filenames)




masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,verbose=5)
# Here we go from nifti files to the signal time series in a numpy
# array. Note how we give confounds to be regressed out during signal
# extraction
C = 0  # index of control
P = 0  # index of patient
for i in range(0, nLength):
    print(i)
    time_series = pd.DataFrame(masker.fit_transform(fmri_filenames[i]))
    confounders_data = pd.read_csv(confounders_filenames[i],  sep='\\t')
    boolVec = confounders_data['scrub']==0
    s = pd.Series(boolVec, name='bools') # We filter the data by "scrub". This is suggested by the dataset author.
    time_series = time_series[s.values]
    if data.phenotypic[i][4][0:1].decode('utf-8') == 'C':
       time_series.to_csv('data/Output/C' + '{}.csv'.format(C))
       C = C+1
    elif data.phenotypic[i][4][0:1].decode('utf-8') == 'P':
       time_series.to_csv('data/Output/P' + '{}.csv'.format(P))
       P = P+1
    print(time_series.shape)



##############################################################
########### Combine all patients and control  ################
##############################################################

# Combine control group
filelist = [ f for f in glob.glob(os.getcwd() + '/data/Output/C*.csv')]
CDat = pd.DataFrame()
for f in filelist:
    CDat = CDat.append(pd.read_csv(f,index_col=0), ignore_index=True)
CDat.to_csv('data/Groupdata/Control.csv')

# Combine patient (test) group
filelist = [ f for f in glob.glob(os.getcwd() + '/data/Output/P*.csv')]
PDat = pd.DataFrame()
for f in filelist:
    PDat = PDat.append(pd.read_csv(f,index_col=0), ignore_index=True)
PDat.to_csv('data/Groupdata/Test.csv')


































