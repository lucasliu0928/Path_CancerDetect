#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NOTE: use paimg9 env
import sys
import os
import numpy as np
import openslide
import matplotlib.pyplot as plt
import random

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import warnings
import torch
import torch.nn as nn

from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset, ConcatDataset
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


####################################
######      USERINPUT       ########
####################################
SELECTED_LABEL = ["AR","HR","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]
TUMOR_FRAC_THRES = 0.9 
feature_extraction_method = 'uni2' #retccl, uni1, prov_gigapath
N_FEATURE = 1024
#SELECTED_FEATURE = get_feature_idexes(feature_extraction_method, include_tumor_fraction = False)
save_image_size = 250
train_overlap = 100
test_overlap = 0
##################
###### DIR  ######
##################
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
folder_name_train = "IMSIZE" + str(save_image_size) + "_OL" + str(train_overlap)
folder_name_test = "IMSIZE" + str(save_image_size) + "_OL" + str(test_overlap)
feature_path_opx_train = os.path.join(proj_dir + 'intermediate_data/5_model_ready_data',"OPX", folder_name_train, 'feature_' + feature_extraction_method, 'TFT' + str(TUMOR_FRAC_THRES))
feature_path_opx_test  = os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', "OPX", folder_name_test, 'feature_' + feature_extraction_method, 'TFT' + str(TUMOR_FRAC_THRES))



################################################
#Create output dir
################################################
outdir =  os.path.join(proj_dir + 'intermediate_data/6_Train_TEST_IDS', "TrainOL" + str(train_overlap) + "_TestOL"+ str(test_overlap) +  "_TFT" + str(TUMOR_FRAC_THRES))
create_dir_if_not_exists(outdir)

##################
#Select GPU
##################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[3]:


################################################
#    Load OPX IDs
################################################
opx_ids_ol0 = torch.load(os.path.join(feature_path_opx_test,'OPX_data.pth'))


# In[4]:


opx_pt_ids = [x[-1] for x in opx_ids_ol0]
print('total patient ID:',len(set(opx_pt_ids)))

opx_sp_ids = [x[-2] for x in opx_ids_ol0]
print('total sample ID:',len(set(opx_sp_ids)))

labels = [x[1].numpy() for x in opx_ids_ol0]
labels_df = pd.DataFrame(np.concatenate(labels))
labels_df.columns = SELECTED_LABEL
labels_df['SAMPLE_ID'] = opx_sp_ids
labels_df['PATIENT_ID'] = opx_pt_ids


# In[5]:


#All unique pateint IDS
unique_pt_ids = [x for x in list(set(opx_pt_ids))]

#This MSI-H OPX must in Test, because at this point, we want to have more MSI to test
unique_msi_ids = list(labels_df.loc[labels_df['MSI_POS'] == 1 , 'PATIENT_ID'].unique()) #24

#All unique patients IDs no MSI
unique_pt_ids_no_msi = [x for x in unique_pt_ids if x not in unique_msi_ids] #242

#Get 75% (199) for Traninig, 25% (67) for Test at patient level
#1.Get test and train from no msi samples
n_test  = 67 - 12 # origial n - msi 
n_train = 199 - 12 #oriignal n - msi
prec_test = n_test/len(unique_pt_ids_no_msi)
train_ids_nomsi, test_ids_nomsi = train_test_split(unique_pt_ids_no_msi, test_size=prec_test, random_state=42)

#2.Get test and train from msi samples 50 and 50
train_ids_msi, test_ids_msi = train_test_split(unique_msi_ids, test_size=0.5, random_state=42)

#Get all train and test
train_ids_full = train_ids_nomsi + train_ids_msi
test_ids = test_ids_nomsi + test_ids_msi


# In[6]:


#For train_ids_full, then k-fold validation
n_splits = 5 
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) #Initialize KFold
fold_ids = {}
for fold, (train_index, val_index) in enumerate(kf.split(train_ids_full)):
    train_ids = [train_ids_full[i] for i in train_index]  # Get train IDs
    val_ids = [train_ids_full[i] for i in val_index]  # Get val IDs
    fold_ids[fold] = {'Train': train_ids, 'Val' :val_ids}  # Store as lists


# In[7]:


#Inital df
train_test_valid_df = labels_df

train_test_valid_df['TRAIN_OR_TEST'] = pd.NA
cond = train_test_valid_df['PATIENT_ID'].isin(train_ids_full)
train_test_valid_df.loc[cond, 'TRAIN_OR_TEST'] = 'TRAIN'
train_test_valid_df.loc[~cond, 'TRAIN_OR_TEST'] = 'TEST'
for k in range(n_splits):
    #Update dataframe
    cond1 = train_test_valid_df['PATIENT_ID'].isin(fold_ids[k]['Train'])
    train_test_valid_df.loc[cond1, 'FOLD' + str(k)] = 'TRAIN'
    cond2 = train_test_valid_df['PATIENT_ID'].isin(fold_ids[k]['Val'])
    train_test_valid_df.loc[cond2, 'FOLD' + str(k)] = 'VALID'
    cond3 = ~(cond1 | cond2)
    train_test_valid_df.loc[cond3, 'FOLD' + str(k)] = 'TEST'

train_test_valid_df.to_csv(os.path.join(outdir, 'train_test_split.csv'))


# In[8]:


train_test_valid_df['TRAIN_OR_TEST'].value_counts()


# In[9]:


# Compute N (count) for each mutation per TRAIN/TEST group
summary_N = train_test_valid_df.groupby("TRAIN_OR_TEST")[SELECTED_LABEL].sum()

# Compute % within each TRAIN/TEST group
summary_percentage = summary_N.div(summary_N.sum(axis=1), axis=0) * 100  # Normalize per row
summary_percentage = summary_percentage.round(1)  # Round to 1 decimal place

# Format as "N (X.X%)" using f-strings
summary_combined = summary_N.astype(int).astype(str) + " (" + summary_percentage.astype(str) + "%)"

# Transpose the dataframe for better readability
summary_combined = summary_combined.T

summary_combined.to_csv(os.path.join(outdir, 'label_count.csv'))


# In[ ]:


# #This new added OPX must in train or test, because at this point, we want to have more MSI to test
# train_must = ['OPX_207','OPX_209','OPX_213','OPX_214','OPX_215']
# test_must = ['OPX_208','OPX_210','OPX_211','OPX_212','OPX_216']
# train_test_valid_df.loc[train_test_valid_df['SAMPLE_ID'].isin(train_must), ['FOLD0','FOLD1','FOLD2','FOLD3','FOLD4']] = 'TRAIN'
# train_test_valid_df.loc[train_test_valid_df['SAMPLE_ID'].isin(test_must), ['FOLD0','FOLD1','FOLD2','FOLD3','FOLD4']] = 'TEST'

