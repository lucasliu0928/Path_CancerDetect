#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NOTE: use paimg9 env
import sys
import os
import numpy as np
import openslide
import pandas as pd
import warnings
import torch
import torch.nn as nn

sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists, count_label, set_seed
from train_utils import ModelReadyData_diffdim_V2, get_feature_idexes, get_sample_feature, get_sample_label, combine_feature_and_label
warnings.filterwarnings("ignore")


# In[2]:


####################################
######      USERINPUT       ########
####################################
pixel_overlap = 100    
save_image_size = 250
TUMOR_FRAC_THRES = 0.9
cohort_name = "OPX"  
feature_extraction_method = 'uni2' #retccl, uni1, prov_gigapath
SELECTED_LABEL = ["AR","HR","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]
SELECTED_FEATURE = get_feature_idexes(feature_extraction_method, include_tumor_fraction = False)

##################
###### DIR  ######
##################
folder_name = "IMSIZE" + str(save_image_size) + "_OL" + str(pixel_overlap)
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
feature_path = os.path.join(proj_dir,'intermediate_data','4_tile_feature', cohort_name, folder_name)
info_path =   os.path.join(proj_dir,'intermediate_data','3_otherinfo', cohort_name, folder_name)
info_path2 =    os.path.join(proj_dir,'intermediate_data','2_cancer_detection', cohort_name, folder_name)


################################################
#Create output dir
################################################
outdir =  os.path.join(proj_dir + 'intermediate_data/5_model_ready_data',
                       cohort_name, 
                       folder_name, 
                       'feature_' + feature_extraction_method, 
                       'TFT' + str(TUMOR_FRAC_THRES))
create_dir_if_not_exists(outdir)

##################
#Select GPU
##################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
set_seed(0)


# In[3]:


############################################################################################################
#Load all tile info df
#This file contains all tiles without cancer fraction exclusion and  has tissue membership > 0.9, white space < 0.9 (non white space > 0.1)
############################################################################################################
all_tile_info_df = pd.read_csv(os.path.join(info_path, "all_tile_info.csv"))
selected_ids = list(all_tile_info_df['SAMPLE_ID'].unique())
selected_ids.sort()


# In[4]:


############################################################################################################
#Get model ready data
############################################################################################################
comb_df_list = []
ct = 0 
for pt in selected_ids:
    if ct % 10 == 0 : print(ct)
    #Get feature
    feature_df = get_sample_feature(pt, feature_path, feature_extraction_method)
    
    #Get label
    label_df = get_sample_label(pt,all_tile_info_df)
    
    #Merge feature and label
    comb_df = combine_feature_and_label(feature_df,label_df)
    
    #Select tumor fraction > X tiles
    comb_df = comb_df.loc[comb_df['TUMOR_PIXEL_PERC'] >= TUMOR_FRAC_THRES].copy()
    comb_df = comb_df.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
    comb_df = comb_df.reset_index(drop = True)
    comb_df_list.append(comb_df)
    ct += 1

all_comb_df = pd.concat(comb_df_list)

#Get model ready data
data = ModelReadyData_diffdim_V2(comb_df_list, SELECTED_FEATURE, SELECTED_LABEL)
torch.save(data, os.path.join(outdir, cohort_name + '_data.pth'))


# In[5]:


############################################################################################################
#Count Distribution
############################################################################################################
#Tile level
counts1 = count_label(all_comb_df, SELECTED_LABEL, cohort_name + "_TILE")
#print(counts1)
patient_level_comb_df = all_comb_df.drop_duplicates(subset = ['SAMPLE_ID'])
counts2 = count_label(patient_level_comb_df, SELECTED_LABEL, cohort_name + "_SAMPLE")
#print(counts2)
counts = counts2.merge(counts1, left_index = True, right_index = True)
counts.to_csv(os.path.join(outdir, cohort_name + '_counts.csv'))


# In[6]:


all_comb_df.shape

