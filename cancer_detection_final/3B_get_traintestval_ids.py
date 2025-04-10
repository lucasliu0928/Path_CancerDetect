#!/usr/bin/env python
# coding: utf-8
#NOTE: use paimg9 env
import sys
import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import warnings
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
from sklearn.model_selection import KFold, train_test_split,StratifiedKFold
sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")
import argparse

def count_mutation_perTrainTest(train_test_df, selected_label):
    # Compute N (count) per TRAIN/TEST group
    summary_n = train_test_df.groupby("TRAIN_OR_TEST")[selected_label].sum()
    
    # Compute % per  each TRAIN/TEST group
    summary_percentage = (summary_n.div(summary_n.sum(axis=1), axis=0) * 100).round(1)  # Normalize per row
    summary_combined = summary_n.astype(int).astype(str) + " (" + summary_percentage.astype(str) + "%)" #format
    summary_combined = summary_combined.T
    
    return summary_combined

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--save_image_size', default=250, type=int, help='the size of extracted tiles')
parser.add_argument('--pixel_overlap', default=100, type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--feature_extraction_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--cohort_name', default='TCGA_PRAD', type=str, help='data set name: OPX or TCGA_PRAD')
parser.add_argument('--tile_info_path', default= '3A_otherinfo', type=str, help='tile info folder name')
parser.add_argument('--out_folder', default= '3B_Train_TEST_IDS', type=str, help='out folder name')

args = parser.parse_args()

############################################################################################################
#USER INPUT 
############################################################################################################
SELECTED_LABEL = ["AR","HR","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]


##################
###### DIR  ######
##################
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
folder_name = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.pixel_overlap)
info_dir = os.path.join(proj_dir,'intermediate_data',args.tile_info_path, args.cohort_name, folder_name)
outdir =  os.path.join(proj_dir + 'intermediate_data', args.out_folder, args.cohort_name, "TFT" + str(args.TUMOR_FRAC_THRES))
create_dir_if_not_exists(outdir)


##################
#Select GPU
##################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


################################################
#    Load labels from tile info
################################################
tile_info_df = pd.read_csv(os.path.join(info_dir,"all_tile_info.csv"))
tile_info_df = tile_info_df[tile_info_df['TUMOR_PIXEL_PERC']>=args.TUMOR_FRAC_THRES]
tile_info_df_pt = tile_info_df.drop_duplicates(subset = ['PATIENT_ID']) #patient-level


################################################
#Get All patient IDs
################################################
unique_pt_ids = list(tile_info_df_pt['PATIENT_ID'].unique()) #266 OPX, 402 TCGA

################################################
#Get numers for traini nad test at patient level, 75%, 25%
################################################
n_train = round(len(unique_pt_ids)*0.75) #200
n_test =  len(unique_pt_ids) - n_train #66


################################################
#Get half MSI in train and test
#Because we want to have more MSI in test now
################################################
#Has MSI
unique_msi_ids = list(tile_info_df_pt.loc[tile_info_df_pt['MSI_POS'] == 1 , 'PATIENT_ID'].unique()) #24

#Get test and train from msi samples half and half
train_ids_msi, test_ids_msi = train_test_split(unique_msi_ids, test_size=0.5, random_state=42)


################################################
#Get train and test from no MSI
################################################
unique_nomsi_ids = list(tile_info_df_pt.loc[tile_info_df_pt['MSI_POS'] == 0 , 'PATIENT_ID'].unique()) #242

#1.Get Ns of test and train from no msi samples
n_train_nomsi = n_train - len(train_ids_msi) #oriignal n - msi
n_test_nomsi  = n_test -  len(test_ids_msi)  # origial n - msi 
prec_test_nomsi = round(n_test_nomsi/len(unique_nomsi_ids),2) #0.22
train_ids_nomsi, test_ids_nomsi = train_test_split(unique_nomsi_ids, test_size=prec_test_nomsi, random_state=42)

################################################
#Get all train and test
################################################
train_ids  = train_ids_nomsi + train_ids_msi
test_ids   = test_ids_nomsi + test_ids_msi


# #For train_ids_full, then k-fold validation
# n_splits = 5 
# kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) #Initialize KFold
# fold_ids = {}
# for fold, (train_index, val_index) in enumerate(kf.split(train_ids)):
#     train_ids_fold = [train_ids[i] for i in train_index]  # Get train IDs
#     val_ids = [train_ids[i] for i in val_index]  # Get val IDs
#     fold_ids[fold] = {'Train': train_ids_fold, 'Val' :val_ids}  # Store as lists

#stratified by y    , make sure we have pos in validation
n_splits = 5 
y = tile_info_df_pt.loc[tile_info_df_pt['PATIENT_ID'].isin(train_ids), SELECTED_LABEL]
if args.cohort_name == 'TCGA_PRAD':
    y.drop(columns=["TMB_HIGHorINTERMEDITATE"], inplace = True)
mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_ids = {}
for fold, (train_index, val_index) in enumerate(mskf.split(np.zeros(len(y)), y)):
    train_ids_fold = [train_ids[i] for i in train_index]  # Get train IDs
    val_ids = [train_ids[i] for i in val_index]  # Get val IDs
    fold_ids[fold] = {'Train': train_ids_fold, 'Val' :val_ids}  # Store as lists



################################################
#Train and Test and VAl df
################################################
train_test_valid_df = tile_info_df_pt[['SAMPLE_ID', 'PATIENT_ID'] + SELECTED_LABEL].copy()
train_test_valid_df['TRAIN_OR_TEST'] = pd.NA
cond = train_test_valid_df['PATIENT_ID'].isin(train_ids)
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



train_test_valid_df['TRAIN_OR_TEST'].value_counts()

################################################
#Count mutation Sample level by Train and Test
################################################
count_pt = count_mutation_perTrainTest(train_test_valid_df,SELECTED_LABEL)
count_pt.to_csv(os.path.join(outdir, 'label_count_patient_level.csv'))