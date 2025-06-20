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
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists
from misc_utils import count_mutation_perTrainTest, count_mutation_perTrainTestVAL, get_pos_neg_ids
from misc_utils import generate_balanced_cv_list

warnings.filterwarnings("ignore")
import argparse




############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--save_image_size', default=250, type=int, help='the size of extracted tiles')
parser.add_argument('--pixel_overlap', default=100, type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--cohort_name', default='z_nostnorm_OPX', type=str, help='data set name: OPX or TCGA_PRAD or Neptune' or 'z_nostnorm_OPX')
parser.add_argument('--tile_info_path', default= '3A_otherinfo', type=str, help='tile info folder name')
parser.add_argument('--out_folder', default= '3B_Train_TEST_IDS', type=str, help='out folder name')

args = parser.parse_args()

############################################################################################################
#USER INPUT 
############################################################################################################
#SELECTED_LABEL = ["AR","HR1","HR2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]

if 'TCGA' in args.cohort_name:
    SELECTED_LABEL = ["AR","HR2","PTEN","RB1","TP53","MSI_POS"]
else:
    SELECTED_LABEL = ["AR","HR2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]

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
tile_info_df_pt = tile_info_df.drop_duplicates(subset = ['PATIENT_ID']).copy() #patient-level
tile_info_df_sp = tile_info_df.drop_duplicates(subset = ['SAMPLE_ID']).copy()  #sample-level


################################################
#   Generate balanced cross validation set sd 
################################################
n_Folds = 5
p_test = 0.25 
training_lists, validation_lists, test_list = generate_balanced_cv_list(tile_info_df_pt, SELECTED_LABEL, n_Folds, p_test)



################################################
#Train and Test and VAl df
################################################
train_test_valid_df = tile_info_df_pt[['SAMPLE_ID', 'PATIENT_ID'] + SELECTED_LABEL].copy()
train_test_valid_df['TRAIN_OR_TEST'] = pd.NA
cond = train_test_valid_df['PATIENT_ID'].isin(test_list)
train_test_valid_df.loc[cond, 'TRAIN_OR_TEST'] = 'TEST'
train_test_valid_df.loc[~cond, 'TRAIN_OR_TEST'] = 'TRAIN'

for k in range(n_Folds):
    #Update dataframe
    cond1 = train_test_valid_df['PATIENT_ID'].isin(training_lists[k])
    train_test_valid_df.loc[cond1, 'FOLD' + str(k)] = 'TRAIN'
    cond2 = train_test_valid_df['PATIENT_ID'].isin(validation_lists[k])
    train_test_valid_df.loc[cond2, 'FOLD' + str(k)] = 'VALID'
    cond3 = ~(cond1 | cond2)
    train_test_valid_df.loc[cond3, 'FOLD' + str(k)] = 'TEST'

train_test_valid_df.to_csv(os.path.join(outdir, 'train_test_split.csv'))



################################################
#Count mutation patient level by Train and Test and val
################################################
count_pt_traintest = count_mutation_perTrainTest(train_test_valid_df,SELECTED_LABEL)
count_pt_traintest['Val N (%)'] = pd.NA
count_pt_traintest['FOLD'] = pd.NA
count_pt_traintest = count_pt_traintest[['Outcome', 'Train N (%)', 'Val N (%)', 'Test N (%)', 'FOLD']]

count_pt_traintestval = count_mutation_perTrainTestVAL(train_test_valid_df, SELECTED_LABEL, n_Folds = 5)


count_pt = pd.concat([count_pt_traintest,count_pt_traintestval])
count_pt.to_csv(os.path.join(outdir, 'label_count_patient_level.csv'))


