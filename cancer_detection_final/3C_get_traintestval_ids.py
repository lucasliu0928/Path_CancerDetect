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
from misc_utils import generate_balanced_cv_list, mutation_sample_summary

warnings.filterwarnings("ignore")
import argparse




############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--cohort_name', default='Neptune', type=str, help='data set name: OPX or TCGA_PRAD or Neptune')
parser.add_argument('--label_path', default= '3B_labels_final_sample', type=str, help='tile info folder name')
parser.add_argument('--out_folder', default= '3C_Train_TEST_IDS', type=str, help='out folder name')

args = parser.parse_args()

############################################################################################################
#USER INPUT 
############################################################################################################
#SELECTED_LABEL = ["AR","HR1","HR2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]

if 'TCGA' in args.cohort_name:
    SELECTED_LABEL = ["AR","HR1","HR2","PTEN","RB1","TP53","MSI_POS"]
else:
    SELECTED_LABEL = ["AR","HR1","HR2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]

##################
###### DIR  ######
##################
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
label_path = os.path.join(proj_dir,'intermediate_data',args.label_path, args.cohort_name, "TFT" + str(args.TUMOR_FRAC_THRES))
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
all_sp_label_df = pd.read_csv(os.path.join(label_path, "final_sample_label_df.csv"))
all_sp_label_df_pt = all_sp_label_df.drop_duplicates(subset = ['PATIENT_ID']).copy() #patient-level
all_sp_label_df_sp = all_sp_label_df.drop_duplicates(subset = ['SAMPLE_ID']).copy()  #sample-level


################################################
#   Generate balanced cross validation set sd 
################################################
n_Folds = 5
p_test = 0.25 
training_lists, validation_lists, test_list = generate_balanced_cv_list(all_sp_label_df_pt, SELECTED_LABEL, n_Folds, p_test)


################################################
#Train and Test and VAl df
################################################
train_test_valid_df = all_sp_label_df_sp.copy()
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


# ################################################
# #Count mutation patient level by Train and Test and val
# ################################################
# --- Sample-level summary --
original_summary = mutation_sample_summary(train_test_valid_df, SELECTED_LABEL)
original_summary['Split'] = 'ALL'
train_summary    = mutation_sample_summary(train_test_valid_df.loc[train_test_valid_df['TRAIN_OR_TEST'] == 'TRAIN'], SELECTED_LABEL)
train_summary['Split'] = 'ALL_TRAIN'
test_summary     = mutation_sample_summary(train_test_valid_df.loc[train_test_valid_df['TRAIN_OR_TEST'] == 'TEST'], SELECTED_LABEL)
test_summary['Split'] = 'ALL_TEST'

fold_sum = []
for i in range(0,5):
    for co in ['TRAIN','TEST','VALID']:
        cur_sum = mutation_sample_summary(train_test_valid_df.loc[train_test_valid_df['FOLD' + str(i)] == co], SELECTED_LABEL)
        cur_sum['Split'] = 'FOLD' + str(i) + '_' + co
        fold_sum.append(cur_sum)
fold_summary = pd.concat(fold_sum)

final_summary = pd.concat([original_summary, train_summary, test_summary, fold_summary])

final_summary.to_csv(os.path.join(outdir, 'Label_Count_Sample_Level.csv'))

