#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 12:49:26 2025

@author: jliu6
"""

#!/usr/bin/env python
# coding: utf-8

#NOTE: use python env acmil in ACMIL folder
import sys
import os
import numpy as np
#%matplotlib inline
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import warnings
sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")


#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse

#Run: python3 -u 7_train_dynamic_tiles_ACMIL_AddReg_working-MultiTasking_NewFeature_TCGA_ACMIL_UpdatedOPX.py --train_cohort TCGA_PRAD --SELECTED_MUTATION AR

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Performance")
parser.add_argument('--s_fold', default=0, type=int, help='select fold')
parser.add_argument('--loss_method', default='', type=str, help='ATTLOSS or ''')
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--learning_method', default='acmil', type=str, help=': e.g., acmil, abmil')
parser.add_argument('--train_cohort', default= 'OPX', type=str, help='TCGA_PRAD or OPX')
parser.add_argument('--train_overlap', default=100, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--mutation', default='MT', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--perf_dir', default= 'pred_out_042125', type=str, help='out folder name')


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    ####################################
    ######      USERINPUT       ########
    ####################################
    label_avail = ["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI_POS"]    #All label avaiable
    
    if args.mutation == 'MT':
        if args.train_cohort == 'OPX':
            SELECTED_LABEL = ["AR","HR1","PTEN","RB1","TP53","TMB","MSI_POS"]
        elif args.train_cohort == 'TCGA_PRAD' or args.train_cohort == 'TCGA_OPX':
            SELECTED_LABEL = ["AR","HR1","PTEN","RB1","TP53","MSI_POS"]   #without TMB
    else:
        SELECTED_LABEL = [args.mutation]
            
        
    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    folder_name1 = args.fe_method + '/TrainOL' + str(args.train_overlap) +  '_TestOL' + str(args.test_overlap) + '_TFT' + str(args.tumor_frac)  + "/"
    perf_path = os.path.join(proj_dir + "intermediate_data/" + args.perf_dir,
                           'trainCohort_' + args.train_cohort,
                           args.learning_method,
                           folder_name1,
                           'FOLD' + str(args.s_fold),
                           args.mutation,
                           "perf")
    
    perf_folders = os.listdir(perf_path)
    metric_cols = ['AUC', 'Recall', 'Specificity', 'ACC', 'Precision', 'PR_AUC', 'F1', 'F2', 'F3']

    test_perf_list = []
    tcga_perf_list = []
    for f in perf_folders:
        if f != 'GAMMA_10_ALPHA_0.3' and f != 'GAMMA_11_ALPHA_0.5':
            cur_perf = pd.read_csv(os.path.join(perf_path,f,'n_token3_TEST_perf.csv'))
            mean_row = cur_perf[metric_cols].mean()
            mean_row['OUTCOME'] = 'MEAN'
            cur_perf = pd.concat([cur_perf, pd.DataFrame([mean_row])], ignore_index=True)
            cur_perf['COHORT'] = 'OPX_TEST'
            cur_perf['FOLDER'] = f
            
            test_perf_list.append(cur_perf)
            cur_perf2 = pd.read_csv(os.path.join(perf_path,f,'n_token3_EXT_perf.csv'))
            mean_row = cur_perf2[metric_cols].mean()
            mean_row['OUTCOME'] = 'MEAN'
            cur_perf2 = pd.concat([cur_perf2, pd.DataFrame([mean_row])], ignore_index=True)
            cur_perf2['COHORT'] = 'EXT'
            cur_perf2['FOLDER'] = f
            tcga_perf_list.append(cur_perf2)
        
        
    test_perf_df = pd.concat(test_perf_list)
    tcga_perf_df = pd.concat(tcga_perf_list)


perf_df = pd.concat([tcga_perf_df,test_perf_df])

perf_df_avg = tcga_perf_df.loc[tcga_perf_df['OUTCOME'] == 'MSI_POS']

# Filter by OUTCOME and COHORT
filtered = perf_df[
    perf_df['OUTCOME'].isin(['MSI_POS', 'HR']) &
    perf_df['COHORT'].isin(['TCGA', 'OPX'])
].copy()

# # Group by FOLDER and compute average AUC and Recall across all rows in that folder
# folder_scores = filtered.groupby('FOLDER')[['AUC', 'Recall']].mean()

# # Optionally: normalize AUC and Recall to give equal weight to both metrics
# folder_scores['AUC_rank'] = folder_scores['AUC'].rank(ascending=False)
# folder_scores['Recall_rank'] = folder_scores['Recall'].rank(ascending=False)
# folder_scores['Avg_rank'] = (folder_scores['AUC_rank'] + folder_scores['Recall_rank']) / 2

# # Get the best folder
# best_folder = folder_scores.sort_values('Avg_rank').head(1)


# best = 'GAMMA_8_ALPHA_0.9'
# best_tcga_df = tcga_perf_df.loc[tcga_perf_df['FOLDER'] ==best]
# best_opx_df = test_perf_df.loc[test_perf_df['FOLDER'] == best]

    


            