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
from misc_utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")
from train_utils import FocalLoss, get_feature_idexes, get_selected_labels

#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse

#Run: python3 -u 7_train_dynamic_tiles_ACMIL_AddReg_working-MultiTasking_NewFeature_TCGA_ACMIL_UpdatedOPX.py --train_cohort TCGA_PRAD --SELECTED_MUTATION AR



import os
import pandas as pd

# Define the root directory containing all the GAMMA_* folders
root_dir = "/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/pred_out_050625_stnormed/trainCohort_TCGA_OPXGRLFalse/acmil/uni2/TrainOL0_TestOL0_TFT0.9/"
root_dir = root_dir + 'FOLD2/MTHR_TYPEHR2/perf/'

# Initialize a list to hold DataFrames
test_perf_dfs = []

# Walk through all subdirectories
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith("n_token3_EXT_perf_bootstrap.csv"):
            full_path = os.path.join(dirpath, file)
            df = pd.read_csv(full_path)
            df['FOLDER'] = full_path.split('/')[-2]
            test_perf_dfs.append(df)

test_perf_dfs= pd.concat(test_perf_dfs)
test_perf_dfs = test_perf_dfs.loc[test_perf_dfs['OUTCOME'] == 'MSI_POS']
            
############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Performance")
parser.add_argument('--s_fold', default=2, type=int, help='select fold')
parser.add_argument('--loss_method', default='', type=str, help='ATTLOSS or ''')
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--learning_method', default='acmil', type=str, help=': e.g., acmil, abmil')
parser.add_argument('--train_cohort', default= 'TCGA_OPX', type=str, help='TCGA_PRAD or OPX')
parser.add_argument('--external_cohort1', default= 'z_nostnorm_TCGA_PRAD', type=str, help='TCGA_PRAD or OPX or Neptune')
parser.add_argument('--external_cohort2', default= 'Neptune', type=str, help='TCGA_PRAD or OPX or Neptune')
parser.add_argument('--train_overlap', default=0, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--mutation', default='MT', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--hr_type', default= "HR2", type=str, help='HR version 1 or 2 (2 only include 3 genes)')
parser.add_argument('--perf_dir', default= 'pred_out_050625_stnormed', type=str, help='out folder name')


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    ####################################
    ######      USERINPUT       ########
    ####################################
    #Label
    SELECTED_LABEL, selected_label_index = get_selected_labels(args.mutation, args.hr_type, args.train_cohort)
    print(SELECTED_LABEL)
    print(selected_label_index)
    
    args.GRL = True
        
    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    folder_name1 = args.fe_method + '/TrainOL' + str(args.train_overlap) +  '_TestOL' + str(args.test_overlap) + '_TFT' + str(args.tumor_frac)  + "/" 
    perf_path = os.path.join(proj_dir + "intermediate_data/" + args.perf_dir,
                           'trainCohort_' + args.train_cohort + 'GRL' + str(args.GRL),
                           args.learning_method,
                           folder_name1,
                           'FOLD' + str(args.s_fold),
                           args.mutation + 'HR_TYPE' + args.hr_type,
                           "perf")    
    perf_folders = os.listdir(perf_path)
    metric_cols = ['AUC', 'Recall', 'Specificity', 'ACC', 'Precision', 'PR_AUC', 'F1', 'F2', 'F3']

    # test_perf_list = []
    # ext_perf_list = []
    # for f in perf_folders:
    #     if f != 'GAMMA_8_ALPHA_0.8':
    #         cur_perf = pd.read_csv(os.path.join(perf_path,f,'n_token3_TEST_perf.csv'))
    #         mean_row = cur_perf[metric_cols].mean()
    #         mean_row['OUTCOME'] = 'MEAN'
    #         cur_perf = pd.concat([cur_perf, pd.DataFrame([mean_row])], ignore_index=True)
    #         cur_perf['COHORT'] = 'OPX_TEST'
    #         cur_perf['FOLDER'] = f
    #         test_perf_list.append(cur_perf)
            
    #         cur_perf2 = pd.read_csv(os.path.join(perf_path,f,'n_token3_EXT_perf.csv'))
    #         mean_row = cur_perf2[metric_cols].mean()
    #         mean_row['OUTCOME'] = 'MEAN'
    #         cur_perf2 = pd.concat([cur_perf2, pd.DataFrame([mean_row])], ignore_index=True)
    #         cur_perf2['COHORT'] = 'EXT'
    #         cur_perf2['FOLDER'] = f
    #         ext_perf_list.append(cur_perf2)
            
    # test_perf_df = pd.concat(test_perf_list)
    # ext_perf_df = pd.concat(ext_perf_list)
    # perf_df = pd.concat([test_perf_df,ext_perf_df])

    test_perf_list = []
    tcga_perf_list = []
    nep_perf_list = []
    for f in perf_folders:
            cur_perf = pd.read_csv(os.path.join(perf_path,f,'n_token3_OPX_perf.csv'))
            mean_row = cur_perf[metric_cols].mean()
            mean_row['OUTCOME'] = 'MEAN'
            cur_perf = pd.concat([cur_perf, pd.DataFrame([mean_row])], ignore_index=True)
            cur_perf['COHORT'] = 'OPX_TEST'
            cur_perf['FOLDER'] = f
            test_perf_list.append(cur_perf)
            
            cur_perf2 = pd.read_csv(os.path.join(perf_path,f,'n_token3_TCGA_perf.csv'))
            mean_row = cur_perf2[metric_cols].mean()
            mean_row['OUTCOME'] = 'MEAN'
            cur_perf2 = pd.concat([cur_perf2, pd.DataFrame([mean_row])], ignore_index=True)
            cur_perf2['COHORT'] = 'TCGA'
            cur_perf2['FOLDER'] = f
            tcga_perf_list.append(cur_perf2)
            
            cur_perf3 = pd.read_csv(os.path.join(perf_path,f,'n_token3_EXT_perf.csv'))
            mean_row = cur_perf3[metric_cols].mean()
            mean_row['OUTCOME'] = 'MEAN'
            cur_perf3 = pd.concat([cur_perf3, pd.DataFrame([mean_row])], ignore_index=True)
            cur_perf3['COHORT'] = 'Neptune'
            cur_perf3['FOLDER'] = f
            nep_perf_list.append(cur_perf3)
        
        
    test_perf_df = pd.concat(test_perf_list)
    tcga_perf_df = pd.concat(tcga_perf_list)
    nep_perf_df = pd.concat(nep_perf_list)
    #Combine all performance
    perf_df = pd.concat([test_perf_df,tcga_perf_df,nep_perf_df])
    #perf_df = perf_df.loc[perf_df['OUTCOME'] == 'MSI_POS']
    
    
    filtered = test_perf_df[
        test_perf_df['OUTCOME'].isin(['MSI_POS'])
    ].copy()
    print(filtered)
    
    f = 'GAMMA_7_ALPHA_0.7'
    cur_perf = pd.read_csv(os.path.join(perf_path,f,'n_token3_OPX_perf_bootstrap.csv'))
    print(cur_perf)
    cur_perf = pd.read_csv(os.path.join(perf_path,f,'n_token3_TCGA_perf_bootstrap.csv'))
    print(cur_perf)
    cur_perf = pd.read_csv(os.path.join(perf_path,f,'n_token3_EXT_perf_bootstrap.csv'))
    print(cur_perf)
    # # Filter by OUTCOME and COHORT
    # filtered = perf_df[
    #     perf_df['OUTCOME'].isin(['MSI_POS', 'HR2'])
    # ].copy()
    
    # #Fin the best for both cohort
    # df = filtered
    
    # # Convert columns to numeric
    # df[['AUC', 'PR_AUC']] = df[['AUC', 'PR_AUC']].apply(pd.to_numeric, errors='coerce')
    
    # # Group by FOLDER and COHORT to compute mean AUC and PR_AUC per folder per cohort
    # agg_df = df.groupby(['FOLDER', 'COHORT'])[['AUC', 'PR_AUC']].mean().reset_index()
    
    # # Pivot to make comparison easier
    # pivot_df = agg_df.pivot(index='FOLDER', columns='COHORT', values=['AUC', 'PR_AUC'])
    
    # # Drop folders that don't appear in both cohorts (i.e., drop NaNs)
    # pivot_df = pivot_df.dropna()
    
    # # Compute average metrics across cohorts
    # pivot_df['AUC_MEAN'] = pivot_df['AUC'].mean(axis=1)
    # pivot_df['PR_AUC_MEAN'] = pivot_df['PR_AUC'].mean(axis=1)
    # pivot_df['SCORE'] = pivot_df['AUC_MEAN'] + pivot_df['PR_AUC_MEAN']  # or use weighted avg
    
    # # Find the best-performing folder
    # best_folder = pivot_df['SCORE'].idxmax()
    # best_score = pivot_df.loc[best_folder]
    
    # print("Best balanced folder across cohorts:", best_folder)
    # print("Average AUC:", best_score['AUC_MEAN'])
    # print("Average PR_AUC:", best_score['PR_AUC_MEAN'])
    
    # print(best_folder)
    # perf_df1 = test_perf_df.loc[test_perf_df['FOLDER'] == best_folder]
    # perf_df2 = tcga_perf_df.loc[tcga_perf_df['FOLDER'] == best_folder]
    # perf_df3 = nep_perf_df.loc[nep_perf_df['FOLDER'] == best_folder]
    
    
    
    # # # Group by FOLDER and compute average AUC and Recall across all rows in that folder
    # # folder_scores = filtered.groupby('FOLDER')[['AUC', 'Recall']].mean()
    
    # # # Optionally: normalize AUC and Recall to give equal weight to both metrics
    # # folder_scores['AUC_rank'] = folder_scores['AUC'].rank(ascending=False)
    # # folder_scores['Recall_rank'] = folder_scores['Recall'].rank(ascending=False)
    # # folder_scores['Avg_rank'] = (folder_scores['AUC_rank'] + folder_scores['Recall_rank']) / 2
    
    # # # Get the best folder
    # # best_folder = folder_scores.sort_values('Avg_rank').head(1)
    
    
    # # best = 'GAMMA_8_ALPHA_0.9'
    # # best_tcga_df = tcga_perf_df.loc[tcga_perf_df['FOLDER'] ==best]
    # # best_opx_df = test_perf_df.loc[test_perf_df['FOLDER'] == best]

    


            