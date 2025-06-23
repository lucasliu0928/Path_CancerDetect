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


                
    
    
import os
import pandas as pd
def compute_meanstd_perf(indata, cohort_name):
    outcome = ['AR', 'HR2', 'PTEN', 'RB1', 'TP53', 'MSI_POS']
    metrics = ['AUC', 'Recall', 'Specificity', 'ACC', 'Precision','PR_AUC', 'F1', 'F2', 'F3']
    summary_list = []
    for oc in outcome:
        df = indata.loc[indata['OUTCOME'] == oc]
        
        # Compute mean and std
        mean_values = df[metrics].mean()
        std_values = df[metrics].std()
        
        # Format to "mean ± std" with 2-digit precision
        # Format "mean ± std" and store in a new DataFrame
        formatted = {metric: f"{mean_values[metric]:.2f} ± {std_values[metric]:.2f}" for metric in metrics}
        summary_transposed = pd.DataFrame([formatted])
        summary_transposed['OUTCOME'] = oc
        summary_list.append(summary_transposed)
    summary_list = pd.concat(summary_list)
    summary_list['COHORT'] = cohort_name

    return summary_list

# Define the root directory containing all the GAMMA_* folders
root_dir = "/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/pred_out_061825_sample1000tiles_train_z_nonorm_OPX_TCGA_GRLFALSE/"
root_dir = root_dir + "trainCohort_z_nostnorm_OPX_TCGA_GRLFalse/acmil/uni2/TrainOL100_TestOL0_TFT0.9/"
#root_dir = root_dir + 'FOLD3/MTHR_TYPEHR2/perf/'

folds_list = ['FOLD0','FOLD1', 'FOLD2', 'FOLD3','FOLD4']
# Initialize a list to hold DataFrames
holdout_test_perf_dfs = []
val_perf_dfs = []
nep1_perf_dfs = []
nep2_perf_dfs = []

for f in folds_list:
    dirpath = root_dir + f + '/'
    perfpath = dirpath + 'MTHR_TYPEHR2/perf/GAMMA_6.0_ALPHA_0.9/'
    
    val_df = pd.read_csv(perfpath + 'n_token3_VAL__perf.csv')
    val_df['FOLD'] = f
    holdout_test_df = pd.read_csv(perfpath + 'n_token3_TEST_COMB_perf.csv')
    holdout_test_df['FOLD'] = f
    nep_df1 = pd.read_csv(perfpath + 'n_token3_NEP_perf.csv')
    nep_df1['FOLD'] = f
    nep_df2 = pd.read_csv(perfpath + 'n_token3_z_nostnorm_NEP_perf.csv')
    nep_df2['FOLD'] = f
    
    val_perf_dfs.append(val_df)
    holdout_test_perf_dfs.append(holdout_test_df)
    nep1_perf_dfs.append(nep_df1)
    nep2_perf_dfs.append(nep_df2)

val_perf_dfs= pd.concat(val_perf_dfs)
holdout_test_perf_dfs= pd.concat(holdout_test_perf_dfs)
nep1_perf_dfs= pd.concat(nep1_perf_dfs)
nep2_perf_dfs= pd.concat(nep2_perf_dfs)    

val_perf_dfs = compute_meanstd_perf(val_perf_dfs, cohort_name = 'VAL')
holdout_test_perf_dfs = compute_meanstd_perf(holdout_test_perf_dfs, cohort_name = 'HOLD_OUT_TEST_TCGA_OPX')
nep1_perf_dfs = compute_meanstd_perf(nep1_perf_dfs, cohort_name = 'NEP_STN')
nep2_perf_dfs = compute_meanstd_perf(nep2_perf_dfs, cohort_name = 'NEP_noSTN')

final_df = pd.concat([val_perf_dfs,holdout_test_perf_dfs,nep1_perf_dfs,nep2_perf_dfs])
final_df = final_df[['COHORT','OUTCOME','AUC', 'Recall', 'PR_AUC', 'Specificity', 'ACC', 'Precision', 'F1', 'F2', 'F3']]
final_df.to_csv(root_dir + "allfolds_perf.csv",index = False, encoding='utf-8-sig')



        
        
    
    # # # Walk through all subdirectories
    # # for dirpath, dirnames, filenames in os.walk(root_dir):
    # #     print(dirpath, dirnames, filenames)
    # #     for file in filenames:
    # #         if file.endswith("_TEST_COMB_perf.csv"):
    # #             full_path = os.path.join(dirpath, file)
    # #             df = pd.read_csv(full_path)
    # #             df['FOLDER'] = full_path.split('/')[-2]
    # #             test_perf_dfs.append(df)
    
    # # test_perf_dfs= pd.concat(test_perf_dfs)
    # # test_perf_dfs = test_perf_dfs.loc[test_perf_dfs['OUTCOME'] == 'MSI_POS']
                
    # ############################################################################################################
    # #Parser
    # ############################################################################################################
    # parser = argparse.ArgumentParser("Performance")
    # parser.add_argument('--s_fold', default=2, type=int, help='select fold')
    # parser.add_argument('--loss_method', default='', type=str, help='ATTLOSS or ''')
    # parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
    # parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
    # parser.add_argument('--learning_method', default='acmil', type=str, help=': e.g., acmil, abmil')
    # parser.add_argument('--train_cohort', default= 'TCGA_OPX', type=str, help='TCGA_PRAD or OPX')
    # parser.add_argument('--external_cohort1', default= 'z_nostnorm_TCGA_PRAD', type=str, help='TCGA_PRAD or OPX or Neptune')
    # parser.add_argument('--external_cohort2', default= 'Neptune', type=str, help='TCGA_PRAD or OPX or Neptune')
    # parser.add_argument('--train_overlap', default=0, type=int, help='train data pixel overlap')
    # parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
    # parser.add_argument('--mutation', default='MT', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
    # parser.add_argument('--hr_type', default= "HR2", type=str, help='HR version 1 or 2 (2 only include 3 genes)')
    # parser.add_argument('--perf_dir', default= 'pred_out_050625_stnormed', type=str, help='out folder name')
    
    
    # if __name__ == '__main__':
        
    #     args = parser.parse_args()
        
    #     ####################################
    #     ######      USERINPUT       ########
    #     ####################################
    #     #Label
    #     SELECTED_LABEL, selected_label_index = get_selected_labels(args.mutation, args.hr_type, args.train_cohort)
    #     print(SELECTED_LABEL)
    #     print(selected_label_index)
        
    #     args.GRL = True
            
    #     ##################
    #     ###### DIR  ######
    #     ##################
    #     proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    #     folder_name1 = args.fe_method + '/TrainOL' + str(args.train_overlap) +  '_TestOL' + str(args.test_overlap) + '_TFT' + str(args.tumor_frac)  + "/" 
    #     perf_path = os.path.join(proj_dir + "intermediate_data/" + args.perf_dir,
    #                            'trainCohort_' + args.train_cohort + 'GRL' + str(args.GRL),
    #                            args.learning_method,
    #                            folder_name1,
    #                            'FOLD' + str(args.s_fold),
    #                            args.mutation + 'HR_TYPE' + args.hr_type,
    #                            "perf")    
    #     perf_folders = os.listdir(perf_path)
    #     metric_cols = ['AUC', 'Recall', 'Specificity', 'ACC', 'Precision', 'PR_AUC', 'F1', 'F2', 'F3']
    
    #     # test_perf_list = []
    #     # ext_perf_list = []
    #     # for f in perf_folders:
    #     #     if f != 'GAMMA_8_ALPHA_0.8':
    #     #         cur_perf = pd.read_csv(os.path.join(perf_path,f,'n_token3_TEST_perf.csv'))
    #     #         mean_row = cur_perf[metric_cols].mean()
    #     #         mean_row['OUTCOME'] = 'MEAN'
    #     #         cur_perf = pd.concat([cur_perf, pd.DataFrame([mean_row])], ignore_index=True)
    #     #         cur_perf['COHORT'] = 'OPX_TEST'
    #     #         cur_perf['FOLDER'] = f
    #     #         test_perf_list.append(cur_perf)
                
    #     #         cur_perf2 = pd.read_csv(os.path.join(perf_path,f,'n_token3_EXT_perf.csv'))
    #     #         mean_row = cur_perf2[metric_cols].mean()
    #     #         mean_row['OUTCOME'] = 'MEAN'
    #     #         cur_perf2 = pd.concat([cur_perf2, pd.DataFrame([mean_row])], ignore_index=True)
    #     #         cur_perf2['COHORT'] = 'EXT'
    #     #         cur_perf2['FOLDER'] = f
    #     #         ext_perf_list.append(cur_perf2)
                
    #     # test_perf_df = pd.concat(test_perf_list)
    #     # ext_perf_df = pd.concat(ext_perf_list)
    #     # perf_df = pd.concat([test_perf_df,ext_perf_df])
    
    #     test_perf_list = []
    #     tcga_perf_list = []
    #     nep_perf_list = []
    #     for f in perf_folders:
    #             cur_perf = pd.read_csv(os.path.join(perf_path,f,'n_token3_OPX_perf.csv'))
    #             mean_row = cur_perf[metric_cols].mean()
    #             mean_row['OUTCOME'] = 'MEAN'
    #             cur_perf = pd.concat([cur_perf, pd.DataFrame([mean_row])], ignore_index=True)
    #             cur_perf['COHORT'] = 'OPX_TEST'
    #             cur_perf['FOLDER'] = f
    #             test_perf_list.append(cur_perf)
                
    #             cur_perf2 = pd.read_csv(os.path.join(perf_path,f,'n_token3_TCGA_perf.csv'))
    #             mean_row = cur_perf2[metric_cols].mean()
    #             mean_row['OUTCOME'] = 'MEAN'
    #             cur_perf2 = pd.concat([cur_perf2, pd.DataFrame([mean_row])], ignore_index=True)
    #             cur_perf2['COHORT'] = 'TCGA'
    #             cur_perf2['FOLDER'] = f
    #             tcga_perf_list.append(cur_perf2)
                
    #             cur_perf3 = pd.read_csv(os.path.join(perf_path,f,'n_token3_EXT_perf.csv'))
    #             mean_row = cur_perf3[metric_cols].mean()
    #             mean_row['OUTCOME'] = 'MEAN'
    #             cur_perf3 = pd.concat([cur_perf3, pd.DataFrame([mean_row])], ignore_index=True)
    #             cur_perf3['COHORT'] = 'Neptune'
    #             cur_perf3['FOLDER'] = f
    #             nep_perf_list.append(cur_perf3)
            
            
    #     test_perf_df = pd.concat(test_perf_list)
    #     tcga_perf_df = pd.concat(tcga_perf_list)
    #     nep_perf_df = pd.concat(nep_perf_list)
    #     #Combine all performance
    #     perf_df = pd.concat([test_perf_df,tcga_perf_df,nep_perf_df])
    #     #perf_df = perf_df.loc[perf_df['OUTCOME'] == 'MSI_POS']
        
        
    #     filtered = test_perf_df[
    #         test_perf_df['OUTCOME'].isin(['MSI_POS'])
    #     ].copy()
    #     print(filtered)
        
    #     f = 'GAMMA_7_ALPHA_0.7'
    #     cur_perf = pd.read_csv(os.path.join(perf_path,f,'n_token3_OPX_perf_bootstrap.csv'))
    #     print(cur_perf)
    #     cur_perf = pd.read_csv(os.path.join(perf_path,f,'n_token3_TCGA_perf_bootstrap.csv'))
    #     print(cur_perf)
    #     cur_perf = pd.read_csv(os.path.join(perf_path,f,'n_token3_EXT_perf_bootstrap.csv'))
    #     print(cur_perf)
    #     # # Filter by OUTCOME and COHORT
    #     # filtered = perf_df[
    #     #     perf_df['OUTCOME'].isin(['MSI_POS', 'HR2'])
    #     # ].copy()
        
    #     # #Fin the best for both cohort
    #     # df = filtered
        
    #     # # Convert columns to numeric
    #     # df[['AUC', 'PR_AUC']] = df[['AUC', 'PR_AUC']].apply(pd.to_numeric, errors='coerce')
        
    #     # # Group by FOLDER and COHORT to compute mean AUC and PR_AUC per folder per cohort
    #     # agg_df = df.groupby(['FOLDER', 'COHORT'])[['AUC', 'PR_AUC']].mean().reset_index()
        
    #     # # Pivot to make comparison easier
    #     # pivot_df = agg_df.pivot(index='FOLDER', columns='COHORT', values=['AUC', 'PR_AUC'])
        
    #     # # Drop folders that don't appear in both cohorts (i.e., drop NaNs)
    #     # pivot_df = pivot_df.dropna()
        
    #     # # Compute average metrics across cohorts
    #     # pivot_df['AUC_MEAN'] = pivot_df['AUC'].mean(axis=1)
    #     # pivot_df['PR_AUC_MEAN'] = pivot_df['PR_AUC'].mean(axis=1)
    #     # pivot_df['SCORE'] = pivot_df['AUC_MEAN'] + pivot_df['PR_AUC_MEAN']  # or use weighted avg
        
    #     # # Find the best-performing folder
    #     # best_folder = pivot_df['SCORE'].idxmax()
    #     # best_score = pivot_df.loc[best_folder]
        
    #     # print("Best balanced folder across cohorts:", best_folder)
    #     # print("Average AUC:", best_score['AUC_MEAN'])
    #     # print("Average PR_AUC:", best_score['PR_AUC_MEAN'])
        
    #     # print(best_folder)
    #     # perf_df1 = test_perf_df.loc[test_perf_df['FOLDER'] == best_folder]
    #     # perf_df2 = tcga_perf_df.loc[tcga_perf_df['FOLDER'] == best_folder]
    #     # perf_df3 = nep_perf_df.loc[nep_perf_df['FOLDER'] == best_folder]
        
        
        
    #     # # # Group by FOLDER and compute average AUC and Recall across all rows in that folder
    #     # # folder_scores = filtered.groupby('FOLDER')[['AUC', 'Recall']].mean()
        
    #     # # # Optionally: normalize AUC and Recall to give equal weight to both metrics
    #     # # folder_scores['AUC_rank'] = folder_scores['AUC'].rank(ascending=False)
    #     # # folder_scores['Recall_rank'] = folder_scores['Recall'].rank(ascending=False)
    #     # # folder_scores['Avg_rank'] = (folder_scores['AUC_rank'] + folder_scores['Recall_rank']) / 2
        
    #     # # # Get the best folder
    #     # # best_folder = folder_scores.sort_values('Avg_rank').head(1)
        
        
    #     # # best = 'GAMMA_8_ALPHA_0.9'
    #     # # best_tcga_df = tcga_perf_df.loc[tcga_perf_df['FOLDER'] ==best]
    #     # # best_opx_df = test_perf_df.loc[test_perf_df['FOLDER'] == best]
    
        
    
    
                