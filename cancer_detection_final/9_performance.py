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
    outcome = ['AR', 'HR2', 'PTEN', 'RB1', 'TP53', 'MSI']
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
root_dir = "/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/pred_out_081225_V2/"
root_dir = root_dir + "trainCohort_union_STNandNSTN_OPX_TCGA_NEP_Samples1000/acmil/uni2/TrainOL100_TestOL0_TFT0.9/"



folds_list = ['FOLD0','FOLD1', 'FOLD2', 'FOLD3','FOLD4']
mutations = ['AR', 'HR2', 'PTEN', 'RB1', 'TP53', 'MSI']

mutation_perf_list = []

for mutation in mutations:
    # Initialize a list to hold DataFrames
    holdout_test_perf_dfs = []
    val_perf_dfs = []
    nep1_perf_dfs = []
    nep2_perf_dfs = []
    nep3_perf_dfs = []
    
    for f in folds_list:
        dirpath = root_dir + f + '/'
        perfpath = dirpath + mutation + '/perf/GAMMA_0_ALPHA_-1/'
        
        val_df = pd.read_csv(perfpath + 'n_token3_VAL_perf.csv')
        val_df['FOLD'] = f
        holdout_test_df = pd.read_csv(perfpath + 'n_token3_TEST_COMB_perf.csv')
        holdout_test_df['FOLD'] = f
        nep_df1 = pd.read_csv(perfpath + 'n_token3_TEST_Neptune_perf.csv')
        nep_df1['FOLD'] = f
        nep_df2 = pd.read_csv(perfpath + 'n_token3_TEST_OPX_perf.csv')
        nep_df2['FOLD'] = f
        nep_df3 = pd.read_csv(perfpath + 'n_token3_TEST_TCGA_PRAD_perf.csv')
        nep_df3['FOLD'] = f
        
        val_perf_dfs.append(val_df)
        holdout_test_perf_dfs.append(holdout_test_df)
        nep1_perf_dfs.append(nep_df1)
        nep2_perf_dfs.append(nep_df2)
        nep3_perf_dfs.append(nep_df3)
    
    val_perf_dfs= pd.concat(val_perf_dfs)
    holdout_test_perf_dfs= pd.concat(holdout_test_perf_dfs)
    nep1_perf_dfs= pd.concat(nep1_perf_dfs)
    nep2_perf_dfs= pd.concat(nep2_perf_dfs)    
    nep3_perf_dfs= pd.concat(nep3_perf_dfs)  
    
    val_perf_dfs = compute_meanstd_perf(val_perf_dfs, cohort_name = 'VAL')
    holdout_test_perf_dfs = compute_meanstd_perf(holdout_test_perf_dfs, cohort_name = 'HOLD_OUT_TEST_TCGA_OPX')
    nep1_perf_dfs = compute_meanstd_perf(nep1_perf_dfs, cohort_name = 'Neptune')
    nep2_perf_dfs = compute_meanstd_perf(nep2_perf_dfs, cohort_name = 'OPX')
    nep3_perf_dfs = compute_meanstd_perf(nep3_perf_dfs, cohort_name = 'TCGA_PRAD')
    
    final_df = pd.concat([val_perf_dfs,holdout_test_perf_dfs,nep1_perf_dfs,nep2_perf_dfs, nep3_perf_dfs])
    final_df = final_df[['COHORT','OUTCOME','AUC', 'Recall', 'PR_AUC', 'Specificity', 'ACC', 'Precision', 'F1', 'F2', 'F3']]
    final_df = final_df.loc[final_df['AUC'] != 'nan ± nan']

    mutation_perf_list.append(final_df)
    

all_final_df = pd.concat(mutation_perf_list)
all_final_df.to_csv(root_dir + "allfolds_perf.csv",index = False, encoding='utf-8-sig')



        
        
    
 