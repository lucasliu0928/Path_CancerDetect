#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 23:26:41 2025

@author: jliu6
"""

import os
import sys
import argparse
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, '../Utils/')
from data_loader import merge_data_lists, load_dataset_splits, filter_by_tumor_fraction
from data_loader import combine_all, just_test
from Loss import FocalLoss, compute_logit_adjustment, compute_label_freq
from misc_utils import str2bool, create_dir_if_not_exists
from TransferMIL_utils import build_model, run_eval
from Eval import bootstrap_ci_from_df_stratified, generate_attention_csv
from data_loader import H5Cases
from data_loader import get_train_test_valid_h5
from Eval import compute_performance


def get_final_perf(pred_data, cohort, logit_adj_infer = True):
                   
    if logit_adj_infer:
        y_true, prob, pred = pred_data['True_y'], pred_data['adj_prob_1'], pred_data['Pred_Class_adj']
    else:
        y_true, prob, pred = pred_data['True_y'], pred_data['prob_1'], pred_data['Pred_Class']
    perf_tb = compute_performance(y_true, prob, pred, cohort)
    perf_tb['best_thresh'] = pd.NA
    
    return perf_tb
        
# source ~/.bashrc
# conda activate mil
# python3 -u 7_transferMILunion2_test.py --tumor_frac 0.9 --model_name ABMIL --fe_method uni1
# python3 -u 7_transferMILunion2_test.py --tumor_frac 0.9 --model_name ABMIL --fe_method prov_gigapath
# python3 -u 7_transferMILunion2_test.py --tumor_frac 0.9 --model_name ABMIL --fe_method retccl

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Test")
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--logit_adj_infer', default=True, type=str2bool, help='Train with logit adjustment')
parser.add_argument('--model_name', default= 'MSI_traintf0.0_Transfer_MIL_uni2', type=str, help='model name')
parser.add_argument('--out_folder', default= 'pred_out_100125_union2', type=str, help='out folder name')


            
if __name__ == '__main__':
    
    args = parser.parse_args()
    fold_list = [0,1,2,3,4]
    model_list = ['MSI_traintf0.9_ABMIL_prov_gigapath',
                  'MSI_traintf0.9_ABMIL_retccl',
                  'MSI_traintf0.9_ABMIL_uni1',
                  'MSI_traintf0.9_ABMIL_uni2',
                  'MSI_traintf0.9_ABMIL_virchow2',
                  'MSI_traintf0.9_Transfer_MIL_uni2',
                  'MSI_traintf0.8_Transfer_MIL_uni2',
                  'MSI_traintf0.7_Transfer_MIL_uni2',
                  'MSI_traintf0.6_Transfer_MIL_uni2',
                  'MSI_traintf0.5_Transfer_MIL_uni2', 
                  'MSI_traintf0.4_Transfer_MIL_uni2',
                  'MSI_traintf0.3_Transfer_MIL_uni2',
                  'MSI_traintf0.2_Transfer_MIL_uni2',
                  'MSI_traintf0.1_Transfer_MIL_uni2',
                  'MSI_traintf0.0_Transfer_MIL_uni2']
    for model_name in model_list:
        
        args.model_name = model_name
        ##################
        ###### DIR  ######
        ##################
        proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
        folder = os.path.join(proj_dir, "intermediate_data", args.out_folder, args.model_name)
    
            
    
        for f in fold_list:
            infer_tf = 0.0
            pred_dir = os.path.join(folder, "FOLD" + str(f) ,"predictions_" + str(infer_tf))
            perf_dir = os.path.join(folder, "FOLD" + str(f) ,"perf_" + str(infer_tf))
            out_path = perf_dir
            create_dir_if_not_exists(out_path)
            
            #All cohrot test
            pred_df = pd.read_csv(os.path.join(pred_dir, "after_finetune_prediction.csv"))
            
        
            #OPX TEST
            cond = pred_df['SAMPLE_ID'].str.contains("OPX")
            pred_df_opx = pred_df[cond].copy()
            
            #TCGA TEST
            cond2 = pred_df['SAMPLE_ID'].str.contains("TCGA")
            pred_df_tcga = pred_df[cond2].copy()
            
            #NEP ALL
            cond3 = pred_df['SAMPLE_ID'].str.contains("NEP")
            pred_df_nep = pred_df[cond3].copy()
            
            #ALL TEST without NEP
            pred_df_opx_uw = pred_df[cond | cond2].copy()
            
            
            #Compute all cohort performance
            all_perf_tb = get_final_perf(pred_df, "OPX_TCGA_TEST_and_NEP_ALL", logit_adj_infer = args.logit_adj_infer)
            perf_tb_tcga_opx = get_final_perf(pred_df_opx_uw, "OPX_TCGA_TEST", logit_adj_infer = args.logit_adj_infer)
            perf_tb_opx = get_final_perf(pred_df_opx, "OPX_test", logit_adj_infer = args.logit_adj_infer)
            perf_tb_tcga = get_final_perf(pred_df_tcga, "TCGA_test", logit_adj_infer = args.logit_adj_infer)
            perf_tb_nep = get_final_perf(pred_df_nep, "NEP_ALL", logit_adj_infer = args.logit_adj_infer)
            #Load validation performance
            perf_val = pd.read_csv(os.path.join(perf_dir, "after_finetune_performance.csv"), index_col = 0)
            perf_val = perf_val.loc[perf_val.index == 'OPX_TCGA_valid']
            
            #Combine all cohort
            comb_perf = pd.concat([perf_val, perf_tb_opx, perf_tb_tcga, perf_tb_nep, perf_tb_tcga_opx, all_perf_tb])
    
            comb_perf.to_csv(os.path.join(out_path, "after_finetune_performance_Add_TCGAOPXonlyTest.csv"))
        

            
            
         
                
       