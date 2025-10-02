#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 23:26:41 2025

@author: jliu6
"""

from src.builder import create_model
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import copy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,fbeta_score,average_precision_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


sys.path.insert(0, '../Utils/')
from data_loader import merge_data_lists, load_dataset_splits
from data_loader import combine_all, just_test, downsample, uniform_sample_all_samples
from Loss import FocalLoss, compute_logit_adjustment
from misc_utils import str2bool
from misc_utils import create_dir_if_not_exists, set_seed
from plot_utils import plot_loss
from TransferMIL_utils import build_model, EarlyStopper, run_eval, train, validate
 
#FOR MIL-Lab
sys.path.insert(0, os.path.normpath(os.path.join(os.getcwd(), '..', '..', 'other_model_code','MIL-Lab',"src")))
from models.abmil import ABMILModel
from models.dsmil import DSMILModel
from models.transmil import TransMILModel

# source ~/.bashrc
# conda activate mil
############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--tumor_frac', default= 0.0, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath, virchow2')
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='MSI', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--logit_adj_train', default=True, type=str2bool, help='Train with logit adjustment')
parser.add_argument('--logit_adj_infer', default=True, type=str2bool, help='Train with logit adjustment')
parser.add_argument('--out_folder', default= 'pred_out_100125', type=str, help='out folder name')

############################################################################################################
#     Model Para
############################################################################################################
parser.add_argument('--DIM_OUT', default=512, type=int, help='')
parser.add_argument('--droprate', default=0.01, type=float, help='drop out rate')
parser.add_argument('--lr', default = 3e-4, type=float, help='learning rate') #0.01 works for DA with union , OPX + TCGA
parser.add_argument('--train_epoch', default=10, type=int, help='')

            
if __name__ == '__main__':
    
    args = parser.parse_args()
    fold_list = [0,1,2,3,4]
    #fold_list = [0]

    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    
    ###################################
    #Load
    ###################################
    cohorts = [
        "z_nostnorm_OPX",
        "z_nostnorm_TCGA_PRAD",
        "z_nostnorm_Neptune",
        "OPX",
        "TCGA_PRAD",
        "Neptune"
    ]
    
    data = {}
    
    for cohort_name in cohorts:
        start_time = time.time()
        base_path = os.path.join(data_dir, 
                                 cohort_name, 
                                 "IMSIZE250_{}", 
                                 f'feature_{args.fe_method}', 
                                 f"TFT{str(args.tumor_frac)}", 
                                 f'{cohort_name}_data.pth')
    
        data[f'{cohort_name}_ol100'] = torch.load(base_path.format("OL100"), weights_only = False)
        data[f'{cohort_name}_ol0'] = torch.load(base_path.format("OL0"),weights_only = False)
        
        elapsed_time = time.time() - start_time
        print(f"Time taken for {cohort_name}: {elapsed_time/60:.2f} minutes")
    
    opx_ol100_nst = data['z_nostnorm_OPX_ol100']
    opx_ol0_nst = data['z_nostnorm_OPX_ol0']
    tcga_ol100_nst = data['z_nostnorm_TCGA_PRAD_ol100']
    tcga_ol0_nst = data['z_nostnorm_TCGA_PRAD_ol0']
    nep_ol100_nst = data['z_nostnorm_Neptune_ol100']
    nep_ol0_nst = data['z_nostnorm_Neptune_ol0']
    
    opx_ol100 = data['OPX_ol100']
    opx_ol0 = data['OPX_ol0']
    tcga_ol100 = data['TCGA_PRAD_ol100']
    tcga_ol0 = data['TCGA_PRAD_ol0']
    nep_ol100 = data['Neptune_ol100']
    nep_ol0= data['Neptune_ol0']
    
        
 
    
    ##########################################################################################
    #Merge st norm and no st-norm
    ##########################################################################################
    opx_union_ol100  = merge_data_lists(opx_ol100_nst, opx_ol100, merge_type = 'union')
    opx_union_ol0    = merge_data_lists(opx_ol0_nst, opx_ol0, merge_type = 'union')
    tcga_union_ol100 = merge_data_lists(tcga_ol100_nst, tcga_ol100, merge_type = 'union')
    tcga_union_ol0   = merge_data_lists(tcga_ol0_nst, tcga_ol0, merge_type = 'union')
    nep_union_ol100  = merge_data_lists(nep_ol100_nst, nep_ol100, merge_type = 'union')
    nep_union_ol0    = merge_data_lists(nep_ol0_nst, nep_ol0, merge_type = 'union')
    
    #Combine
    comb_ol100 = opx_union_ol100 + tcga_union_ol100 
    comb_ol0   = opx_union_ol0 + tcga_union_ol0 

    for f in fold_list:
        ######################
        #Create output-dir
        ######################
        outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                               args.mutation,
                               'FOLD' + str(f))
        outdir1 =  outdir0  + "/saved_model/"
        outdir2 =  outdir0  + "/model_para/"
        outdir3 =  outdir0  + "/logs/"
        outdir4 =  outdir0  + "/predictions/"
        outdir5 =  outdir0  + "/perf/"
        outdir6 =  outdir0  + "/predictions/attention/"
        
        outdir_list = [outdir0,outdir1,outdir2,outdir3,outdir4,outdir5, outdir6]
        
        for out_path in outdir_list:
            create_dir_if_not_exists(out_path)
            
        ####################################
        #Load data
        ####################################    
        #get train test and valid
        opx_split    =  load_dataset_splits(opx_union_ol0, opx_union_ol0, f, args.mutation, concat_tf = False)
        tcga_split   =  load_dataset_splits(tcga_union_ol0, tcga_union_ol0, f, args.mutation, concat_tf = False)
        nep_split    =  load_dataset_splits(nep_ol0, nep_ol0, f, args.mutation, concat_tf = False)
        nep_nst_split    =  load_dataset_splits(nep_ol0_nst, nep_ol0_nst, f, args.mutation, concat_tf = False)
        comb_splits  =  load_dataset_splits(comb_ol0, comb_ol0, f, args.mutation, concat_tf = False)
 

        train_data, train_sp_ids, train_pt_ids, train_cohorts, train_coords  = comb_splits['train']
        test_data, test_sp_ids, test_pt_ids, test_cohorts, _ = comb_splits['test']
        val_data, val_sp_ids, val_pt_ids, val_cohorts,_ = comb_splits['val']
        

        
        # OPX all
        test_data1, test_sp_ids1, test_pt_ids1, test_cohorts1 = combine_all(opx_split)

        # NEP all
        test_data2, test_sp_ids2, test_pt_ids2, test_cohorts2 = combine_all(nep_split)

        # TCGA all
        test_data3, test_sp_ids3, test_pt_ids3, test_cohorts3 = combine_all(tcga_split)

        # OPX / TCGA / NEP test only
        test_data4, test_sp_ids4, test_pt_ids4, test_cohorts4 = just_test(opx_split)
        test_data5, test_sp_ids5, test_pt_ids5, test_cohorts5 = just_test(tcga_split)
        test_data6, test_sp_ids6, test_pt_ids6, test_cohorts6 = just_test(nep_split)
        #test_data7, test_sp_ids7, test_pt_ids7, test_cohorts7 = just_test(nep_nst_split)

        
        #Final combined performance
        final_test_data = test_data2 + test_data4 + test_data5
        final_test_sp_ids = test_sp_ids2 + test_sp_ids4 + test_sp_ids5


        #samplling, sample could has less than 400, if original tile is <400
        train_data = uniform_sample_all_samples(train_data, train_coords, max_bag = 2000, 
                                                grid = 32, sample_by_tf = True, plot = False,
                                                tf_threshold = 0.9) 
        
        train_data, excluded_idx = (
            [x for x in train_data if len(x[0]) != 0],
            [i for i, x in enumerate(train_data) if len(x[0]) == 0]
        )  #exclude non-feature data after tf_threshold: #0.9: n = 421
        train_sp_ids = [x for i, x in enumerate(train_sp_ids) if i not in excluded_idx]


        ####################################################
        #Select GPU
        ####################################################
        args.cuda_device = 0
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(device)
                
        
        #Feature and Label N
        N_FEATURE =  train_data[0][0].shape[1]
        N_LABELS  =  train_data[0][1].shape[1]
        
        
        
        args.lr = 1e-4
        args.logit_adj_train = False
        args.l2_coef = 5e-4
        model_name = "Transfer_MIL"
        
       
       
        # construct the model from src and load the state dict from HuggingFace 
        model_indexs_map = {0: 99, 1:66, 2:51, 3:65, 4: 99} 
        mode_idxes = model_indexs_map[f]        
        checkpoint = torch.load(os.path.join(outdir1,'checkpoint'+ str(mode_idxes) + '.pth'))
        
        model = build_model(model_name = model_name, 
                    device = device, 
                    num_classes=2, 
                    n_feature = N_FEATURE)
        
        model.load_state_dict(checkpoint)
        
            
        #loss_fn = FocalLoss(alpha=-1, gamma=0, reduction='mean')
        loss_fn = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
 

        #val logit adj
        val_loader = DataLoader(dataset=val_data,batch_size=1, shuffle=False)
        logit_adjustments, label_freq = compute_logit_adjustment(val_loader, tau = 0.5)
        avg_loss, pred_df_val, pref_tb_val = run_eval(val_data, val_sp_ids, "OPX_TCGA_valid",     
                                              loss_fn, model= model, device = device,logit_adj_infer = True, 
                                              logit_adj_train = args.logit_adj_train, 
                                              logit_adjustments=logit_adjustments, 
                                              l2_coef = args.l2_coef)
        
        best_th = pref_tb_val['best_thresh'].item()
        avg_loss, pred_df4, pref_tb4 = run_eval(test_data4, test_sp_ids4, "OPX_test",     
                                              loss_fn, model= model, device = device,logit_adj_infer = True, 
                                              logit_adj_train = args.logit_adj_train, 
                                              logit_adjustments=logit_adjustments, 
                                              l2_coef = args.l2_coef, pred_thres = best_th)
        avg_loss, pred_df5, pref_tb5 = run_eval(test_data5, test_sp_ids5, "TCGA_test",    
                                              loss_fn, model= model, device = device,logit_adj_infer = True, 
                                              logit_adj_train = args.logit_adj_train, 
                                              logit_adjustments=logit_adjustments, 
                                              l2_coef = args.l2_coef, pred_thres = best_th)
        avg_loss, pred_df2, pref_tb2 = run_eval(test_data2, test_sp_ids2, "NEP_ALL",
                                              loss_fn, model= model, device = device,logit_adj_infer = True, 
                                              logit_adj_train = args.logit_adj_train, 
                                              logit_adjustments=logit_adjustments, 
                                              l2_coef = args.l2_coef, pred_thres = best_th)
        avg_loss, pred_dff, pref_tbf = run_eval(final_test_data, final_test_sp_ids, "OPX_TCGA_TEST_and_NEP_ALL",
                                              loss_fn, model= model, device = device,logit_adj_infer = True, 
                                              logit_adj_train = args.logit_adj_train, 
                                              logit_adjustments=logit_adjustments, 
                                              l2_coef = args.l2_coef, pred_thres = best_th)
        comb_perf = pd.concat([pref_tb_val, pref_tb4,pref_tb5, pref_tb2, pref_tbf])
        comb_perf.to_csv(os.path.join(outdir5, "after_finetune_performance.csv"))
        
        
        pred_dff.to_csv(os.path.join(outdir4, "after_finetune_prediction.csv"))
        
        
        #bootstrap perforance
        from Eval import bootstrap_ci_from_df, bootstrap_ci_from_df_stratified
        logit_adj_infer = True
        
        pred_list = [pred_df4, pred_df5, pred_df2, pred_dff]
        cohort_list = ['OPX_test', 'TCGA_test','NEP_ALL',"OPX_TCGA_TEST_and_NEP_ALL"]
        
        ci_df_list = []
        for cur_pred_df, name in zip(pred_list,cohort_list):
            
            if logit_adj_infer:
                cur_ci_df = bootstrap_ci_from_df_stratified(cur_pred_df, y_true_col='True_y', y_pred_col='Pred_Class_adj', y_prob_col='adj_prob_1', num_bootstrap=1000, ci=95, seed=42, cohort_name= name, stratified=True)
            else:
                cur_ci_df = bootstrap_ci_from_df_stratified(cur_pred_df, y_true_col='True_y', y_pred_col='Pred_Class', y_prob_col='prob_1', num_bootstrap=1000, ci=95, seed=42, cohort_name= name, stratified=True)
            
            
            #compute the numer of postive
            True_y = cur_pred_df['True_y'] == 1
            n_positive = True_y.sum()
            n_total = len(True_y)
            prop_positive = n_positive / n_total*100
            cur_ci_df['N POS (%)'] = f'{n_positive} ({prop_positive: .2f}%)'
            
            cur_ci_df['cohort'] = name

            ci_df_list.append(cur_ci_df)
        final_ci_df = pd.concat(ci_df_list)
        
        final_ci_df.to_csv(outdir5 + "after_finetune_performance_bootstraping_alltiles.csv")
        

        
        #get attention:
        for (i, parts) in enumerate(zip(test_data4, test_sp_ids4)):
            x, y, tf, sl = parts[0]
            x = x.unsqueeze(0).to(device)      # add batch dim
            y = y.long().view(-1).to(device)
            sp_id = parts[1]
            
            #Run model     
            results, log_dict = model(x,return_attention=True,
                               return_slide_feats=True)
            attention = log_dict['attention']
            s_feature = log_dict['slide_feats']
            
            tile_info = [d['tile_info'] for d in opx_union_ol0 if d['sample_id'] == sp_id][0]
            tile_info['att'] = attention.squeeze().detach().cpu().numpy()
            plot_df = tile_info[['pred_map_location',"att"]]
            plot_df.to_csv(os.path.join(outdir6, sp_id + "_att.csv"))
        
        #TCGA
        for (i, parts) in enumerate(zip(test_data2, test_sp_ids2)):
            x, y, tf, sl = parts[0]
            x = x.unsqueeze(0).to(device)      # add batch dim
            y = y.long().view(-1).to(device)
            sp_id = parts[1]
            
            #Run model     
            results, log_dict = model(x,return_attention=True,
                               return_slide_feats=True)
            attention = log_dict['attention']
            s_feature = log_dict['slide_feats']
            
            tile_info = [d['tile_info'] for d in tcga_union_ol0 if d['sample_id'] == sp_id][0]
            tile_info['att'] = attention.squeeze().detach().cpu().numpy()
            plot_df = tile_info[['pred_map_location',"att"]]
            plot_df.to_csv(os.path.join(outdir6, sp_id + "_att.csv"))
            
            
        #nep
        for (i, parts) in enumerate(zip(test_data5, test_sp_ids5)):
            x, y, tf, sl = parts[0]
            x = x.unsqueeze(0).to(device)      # add batch dim
            y = y.long().view(-1).to(device)
            sp_id = parts[1]
            
            #Run model     
            results, log_dict = model(x,return_attention=True,
                               return_slide_feats=True)
            attention = log_dict['attention']
            s_feature = log_dict['slide_feats']
            
            tile_info = [d['tile_info'] for d in nep_ol0 if d['sample_id'] == sp_id][0]
            tile_info['att'] = attention.squeeze().detach().cpu().numpy()
            plot_df = tile_info[['pred_map_location',"att"]]
            plot_df.to_csv(os.path.join(outdir6, sp_id + "_att.csv"))
            
            
            
            
            
            
            
     
            
   