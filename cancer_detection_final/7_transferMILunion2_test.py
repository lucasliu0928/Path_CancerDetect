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


# source ~/.bashrc
# conda activate mil
############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Test")
parser.add_argument('--tumor_frac', default= 0.0, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath, virchow2')
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='MSI', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--logit_adj_train', default=False, type=str2bool, help='Train with logit adjustment')
parser.add_argument('--logit_adj_infer', default=True, type=str2bool, help='Train with logit adjustment')
parser.add_argument('--out_folder', default= 'pred_out_100125_union2', type=str, help='out folder name')
parser.add_argument('--model_name', default='Transfer_MIL', type=str, help='model name: e.g., Transfer_MIL, ABMIL')

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

    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    model_dir =  os.path.join(proj_dir, 'intermediate_data/',args.out_folder, 
                              args.mutation + "_traintf" + str(args.tumor_frac) + "_" + args.model_name,
                              'locked_models/')
        
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
    
    #updates: load h5
    for cohort_name in cohorts:
        start_time = time.time()
        base_path = os.path.join(data_dir, 
                                 cohort_name, 
                                 "IMSIZE250_{}", 
                                 f'feature_{args.fe_method}', 
                                 "TFT0.0", 
                                 f'{cohort_name}_data.h5')
    
        data[f'{cohort_name}_ol100'] = H5Cases(os.path.join(base_path.format("OL100")))
        data[f'{cohort_name}_ol0']   = H5Cases(os.path.join(base_path.format("OL0")))
        
        elapsed_time = time.time() - start_time
        print(f"Time taken for {cohort_name}: {elapsed_time/60:.2f} minutes")
        
   
    ##########################################################################################
    #Non-st norm 
    ##########################################################################################
    opx_ol100_nst = data['z_nostnorm_OPX_ol100']
    opx_ol0_nst = data['z_nostnorm_OPX_ol0']
    tcga_ol100_nst = data['z_nostnorm_TCGA_PRAD_ol100']
    tcga_ol0_nst = data['z_nostnorm_TCGA_PRAD_ol0']
    nep_ol100_nst = data['z_nostnorm_Neptune_ol100']
    nep_ol0_nst = data['z_nostnorm_Neptune_ol0']
    
    ##########################################################################################
    #ST-norm
    ##########################################################################################
    opx_ol100 = data['OPX_ol100']
    opx_ol0 = data['OPX_ol0']
    tcga_ol100 = data['TCGA_PRAD_ol100']
    tcga_ol0 = data['TCGA_PRAD_ol0']
    nep_ol100 = data['Neptune_ol100']
    nep_ol0= data['Neptune_ol0']
    
    
    
    
    def h5_to_list(ds):
        return [ds[i] for i in range(len(ds))]
    
    
    # Convert all H5Cases objects into lists
    opx_ol0_nst_list   = h5_to_list(opx_ol0_nst)
    tcga_ol0_nst_list   = h5_to_list(tcga_ol0_nst)
    
    opx_ol0_list   = h5_to_list(opx_ol0)
    tcga_ol0_list   = h5_to_list(tcga_ol0)
    nep_ol0_list    = h5_to_list(nep_ol0)
    
    
    ##########################################################################################
    #Merge st norm and no st-norm
    ##########################################################################################
    opx_union_ol0    = merge_data_lists(opx_ol0_nst_list, opx_ol0_list, merge_type = 'union')
    tcga_union_ol0   = merge_data_lists(tcga_ol0_nst_list, tcga_ol0_list, merge_type = 'union')
    
    #Combine
    comb_ol0   = opx_union_ol0 + tcga_union_ol0 


    for f in fold_list:
        tf_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        tf_list = [0.0]
        for tf in tf_list:
            ######################
            #Create output-dir
            ######################
            outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                                   args.mutation + '_traintf' + str(args.tumor_frac) + "_" + args.model_name,
                                   'FOLD' + str(f))
            outdir4 =  outdir0  + "/predictions_" + str(tf) + "/"
            outdir5 =  outdir0  + "/perf_" + str(tf) + "/"
            outdir6 =  outdir4  + "attention/"
            
            outdir_list = [outdir0,outdir4,outdir5, outdir6]
            
            for out_path in outdir_list:
                create_dir_if_not_exists(out_path)
                
            
            ####################################
            #Load data
            ####################################
            #Get Validation data
            comb_splits  =  load_dataset_splits(comb_ol0, comb_ol0, f, args.mutation, concat_tf = False)
            val_data, val_sp_ids, val_pt_ids, val_cohorts, val_coord = comb_splits['val']
            

            
            #get train test and valid split
            opx_split        =  load_dataset_splits(opx_union_ol0, opx_union_ol0, f, args.mutation, concat_tf = False)
            opx_st_split     =  load_dataset_splits(opx_ol0_list, opx_ol0_list, f, args.mutation, concat_tf = False)
            opx_nst_split    =  load_dataset_splits(opx_ol0_nst_list, opx_ol0_nst_list, f, args.mutation, concat_tf = False)

            
            tcga_split        =  load_dataset_splits(tcga_union_ol0, tcga_union_ol0, f, args.mutation, concat_tf = False)
            tcga_st_split     =  load_dataset_splits(tcga_ol0_list, tcga_ol0_list, f, args.mutation, concat_tf = False)
            tcga_nst_split    =  load_dataset_splits(tcga_ol0_nst_list, tcga_ol0_nst_list, f, args.mutation, concat_tf = False)
            
           
            nep_st_split      =  load_dataset_splits(nep_ol0_list, nep_ol0_list, f, args.mutation, concat_tf = False)
            
            
            # NEP combine train, test and valid in to one test
            test_data_nep_st, test_sp_id_nep_st, *_ = combine_all(nep_st_split)

            # OPX test only
            test_data_opx,  test_sp_ids_opx, *_ = just_test(opx_split)
            
            #TCGA test only
            test_data_tcga, test_sp_ids_tcga, *_ = just_test(tcga_split)
    

            ###filtered inference threahold 0.9
            test_data_nep_st, mask_nep_st = filter_by_tumor_fraction(test_data_nep_st, threshold = tf)
            test_data_opx, mask_opx = filter_by_tumor_fraction(test_data_opx, threshold = tf)
            test_data_tcga, mask_tcga = filter_by_tumor_fraction(test_data_tcga, threshold = tf)
            val_data, mask_val = filter_by_tumor_fraction(val_data, threshold = tf)
                        
            ####################################################
            #Select GPU
            ####################################################
            args.cuda_device = 0
            device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
            print(device)
                    
            
            #Feature and Label N
            N_FEATURE =  val_data[0][0].shape[1]
            N_LABELS  =  val_data[0][1].shape[1]
            
            
            
            args.lr = 1e-4
            args.l2_coef = 5e-4
            model_name = "Transfer_MIL"
            
           
           
            # construct the model from src and load the state dict from HuggingFace 
            # model_indexs_map = {0: 99, 1:66, 2:51, 3:65, 4: 99} 
            # mode_idxes = model_indexs_map[f]        
            # checkpoint = torch.load(os.path.join(outdir1,'checkpoint'+ str(mode_idxes) + '.pth'))            
            checkpoint_file = [file for file in os.listdir(model_dir) if "FOLD" + str(f) in file][0]
            checkpoint = torch.load(os.path.join(model_dir,checkpoint_file))
        
            
            model = build_model(model_name = model_name, 
                        device = device, 
                        num_classes=2, 
                        n_feature = N_FEATURE)
            
            model.load_state_dict(checkpoint)
            
                
            loss_fn = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
     
            ####################################################################################
            ##############  Compute logit adjustment using VAL                    ##############
            ####################################################################################
            val_loader = DataLoader(dataset=val_data,batch_size=1, shuffle=False)            
            label_freq, label_freq_array = compute_label_freq(val_loader)
            logit_adjustments = compute_logit_adjustment(label_freq_array, tau = 0.5) #[-0.2093, -4.6176] The rarer class (1) gets a much more negative adjustment, which means during training its logits will be shifted down harder unless the model compensates.


            ####################################################################################
            ##############  Compute logit adjustment using general prevelaence    ##############
            ####################################################################################
            #For external dataset use univerisal MSI prevelence to adjust logit
            label_freq_array_ext = np.array([0.98,0.02])
            logit_adjustments_ext = compute_logit_adjustment(label_freq_array, tau = 0.1)
            
            ####################################################################################
            ### compute performance  for valdiation to get threshhold
            ####################################################################################
            avg_loss, pred_df_val, pref_tb_val = run_eval(val_data, val_sp_ids, "OPX_TCGA_valid",     
                                                  loss_fn, model= model, device = device,logit_adj_infer = True, 
                                                  logit_adj_train = args.logit_adj_train, 
                                                  logit_adjustments=logit_adjustments, 
                                                  l2_coef = args.l2_coef)
            best_th = pref_tb_val['best_thresh'].item()
            
            
            
            ####################################################################################
            #Performance 
            ####################################################################################
            #Validation
            avg_loss, pred_df_val, pref_tb_val = run_eval(val_data, val_sp_ids, "OPX_TCGA_valid",     
                                                  loss_fn, model= model, device = device,logit_adj_infer = True, 
                                                  logit_adj_train = args.logit_adj_train, 
                                                  logit_adjustments=logit_adjustments, 
                                                  l2_coef = args.l2_coef,
                                                  pred_thres = best_th) #updated validation performance by using its own best thres
            
            #OPX
            avg_loss, pred_df_opx, pref_tb_opx = run_eval(test_data_opx, test_sp_ids_opx, "OPX_test",     
                                                  loss_fn, model= model, device = device,logit_adj_infer = True, 
                                                  logit_adj_train = args.logit_adj_train, 
                                                  logit_adjustments=logit_adjustments, 
                                                  l2_coef = args.l2_coef, pred_thres = best_th)
            pref_tb_opx['best_thresh'] = pd.NA

            #tcga
            avg_loss, pred_df_tcga, pref_tb_tcga = run_eval(test_data_tcga, test_sp_ids_tcga, "TCGA_test",    
                                                  loss_fn, model= model, device = device,logit_adj_infer = True, 
                                                  logit_adj_train = args.logit_adj_train, 
                                                  logit_adjustments=logit_adjustments, 
                                                  l2_coef = args.l2_coef, pred_thres = best_th)
            pref_tb_tcga['best_thresh'] = pd.NA

            #NEP
            label_freq_array_ext = np.array([0.98,0.02])
            logit_adjustments_ext = compute_logit_adjustment(label_freq_array_ext, tau = 0.1)
            avg_loss, pred_df_nep, pref_tb_nep = run_eval(test_data_nep_st, test_sp_id_nep_st, "NEP_ALL",
                                                  loss_fn, model= model, device = device,logit_adj_infer = True, 
                                                  logit_adj_train = args.logit_adj_train, 
                                                  logit_adjustments=logit_adjustments_ext, 
                                                  l2_coef = args.l2_coef, pred_thres = best_th)
            pref_tb_nep['best_thresh'] = pd.NA
 
            #Combine all pred df , then compute performance
            all_pred_df = pd.concat([pred_df_opx,pred_df_tcga, pred_df_nep])
            from Eval import compute_performance
            if args.logit_adj_infer:
                y_true, prob, pred = all_pred_df['True_y'], all_pred_df['adj_prob_1'], all_pred_df['Pred_Class_adj']
            else:
                y_true, prob, pred = all_pred_df['True_y'], all_pred_df['prob_1'], all_pred_df['Pred_Class']
            all_perf_tb = compute_performance(y_true, prob, pred, "OPX_TCGA_TEST_and_NEP_ALL")
            all_perf_tb['best_thresh'] = pd.NA
            
            #Combine all cohort
            comb_perf = pd.concat([pref_tb_val, pref_tb_opx, pref_tb_tcga, pref_tb_nep, all_perf_tb])
            comb_perf.to_csv(os.path.join(outdir5, "after_finetune_performance.csv"))
            
            #remove all_pred_df later
            comb_perd = pd.concat([pred_df_opx, pred_df_tcga, pred_df_nep])
            comb_perd.to_csv(os.path.join(outdir4, "after_finetune_prediction.csv"))
            
            
            #get attention:
            #get attention:
            # generate_attention_csv(model, test_data_opx,   test_sp_ids_opx, mask_opx, opx_union_ol0, outdir6)
            # generate_attention_csv(model, test_data_tcga,  test_sp_ids_tcga, mask_tcga, tcga_union_ol0, outdir6)
            # generate_attention_csv(model, test_data_nep_st, test_sp_id_nep_st, mask_nep_st, nep_ol0_list, outdir6)
    
    
                
                
                
                
                
                
         
                
       