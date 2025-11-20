#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 23:26:41 2025

@author: jliu6
"""

import os
#os.chdir(os.path.dirname(__file__))
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
from Eval import get_final_perf, bootstrap_ci_from_df_stratified, generate_attention_csv
from data_loader import H5Cases, h5_to_list

# source ~/.bashrc
# conda activate mil
# python3 -u 7_transferMILunion2_test.py --tumor_frac 0.9 --model_name ABMIL --fe_method uni1
# python3 -u 7_transferMILunion2_test.py --tumor_frac 0.9 --model_name ABMIL --fe_method prov_gigapath
# python3 -u 7_transferMILunion2_test.py --tumor_frac 0.9 --model_name ABMIL --fe_method retccl

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Test")
parser.add_argument('--train_tumor_frac', default= 0.9, type=float, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath, virchow2')
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='MSI', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--logit_adj_train', default=False, type=str2bool, help='Train with logit adjustment')
parser.add_argument('--logit_adj_infer', default=True, type=str2bool, help='Train with logit adjustment')
parser.add_argument('--model_name', default='Transfer_MIL', type=str, help='model name: e.g., Transfer_MIL, ABMIL')
parser.add_argument('--out_folder', default= 'pred_out_100125_union2_check', type=str, help='out folder name')

            
if __name__ == '__main__':
    
    args = parser.parse_args()
    fold_list = [0,1,2,3,4]

    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    model_dir =  os.path.join(proj_dir, 'intermediate_data/',args.out_folder, 
                              args.mutation + "_traintf" + str(args.train_tumor_frac) + "_" + args.model_name + "_" + args.fe_method,
                              'locked_models/')
    folder_name_100 = "IMSIZE250_OL100/TFT0.0"
    folder_name_0 = "IMSIZE250_OL0/TFT0.0"
        
    ###################################
    #Load data
    ###################################
    opx_union_ol0    = H5Cases(os.path.join(data_dir, "Union_OPX", folder_name_0 , f'union_opx_feature_{args.fe_method}_ol0.h5'))
    tcga_union_ol0   = H5Cases(os.path.join(data_dir, "Union_TCGA_PRAD", folder_name_0 , f'union_TCGA_PRAD_feature_{args.fe_method}_ol0.h5'))
    nep_ol0 = H5Cases(os.path.join(data_dir, "Neptune", "IMSIZE250_OL0", f'feature_{args.fe_method}', "TFT0.0", 'Neptune_data.h5'))
        

    #conver h5cases to list
    opx_union_ol0 = h5_to_list(opx_union_ol0)
    tcga_union_ol0 = h5_to_list(tcga_union_ol0)
    nep_ol0 = h5_to_list(nep_ol0)
            
    #Combine test
    comb_ol0   = opx_union_ol0 + tcga_union_ol0 


    for f in fold_list:
        #tf_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        tf_list = [0.0]
        for tf in tf_list:
            ######################
            #Create output-dir
            ######################
            outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                                   args.mutation + '_traintf' + str(args.train_tumor_frac) + "_" + args.model_name + "_" + args.fe_method,
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
            tcga_split        =  load_dataset_splits(tcga_union_ol0, tcga_union_ol0, f, args.mutation, concat_tf = False)
            nep_st_split      =  load_dataset_splits(nep_ol0, nep_ol0, f, args.mutation, concat_tf = False)
            
            
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
            
            
            
            args.l2_coef = 5e-4
            model_name = args.model_name
            
           
           
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
            
            
            #logit_adjustments = logit_adjustments_ext
            
            ####################################################################################
            ### compute performance  for valdiation to get threshhold
            ####################################################################################
            avg_loss, pred_df_val, pref_tb_val =  run_eval(val_data, val_sp_ids, "OPX_TCGA_valid",     
                                                  loss_fn, model= model, device = device,logit_adj_infer = True, 
                                                  logit_adj_train = args.logit_adj_train, 
                                                  logit_adjustments=logit_adjustments, 
                                                  l2_coef = args.l2_coef)
            best_th = pref_tb_val['best_thresh'].item()            
            best_th_pr = pref_tb_val['best_thresh_prauc'].item()

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
            all_perf_tb = get_final_perf(all_pred_df, "OPX_TCGA_TEST_and_NEP_ALL", logit_adj_infer = args.logit_adj_infer)


            #Prediction:  all cohort
            comb_perd = pd.concat([pred_df_opx, pred_df_tcga, pred_df_nep])
            comb_perd.to_csv(os.path.join(outdir4, "after_finetune_prediction.csv"))
            
            #performance: Combine all cohort
            comb_perf = pd.concat([pref_tb_val, pref_tb_opx, pref_tb_tcga, pref_tb_nep, all_perf_tb])
            comb_perf.to_csv(os.path.join(outdir5, "after_finetune_performance.csv"))
            
            #Cperformance: ombine all cohort except Nepune
            pred_df_opx_tcga = pd.concat([pred_df_opx,pred_df_tcga])  
            pref_tb_opx_tcga = get_final_perf(pred_df_opx_tcga, "OPX_TCGA_TEST", logit_adj_infer = args.logit_adj_infer)
            comb_perf2 = pd.concat([pref_tb_val, pref_tb_opx, pref_tb_tcga, pref_tb_nep, pref_tb_opx_tcga, all_perf_tb])
            comb_perf2.to_csv(os.path.join(outdir5, "after_finetune_performance_Add_TCGAOPXonlyTest.csv"))
            

            
            
            #get attention:
            # generate_attention_csv(model, test_data_opx,   test_sp_ids_opx, mask_opx, opx_union_ol0, outdir6)
            # generate_attention_csv(model, test_data_tcga,  test_sp_ids_tcga, mask_tcga, tcga_union_ol0, outdir6)
            # generate_attention_csv(model, test_data_nep_st, test_sp_id_nep_st, mask_nep_st, nep_ol0_list, outdir6)

 