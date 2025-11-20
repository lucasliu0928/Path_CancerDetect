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
from torch.utils.data import DataLoader

sys.path.insert(0, '../Utils/')
from data_loader import merge_data_lists, load_dataset_splits
from data_loader import combine_all, just_test, uniform_sample_all_samples
from Loss import FocalLoss, compute_logit_adjustment
from misc_utils import str2bool
from misc_utils import create_dir_if_not_exists, set_seed
from plot_utils import plot_loss
from TransferMIL_utils import build_model, EarlyStopper, run_eval, train, validate

 
# #FOR MIL-Lab
# sys.path.insert(0, os.path.normpath(os.path.join(os.getcwd(), '..', '..', 'other_model_code','MIL-Lab',"src")))
# from models.abmil import ABMILModel
# from models.dsmil import DSMILModel
# from models.transmil import TransMILModel


# source ~/.bashrc
# conda activate mil
# python3 -u 7_transferMIL.py \
#   --mutation ${MUT} \
#   --tumor_frac 0.0 \
#   --train_epoch 100

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--tumor_frac', default= 0.0, type=float, help='tile tumor fraction threshold')
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
parser.add_argument('--train_epoch', default=100, type=int, help='')

            
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
                                 "TFT0.0", 
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
        
        #Final combined performance
        final_test_data = test_data2 + test_data4 + test_data5
        final_test_sp_ids = test_sp_ids2 + test_sp_ids4 + test_sp_ids5


        #samplling, sample could has less than 400, if original tile is <400
        train_data = uniform_sample_all_samples(train_data, train_coords, max_bag = 2000, 
                                                grid = 32, sample_by_tf = True, plot = False,
                                                tf_threshold = args.tumor_frac) 
        
        train_data, excluded_idx = (
            [x for x in train_data if len(x[0]) != 0],
            [i for i, x in enumerate(train_data) if len(x[0]) == 0]
        )  #exclude non-feature data after tf_threshold: #0.9: n = 421
        train_sp_ids = [x for i, x in enumerate(train_sp_ids) if i not in excluded_idx]


        ####################################################
        #Select GPU
        ####################################################
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(device)
                
        
        #Feature and Label N
        N_FEATURE =  train_data[0][0].shape[1]
        N_LABELS  =  train_data[0][1].shape[1]
        
        
        #ABMILModel(in_dim=1024, num_classes=2, config = None)
        
        args.lr = 1e-4
        args.logit_adj_train = False
        args.l2_coef = 5e-4
        model_name = "Transfer_MIL"
        
       
        
        # construct the model from src and load the state dict from HuggingFace 
        model = build_model(model_name = model_name, 
                    device = device, 
                    num_classes=2, 
                    n_feature = N_FEATURE)
        
        # Iterate and print all parameters
        for param in model.parameters():
            print(param)
            
        #loss_fn = FocalLoss(alpha=-1, gamma=0, reduction='mean')
        loss_fn = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
 

        
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, #3e-4,
        )
        
        
        # Scheduler 
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
        
        #train loader
        train_loader = DataLoader(dataset=train_data,batch_size=1, shuffle=False)


        # Set the network to training mode
        logit_adjustments, label_freq = compute_logit_adjustment(train_loader, tau = 0.5) #[-0.2093, -4.6176] The rarer class (1) gets a much more negative adjustment, which means during training its logits will be shifted down harder unless the model compensates.

        early_stopper = EarlyStopper(patience=10, min_delta=1e-4)  
        
       
        avg_loss, pred_df, pref_tb  = run_eval(train_data, train_sp_ids, "TRAIN", loss_fn, model= model, device = device,logit_adj_infer = True, logit_adj_train = args.logit_adj_train, logit_adjustments=logit_adjustments, l2_coef = args.l2_coef )
        best_th = pref_tb['best_thresh'].item()
        avg_loss, pred_df_val, pref_tb_val = run_eval(val_data, val_sp_ids, "OPX_TCGA_valid",     
                                              loss_fn, model= model, device = device,logit_adj_infer = True, 
                                              logit_adj_train = args.logit_adj_train, 
                                              logit_adjustments=logit_adjustments, 
                                              l2_coef = args.l2_coef, pred_thres = best_th)
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
        comb_perf = pd.concat([pref_tb, pref_tb_val, pref_tb4,pref_tb5, pref_tb2, pref_tbf])
        comb_perf.to_csv(os.path.join(outdir5, "before_finetune_performance.csv"))
        
    


        train_loss = []
        val_loss =[]
        for epoch in range(args.train_epoch):
            
            avg_loss  =  train(train_loader, model, device, loss_fn, optimizer, 
                                     model_name = 'Transfer_MIL' ,l2_coef = args.l2_coef, 
                                     logit_adjustments = logit_adjustments,
                                     logit_adj_train = args.logit_adj_train)
                      
            lr_scheduler.step()
            
            avg_loss_val, pred_df_val = validate(val_data, val_sp_ids, model, device, loss_fn, 
                                         logit_adjustments, 
                                         model_name = 'Transfer_MIL',
                                         logit_adj_infer = args.logit_adj_infer,
                                         logit_adj_train = args.logit_adj_train,
                                         l2_coef = args.l2_coef,
                                         pred_thres = 0.5)
            
            # Manual logging
            train_loss.append(avg_loss.item())
            val_loss.append(avg_loss_val.item())
            
            log_items = [
                f"EPOCH: {epoch}",
                f"lr: {optimizer.param_groups[0]['lr']:.8f}",
                f"train loss: {avg_loss.item():.4f}",
                f"val loss: {avg_loss_val.item():.4f}",
                f"best val: {early_stopper.best:.4f}",
            ]
            print(" | ".join(log_items))
            
            #Save checkpoint
            torch.save(model.state_dict(), outdir1 + "checkpoint" + str(epoch) + '.pth')
        
            # Early stop check
            if early_stopper.step(avg_loss_val.item(), model):
                print(f"Early stopping at epoch {epoch} (best val loss {early_stopper.best:.4f})")
                break
        
        plot_loss(train_loss, val_loss, outdir5)
        
