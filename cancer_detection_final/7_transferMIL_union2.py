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
import numpy as np
import wandb

sys.path.insert(0, '../Utils/')
from data_loader import load_dataset_splits, combine_all, just_test, uniform_sample_all_samples
from Loss import FocalLoss, compute_logit_adjustment, compute_label_freq
from misc_utils import str2bool
from misc_utils import create_dir_if_not_exists, set_seed
from plot_utils import plot_loss
from TransferMIL_utils import build_model, EarlyStopper, run_eval, train, validate
from data_loader import H5Cases, h5_to_list
from Eval import compute_performance


# source ~/.bashrc
# conda activate mil
# python3 -u 7_transferMIL_union2.py --train_tumor_frac 0.9 --fe_method virchow2
#   
#   

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--train_tumor_frac', default= 0.9, type=float, help='tile tumor fraction threshold for training')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath, virchow2')
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='MSI', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--logit_adj_train', default=False, type=str2bool, help='Train with logit adjustment')
parser.add_argument('--logit_adj_infer', default=True, type=str2bool, help='Train with logit adjustment')
parser.add_argument('--out_folder', default= 'pred_out_111625_union2', type=str, help='out folder name')
parser.add_argument('--model_name', default='Transfer_MIL', type=str, help='model name: e.g., Transfer_MIL, ABMIL')



############################################################################################################
#     Model Para
############################################################################################################
parser.add_argument('--DIM_OUT', default=512, type=int, help='')
parser.add_argument('--droprate', default=0.01, type=float, help='drop out rate')
parser.add_argument('--lr_base', default = 1e-4, type=float, help='learning rate') 
parser.add_argument('--lr_min', default = 1e-6, type=float, help='min learning rate') 
parser.add_argument('--lr_wstep', default = 5, type=int, help='warm up steps') 

parser.add_argument('--focal_alpha', default = 0.25, type=float, help='focal alpha') 
parser.add_argument('--focal_gamma', default = 2, type=int, help='focal gamma') 
parser.add_argument('--l2_coef', default = 5e-4, type=float, help='L2 Reg Coef') 
parser.add_argument('--train_epoch', default=100, type=int, help='')



            
if __name__ == '__main__':
    set_seed(42)
    args = parser.parse_args()
    fold_list = [0,1,2,3,4]
    #fold_list = [0]

    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    folder_name_100 = "IMSIZE250_OL100/TFT0.0"
    folder_name_0 = "IMSIZE250_OL0/TFT0.0"
    
    ###################################
    #Load data
    ###################################
    opx_union_ol100  = H5Cases(os.path.join(data_dir, "Union_OPX", folder_name_100 , f'union_opx_feature_{args.fe_method}_ol100.h5'))
    opx_union_ol0    = H5Cases(os.path.join(data_dir, "Union_OPX", folder_name_0 , f'union_opx_feature_{args.fe_method}_ol0.h5'))
    tcga_union_ol100 = H5Cases(os.path.join(data_dir, "Union_TCGA_PRAD", folder_name_100 , f'union_TCGA_PRAD_feature_{args.fe_method}_ol100.h5'))
    tcga_union_ol0 =  H5Cases(os.path.join(data_dir, "Union_TCGA_PRAD", folder_name_0 , f'union_TCGA_PRAD_feature_{args.fe_method}_ol0.h5'))

    nep_ol0 = H5Cases(os.path.join(data_dir, "Neptune", "IMSIZE250_OL0", f'feature_{args.fe_method}', "TFT0.0", 'Neptune_data.h5'))
        

    #conver h5cases to list
    opx_union_ol100 = h5_to_list(opx_union_ol100)
    opx_union_ol0 = h5_to_list(opx_union_ol0)
    tcga_union_ol100 = h5_to_list(tcga_union_ol100)
    tcga_union_ol0 = h5_to_list(tcga_union_ol0)
    nep_ol0 = h5_to_list(nep_ol0)


    
    ##########################################################################################
    #Combine TCGA and OPX
    ##########################################################################################
    #Combine
    comb_ol100 = opx_union_ol100 + tcga_union_ol100 
    comb_ol0   = opx_union_ol0 + tcga_union_ol0 

    
    ######################
    #Train
    ######################
    for f in fold_list:
        ######################
        #Create output-dir
        ######################
        outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                               args.mutation + '_traintf' + str(args.train_tumor_frac) + "_" + args.model_name + "_" + args.fe_method,
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
                                                tf_threshold = args.train_tumor_frac) 
        
        train_data, excluded_idx = (
            [x for x in train_data if len(x[0]) != 0],
            [i for i, x in enumerate(train_data) if len(x[0]) == 0]
        )  #exclude non-feature data after tf_threshold: #0.9: n = 421
        train_sp_ids = [x for i, x in enumerate(train_sp_ids) if i not in excluded_idx]
        
        
        
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import torch
        import torch.nn.functional as F
        
        train_avg_features = torch.concat([item[0].mean(axis = 0).unsqueeze(0) for item in train_data], axis = 0)
        train_labels = torch.concat([item[1] for item in train_data]).squeeze() 
        pos_indices = (train_labels == 1).nonzero(as_tuple=True)[0]
        X_pos = train_avg_features[pos_indices]
      
        # compute cosine similarity between every sample and every positive sample
        similarity_matrix = F.cosine_similarity(
            train_avg_features.unsqueeze(1),  # (N, 1, D)
            train_avg_features.unsqueeze(0),  # (1, N, D)
            dim=2            # -> (N, N)
        )
        
        # 1. Remove self similarity by setting diagonal to -inf
        sim = similarity_matrix.clone()
        sim.fill_diagonal_(-float('inf'))
        
        # 2. Get top-5 similar instance indices and scores for each sample
        top_k = 5
        values, indices = torch.topk(sim, k=top_k, dim=1)
        neighbor_indices = indices[pos_indices].reshape(-1)
        neighbor_indices = torch.unique(neighbor_indices)
        combined_indices = torch.unique(
                    torch.cat([pos_indices, neighbor_indices])
                )
        
        train_data_selected = [x for i, x in enumerate(train_data) if i in combined_indices]
        train_sp_ids_selected = [x for i, x in enumerate(train_sp_ids) if i in combined_indices]

        
        train_data = train_data_selected
        train_sp_ids = train_sp_ids_selected
        
        ####################################################
        #Select GPU
        ####################################################
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(device)
                
        
        #Feature and Label N
        N_FEATURE =  train_data[0][0].shape[1]
        N_LABELS  =  train_data[0][1].shape[1]
        
                
        # Start a new wandb run to track this script.
        run = wandb.init(
            # Set the wandb project where this run will be logged.
            project="MSI_PRED",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": args.lr_base,
                "architecture": args.model_name,
                "train_data": "opx_union_ol0 and TCGA_union_ol0",
                "focal_alpha": args.focal_alpha,
                "focal_gamma": args.focal_gamma,
                "epochs": args.train_epoch,
            },
        )
        
        

       
        # construct the model from src and load the state dict from HuggingFace 
        model = build_model(model_name = args.model_name, 
                    device = device, 
                    num_classes=2, 
                    n_feature = N_FEATURE)
        
        # Iterate and print all parameters
        for param in model.parameters():
            print(param)
        
        # #Freeze
        # # 1. First freeze everything
        # for p in model.parameters():
        #     p.requires_grad = False
        
        # # 2. Unfreeze ONLY the classifier
        # for p in model.model.classifier.parameters():
        #     p.requires_grad = True
            
            
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
       
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
        print("Trainable:", trainable)
        print("Frozen:", frozen)
    
            
        loss_fn = FocalLoss(alpha=args.focal_alpha, gamma= args.focal_gamma, reduction='mean')


        

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr= args.lr_base)
        
        # Scheduler 
        #Warm-up + Cosine LR
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=95, eta_min= args.lr_min)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
        
        #train loader
        train_loader = DataLoader(dataset=train_data,batch_size=1, shuffle=False)

        #early stopping
        early_stopper = EarlyStopper(patience=50, min_delta=1e-4)


        # Set the network to training mode
        #  Compute logit adjustment using train                    
        label_freq, label_freq_array = compute_label_freq(train_loader)
        logit_adjustments = compute_logit_adjustment(label_freq_array, tau = 0.5) #[-0.2093, -4.6176] The rarer class (1) gets a much more negative adjustment, which means during training its logits will be shifted down harder unless the model compensates.
          
    
        #  Compute logit adjustment using general prevelaence    ##############
        #For external dataset use univerisal MSI prevelence to adjust logit
        label_freq_array_ext = np.array([0.98,0.02])
        logit_adjustments_ext = compute_logit_adjustment(label_freq_array, tau = 0.1)
        
       
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
                                              logit_adjustments=logit_adjustments_ext, 
                                          l2_coef = args.l2_coef, pred_thres = best_th)
        #Combine all pred df , then compute performance
        all_pred_df = pd.concat([pred_df4,pred_df5, pred_df2])
        
        if args.logit_adj_infer:
            y_true, prob, pred = all_pred_df['True_y'], all_pred_df['adj_prob_1'], all_pred_df['Pred_Class_adj']
        else:
            y_true, prob, pred = all_pred_df['True_y'], all_pred_df['prob_1'], all_pred_df['Pred_Class']
        pref_tbf = compute_performance(y_true, prob, pred, "OPX_TCGA_TEST_and_NEP_ALL")    
        
        
        comb_perf = pd.concat([pref_tb, pref_tb_val, pref_tb4,pref_tb5, pref_tb2, pref_tbf])
        comb_perf.to_csv(os.path.join(outdir5, "before_finetune_performance.csv"))
        
    


        train_loss = []
        val_loss =[]
        for epoch in range(args.train_epoch):
            
            avg_loss  =  train(train_loader, model, device, loss_fn, optimizer, 
                                     model_name = args.model_name ,l2_coef = args.l2_coef, 
                                     logit_adjustments = logit_adjustments,
                                     logit_adj_train = args.logit_adj_train)
                      
            lr_scheduler.step()
            
            avg_loss_val, pred_df_val = validate(val_data, val_sp_ids, model, device, loss_fn, 
                                         logit_adjustments, 
                                         model_name = args.model_name,
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
            
            
            # Log metrics to wandb.
            run.log({"val_loss": avg_loss_val, "train_loss": avg_loss})

            #Save checkpoint
            torch.save(model.state_dict(), outdir1 + "FOLD" + str(f) + "_checkpoint" + str(epoch) + '.pth')
        
            # Early stop check
            if early_stopper.step(avg_loss_val.item(), model):
                print(f"Early stopping at epoch {epoch} (best val loss {early_stopper.best:.4f})")
                break
        
        # Finish the run and upload any remaining data.
        run.finish()
        plot_loss(train_loss, val_loss, outdir5)
        
