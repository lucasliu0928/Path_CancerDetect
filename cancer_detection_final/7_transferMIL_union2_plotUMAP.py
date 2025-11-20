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
        
        train_avg_features = torch.concat([item[0].mean(axis = 0).unsqueeze(0) for item in train_data], axis = 0)
        train_labels = torch.concat([item[1] for item in train_data]).squeeze() 
        
      
        tsne = TSNE(
                    n_components=2,      # 2D for plotting
                    perplexity=30,       # try 5–50; depends on dataset size
                    learning_rate=200,   # 10–1000; default usually fine
                    n_iter=1000,         # number of optimization iterations
                    random_state=42      # for reproducibility
                )
        
        X_tsne = tsne.fit_transform(train_avg_features)  # shape (n_samples, 2)
        
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=train_labels, s=15)
        plt.legend(*scatter.legend_elements(), title="Labels")
        plt.title("t-SNE Visualization")
        plt.show()
        
        
        import umap
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        
        # Standardize your data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(train_avg_features)
        
        # Run UMAP
        reducer = umap.UMAP(
            n_neighbors=5,   # controls local vs global structure
            min_dist=0.5,     # how tight the clusters are
            n_components=2,   # 2D for plotting
            random_state=42
        )
        
        X_umap = reducer.fit_transform(X_scaled)
        
        # Plot
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_umap[:,0], X_umap[:,1], c=train_labels, s=10)
        plt.legend(*scatter.legend_elements(), title="Labels")
        plt.title("UMAP Visualization")
        plt.show()
            
 