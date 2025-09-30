#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 20:11:34 2025

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
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


sys.path.insert(0, '../Utils/')
from data_loader import merge_data_lists, load_dataset_splits
from data_loader import combine_all, just_test, downsample, uniform_sample_all_samples
from Loss import FocalLoss, compute_logit_adjustment
from TransMIL import TransMIL
from misc_utils import str2bool
from misc_utils import create_dir_if_not_exists, set_seed
from Embeddings import extract_features, get_slide_embedding, plot_embeddings
from Embeddings import get_feature_label_site
 
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
            
if __name__ == '__main__':
    
    args = parser.parse_args()


    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    
    ######################
    #Create output-dir
    ######################
    outdir = os.path.join(proj_dir, 
                          "intermediate_data",
                          "7A_embeddings")
    create_dir_if_not_exists(outdir)
            
    
    
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
    
    
    #Combine
    comb_ol100 = opx_ol100 + tcga_ol100 + nep_ol100
    comb_ol0   = opx_ol0 + tcga_ol0 + nep_ol0

    
    #compute avarege emebdding
    opx_feature, opx_label, opx_site, opx_ids, opx_ids_ignored =  get_feature_label_site(opx_ol0, tf_threshold = 0.9, agg_method = 'mean')
    nep_feature, nep_label, nep_site, nep_ids, nep_ids_ignored =  get_feature_label_site(nep_ol0, tf_threshold = 0.9, agg_method = 'mean')
    tcga_feature, tcga_label, tcga_site, tcga_ids, tcga_ids_ignored =  get_feature_label_site(tcga_ol0, tf_threshold = 0.9, agg_method = 'mean')
    all_feature, all_label, all_site, all_ids, all_ids_ignored =  get_feature_label_site(comb_ol0, tf_threshold = 0.9, agg_method = 'mean')

    #cohort_map
    cohort_map = {}
    for _id in opx_ids:
        cohort_map[str(_id)] = "OPX"
    for _id in nep_ids:
        cohort_map[str(_id)] = "NEP"
    for _id in tcga_ids:
        cohort_map[str(_id)] = "TCGA"
    # 2) Make a cohort array aligned to all_ids
    cohort = np.array([cohort_map.get(str(_id), "Unknown") for _id in all_ids])
            
        
    #UMAP
    all_labels = ["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI"]
    label_idx = all_labels.index("HR2") 
    plot_embeddings(all_feature, all_site, method="umap", cohort = cohort)

   
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    
    def plot_embeddings(all_feats, all_labels, method="pca", cohort=None, 
                        cohort_order=None, cohort_markers=None, **kwargs):
        """
        all_feats: (N, D)
        all_labels: (N,) numeric/int/str labels
        cohort: optional (N,) array with cohort names (e.g., 'OPX','NEP','TCGA')
        cohort_order: optional list to control plotting order of cohorts
        cohort_markers: optional dict mapping cohort -> marker (e.g., {'OPX':'o','NEP':'s','TCGA':'^'})
        """
    
        # ---- Reduce to 2D ----
        method = method.lower()
        if method == "pca":
            reducer = PCA(n_components=2, **kwargs)
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, **kwargs)
        elif method == "umap":
            reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=10,
                        min_dist=0.05,
                        random_state=42,
                        **kwargs
                    )
        else:
            raise ValueError("Invalid method. Choose from 'pca', 'tsne', or 'umap'.")
        X_proj = reducer.fit_transform(all_feats)
    
        # ---- Prepare label colors ----
        unique_labels = np.unique(all_labels)
        # use tab10 colors cyclically
        cmap = plt.cm.get_cmap("tab10", max(10, len(unique_labels)))
        label_to_color = {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}
    
        # ---- Prepare cohort markers ----
        if cohort is None:
            # No cohort given: just color by labels, single legend
            plt.figure(figsize=(7, 6))
            for lab in unique_labels:
                sel = (all_labels == lab)
                plt.scatter(
                    X_proj[sel, 0], X_proj[sel, 1],
                    c=[label_to_color[lab]] * np.sum(sel),
                    s=16, alpha=0.9, edgecolors="none", label=str(lab)
                )
            plt.legend(title="Label")
            plt.title(f"{method.upper()} projection")
            plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
            plt.tight_layout(); plt.show()
            return X_proj
    
        # Cohort markers (defaults)
        if cohort_markers is None:
            cohort_markers = {"OPX": "o", "NEP": "s", "TCGA": "^", "Unknown": "x"}
        unique_cohorts = np.unique(cohort)
        if cohort_order is None:
            cohort_order = [c for c in ["OPX", "NEP", "TCGA"] if c in unique_cohorts]
            # append any remaining cohorts found
            cohort_order += [c for c in unique_cohorts if c not in cohort_order]
    
        # ---- Plot: color by label, marker by cohort ----
        plt.figure(figsize=(7, 6))
        for coh in cohort_order:
            sel_coh = (cohort == coh)
            if not np.any(sel_coh):
                continue
            # within each cohort, still separate by label so colors map to labels
            for lab in unique_labels:
                sel = sel_coh & (all_labels == lab)
                if not np.any(sel):
                    continue
                plt.scatter(
                    X_proj[sel, 0], X_proj[sel, 1],
                    c=[label_to_color[lab]] * np.sum(sel),
                    marker=cohort_markers.get(coh, "o"),
                    s=50, alpha=0.9, edgecolors="k", linewidths=0.3,
                    label=f"{coh}__{lab}"  # temp; we'll make clean legends below
                )
    
        plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
        plt.title(f"{method.upper()} projection (color = label, marker = cohort)")
    
        # ---- Build clean legends ----
        # Cohort legend from markers
        cohort_handles = []
        for coh in cohort_order:
            if np.any(cohort == coh):
                cohort_handles.append(
                    Line2D([0], [0], marker=cohort_markers.get(coh, "o"),
                           linestyle="", markerfacecolor="white", markeredgecolor="k",
                           markersize=8, label=coh)
                )
        leg1 = plt.legend(handles=cohort_handles, title="Cohort", loc="upper right")
    
        # Label legend from colors
        label_handles = [
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor=label_to_color[lab], markeredgecolor='k',
                   markersize=8, label=str(lab))
            for lab in unique_labels
        ]
        plt.gca().add_artist(leg1)  # keep cohort legend
        plt.legend(handles=label_handles, title="Label", loc="lower right")
    
        plt.tight_layout()
        plt.show()
        return X_proj


    #plot embedding after training
    # train_x, train_y = get_slide_embedding(train_data)
    # test_x, test_y   = get_slide_embedding(test_data)
    # test_x2, test_y2   = get_slide_embedding(test_data2)

    # scaler = MinMaxScaler(feature_range=(0,1))
    # train_x = scaler.fit_transform(train_x)
    # test_x = scaler.fit_transform(test_x)
    # test_x2 = scaler.fit_transform(test_x2)
    
    # knn = KNeighborsClassifier(n_neighbors = 1)
    # knn.fit(train_x, train_y)
    
    # y_pred = knn.predict(train_x)
    # y_pred_prob = knn.predict_proba(train_x)[:,1]
    # compute_performance(train_y,y_pred_prob,y_pred, "OPX_TRAIN")
    
    
    # y_pred = knn.predict(test_x)
    # y_pred_prob = knn.predict_proba(test_x)[:,1]
    # compute_performance(test_y,y_pred_prob,y_pred, "OPX")
    
    # y_pred = knn.predict(test_x2)
    # y_pred_prob = knn.predict_proba(test_x2)[:,1]
    # compute_performance(test_y2,y_pred_prob,y_pred, "NEP")
    
        

