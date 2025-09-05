#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 17:41:03 2025

@author: jliu6
"""

import random
import torch
import numpy as np
import os
import pandas as pd
from PIL import ImageCms, Image
import sys
import matplotlib.pyplot as plt
import umap
import umap.plot
sys.path.insert(0, '../Utils/RandomSplit-main/')
from RandomSplit import MakeBalancedCrossValidation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
    
def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")
        
        
# Min-max normalization function
def minmax_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor



def convert_img(in_img):
    srgb_profile = ImageCms.createProfile("sRGB")
    converted_img = ImageCms.profileToProfile(in_img, srgb_profile, srgb_profile)

    return converted_img



def count_mutation_perTrainTest(train_test_df, selected_label):

    # Compute N (count) per TRAIN/TEST group
    train_df = train_test_df[train_test_df['TRAIN_OR_TEST'] == 'TRAIN']
    test_df  = train_test_df[train_test_df['TRAIN_OR_TEST'] == 'TEST']
    
    n_train = len(train_df)
    n_test = len(test_df)
    
    # Initialize result storage
    results = []
    
    for col in selected_label:
        train_n = train_df[col].sum()
        test_n = test_df[col].sum()
        
        train_pct = (train_n / n_train) * 100 if n_train > 0 else 0
        test_pct = (test_n / n_test) * 100 if n_test > 0 else 0
        
        # Format as "N (%)"
        train_formatted = f"{train_n} ({train_pct:.1f}%)"
        test_formatted = f"{test_n} ({test_pct:.1f}%)"
        
        results.append({
            'Outcome': col,
            'Train N (%)': train_formatted,
            'Test N (%)': test_formatted
        })
    # Convert to DataFrame for nice display
    results_df = pd.DataFrame(results)

    
    return results_df



def count_mutation_perTrainTestVAL(train_test_df, selected_label, n_Folds = 5):
    
    # train_test_df = train_test_valid_df
    # selected_label = SELECTED_LABEL
    
    all_res = []
    for k in range(n_Folds):
        #Update dataframe                                                 
        train_df = train_test_df.loc[train_test_df['FOLD' + str(k)] == 'TRAIN']
        test_df = train_test_df.loc[train_test_df['FOLD' + str(k)] == 'TEST']
        val_df = train_test_df.loc[train_test_df['FOLD' + str(k)] == 'VALID']
                                                                        
    
        n_train = len(train_df)
        n_test = len(test_df)
        n_val = len(val_df)
    
        # Initialize result storage
        results = []
        
        for col in selected_label:
            train_n = train_df[col].sum()
            test_n = test_df[col].sum()
            val_n = val_df[col].sum()
            
            train_pct = (train_n / n_train) * 100 if n_train > 0 else 0
            test_pct = (test_n / n_test) * 100 if n_test > 0 else 0
            val_pct = (val_n / n_val) * 100 if n_test > 0 else 0
            
            # Format as "N (%)"
            train_formatted = f"{train_n} ({train_pct:.1f}%)"
            test_formatted = f"{test_n} ({test_pct:.1f}%)"
            val_formatted = f"{val_n} ({val_pct:.1f}%)"
            
            results.append({
                'Outcome': col,
                'Train N (%)': train_formatted,
                'Val N (%)': val_formatted,
                'Test N (%)': test_formatted,
                
            })
        # Convert to DataFrame for nice display
        results_df = pd.DataFrame(results)
        results_df['FOLD'] = k
        all_res.append(results_df)
    
    all_res_df = pd.concat(all_res)

    
    return all_res_df


def get_pos_neg_ids(tile_info_pt, label_name):
    #Postive IDs
    pos_ids = list(tile_info_pt.loc[tile_info_pt[label_name] == 1 , 'PATIENT_ID'].unique()) #24
    
    #Neg Ids
    neg_ids = list(tile_info_pt.loc[tile_info_pt[label_name] == 0 , 'PATIENT_ID'].unique()) #242
    
    return pos_ids, neg_ids




def generate_balanced_cv_list(patient_label_data, selected_label, n_Folds = 5, p_test = 0.25):
    
    # patient_label_data = tile_info_df_pt.copy()
    # selected_label = SELECTED_LABEL.copy()
    # n_Folds = 5
    # p_test=0.25
        
    #Start
    label_df = patient_label_data[selected_label].copy()
    label_df.index = patient_label_data['PATIENT_ID']
    label_df['self_count'] = 1 #Add self count for the cases has zero mutations, so the randomsplit algorithm would run

    
    #Get input
    w  = label_df.T
    ids = list(w.columns)
    w = np.array(w)
    column_map = {i: ids[i] for i in range(w.shape[1])}

    training_lists, validation_lists, test_list, res, res_test = MakeBalancedCrossValidation(w, n_Folds, column_map, testing_size=p_test, tries=10)
    
    return training_lists, validation_lists, test_list



def mutation_sample_summary(df_subset, mut_cols):
    sample_total = len(df_subset)
    if sample_total == 0:
        # Return zeros if the subset is empty (avoid divide-by-zero)
        out = pd.DataFrame(index=mut_cols, columns=["Mutated_n", "Total_n", "Percent", "n(%)"])
        out["Mutated_n"] = 0
        out["Total_n"] = 0
        out["Percent"] = 0.0
        out["n(%)"] = "0 (0.0%)"
        return out

    sample_counts = df_subset[mut_cols].sum(numeric_only=True)
    sample_perc   = (sample_counts / sample_total * 100).round(1)

    summary = pd.DataFrame({
        "Mutated_n": sample_counts.astype(int),
        "Total_n": sample_total,
        "Percent": sample_perc
    })
    summary["n(%)"] = summary.apply(lambda x: f"{x.Mutated_n:.0f} ({x.Percent:.1f}%)", axis=1)
    return summary



def get_feature_label_site(indata):
    feature_list = []
    label_list = []
    site_list = []
    for x in indata:
        features = x[0].mean(dim = 0, keepdim = True)
        labels = x[1]
        labels_repeated = labels.expand(features.shape[0], -1)
        site = x[3].unique()[0].item()
        feature_list.append(features)
        label_list.append(labels_repeated)
        site_list.append(site)
        
    all_feature =  torch.concat(feature_list, dim = 0)
    all_labels =  torch.concat(label_list, dim = 0).squeeze().numpy()
    
    return all_feature, all_labels, site_list



def plot_umap(feature_tensor, label_list, site_label_list, corhor_label_list):
    
    color_key = {
            "OPX": "blue",
            "TCGA": "green",
            "NEP": "red"
        }
    
    color_key = {
            0: "blue",
            1: "red",
        }
    
    mapper = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        init="spectral",      
        random_state=42       
    )
    
    mapper = mapper.fit(feature_tensor)
    embedding = mapper.transform(feature_tensor)
    umap.plot.points(mapper, labels=label_list, color_key = color_key)

    
    plt.figure(figsize=(7,6))
    for site in np.unique(site_label_list):
        idx = site_label_list == site
        plt.scatter(
            embedding[idx, 0], embedding[idx, 1],
            c=[color_key[l] for l in label_list[idx]],       # color by label
            marker="o" if site == 0 else "s",                # shape by site
            alpha=0.7, s=30, label=f"Site {site}"
        )
    
    plt.title("UMAP: color=Label, shape=Site")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend()
    plt.show()