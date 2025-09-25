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
from plot_utils import plot_umap
from Loss import FocalLoss, compute_logit_adjustment
from TransMIL import TransMIL
from misc_utils import str2bool
from misc_utils import create_dir_if_not_exists, set_seed
 
#FOR MIL-Lab
sys.path.insert(0, os.path.normpath(os.path.join(os.getcwd(), '..', '..', 'other_model_code','MIL-Lab',"src")))
from models.abmil import ABMILModel
from models.dsmil import DSMILModel
from models.transmil import TransMILModel

# source ~/.bashrc
# conda activate mil

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


import random

def downsample(train_data, n_times=10, n_samples=100, seed=None):
    """
    Downsample positives and negatives with reproducibility option.
    
    Args:
        train_data (list): dataset [(x, label), ...]
        n_times (int): how many resamples
        n_samples (int): number of positives/negatives to select each time
        seed (int or None): random seed for reproducibility
    Returns:
        list of lists: each element is a balanced dataset
    """
    pos_data = [x for x in train_data if x[1] == 1]
    neg_data = [x for x in train_data if x[1] == 0]

    if seed is not None:
        random.seed(seed)

    all_samples = []
    for i in range(n_times):
        selected1 = random.sample(pos_data, n_samples)   # sample positives
        selected0 = random.sample(neg_data, n_samples)   # sample negatives
        selected_all = selected1 + selected0
        random.shuffle(selected_all)
        all_samples.append(selected_all)
    return all_samples

def count_num_tiles(indata, cohort_name):
    n_tiles = [x['x'].shape[0] for x in indata]
    ids = [x['sample_id'] for x in indata]
    labels = [x['y'].squeeze().numpy() for x in indata]
    
    sample_df = pd.DataFrame({'SAMPLIE_ID':ids, 'N_TILES': n_tiles})
    label_df = pd.DataFrame(labels, columns=[f"LABEL_{i}" for i in range(len(labels[0]))])
    label_df.columns = ["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI"]
    sample_df = pd.concat([sample_df.reset_index(drop=True),
                           label_df.reset_index(drop=True)], axis=1)

    
    df = pd.DataFrame({'cohort_name': cohort_name,
                    'AVG_N_TILES': np.mean(n_tiles).round(),
                    'Median_N_TILES': np.median(n_tiles).round(),
                  'MAX_N_TILES': max(n_tiles),
                  'MIN_N_TILES': min(n_tiles)}, index = [0])
    
    return df,sample_df
        

def plot_n_tiles_by_labels(df, label_cols=None, value_col="N_TILES", agg="mean", save_path=None):
    """
    Plot grouped bar charts of N_TILES by binary label columns.
    """
    if label_cols is None:
        label_cols = ["AR", "HR1", "HR2", "PTEN", "RB1", "TP53", "TMB", "MSI"]

    grouped = {}
    for col in label_cols:
        if agg == "mean":
            grouped[col] = df.groupby(col)[value_col].mean()
        elif agg == "sum":
            grouped[col] = df.groupby(col)[value_col].sum()
        else:
            raise ValueError("agg must be 'mean' or 'sum'")

    grouped_df = pd.DataFrame(grouped)

    ax = grouped_df.plot(kind="bar", figsize=(10, 6))
    plt.title(f"{agg.capitalize()} {value_col} by Label Group")
    plt.ylabel(f"{agg.capitalize()} {value_col}")
    plt.xlabel("Label value (0 or 1)")
    plt.xticks(rotation=0)
    plt.legend(title="Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_embeddings(all_feats, all_labels, method="pca", **kwargs):
    method = method.lower()
    
    if method == "pca":
        reducer = PCA(n_components=2, **kwargs)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, **kwargs)
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP is not installed. Install it with `pip install umap-learn`.")
        reducer = umap.UMAP(n_components=2, random_state=42, **kwargs)
    else:
        raise ValueError("Invalid method. Choose from 'pca', 'tsne', or 'umap'.")

    X_proj = reducer.fit_transform(all_feats)

    # Plot with fixed red/blue colors
    plt.figure(figsize=(6, 6))
    label_color_map = {0: "blue", 1: "red"}

    for label in np.unique(all_labels):
        plt.scatter(
            X_proj[all_labels == label, 0],
            X_proj[all_labels == label, 1],
            c=label_color_map.get(label, "gray"),  # gray if unexpected label
            label=str(label),
            s=10
        )

    plt.legend()
    plt.title(f"{method.upper()} projection of embeddings")
    plt.show()
    plt.show()

    return X_proj

##
#train_x, train_y = get_slide_embedding(train_data, model)
#test_x, test_y = get_slide_embedding(test_data, model)

# #get classifer features and plot
# all_feats_train, all_labels_train = extract_features(model, test_data, device, layer_idx=3)
# proj = plot_embeddings(all_feats_train, all_labels_train, method="PCA")

# # Scatter plot
# plt.figure(figsize=(6,6))
# scatter = plt.scatter(proj[:,0], proj[:,1], c=all_labels_train, cmap="bwr", s=20, alpha=0.8)
# plt.colorbar(scatter, label="Label")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.grid(True)
# plt.show()

def compute_performance(y_true,y_pred_prob,y_pred_class, cohort_name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel() #CM

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob, pos_label=1)
    
    # Average precision score = PR-AUC
    PR_AUC = average_precision_score(y_true, y_pred_prob)

    AUC = round(metrics.auc(fpr, tpr),2)
    ACC = round(accuracy_score(y_true, y_pred_class),2)
    F1 = round(f1_score(y_true, y_pred_class),2)
    F2 = round(fbeta_score(y_true, y_pred_class,beta = 2),2)
    F3 = round(fbeta_score(y_true, y_pred_class,beta = 3),2)
    Recall = round(recall_score(y_true, y_pred_class),2)
    Precision = round(precision_score(y_true, y_pred_class),2)
    Specificity = round(tn / (tn + fp),2)
    perf_tb = pd.DataFrame({"AUC": AUC,
                            "PR_AUC":PR_AUC,
                            "Recall": Recall,
                            "Specificity":Specificity,
                            "ACC": ACC,
                            "Precision":Precision,
                            "F1": F1,
                            "F2": F2,
                            "F3": F3},index = [cohort_name])
    
    return perf_tb

        
def get_slide_embedding(indata, net):
    net.eval()
    rep_list = []
    label_list = []
    with torch.no_grad():
        for data_it, data in enumerate(indata):
            
            #Get data
            x = data[0].to(device, dtype=torch.float32).unsqueeze(0) #[1, N_Patches, N_FEATURE]
            y = data[1][0].long().view(-1).to(device)
            
            #Run model            
            results, results2 = net(
                x,
                return_slide_feats = True)
            
            rep_list.append(results2['slide_feats'].detach().cpu())
            label_list.append(y.detach().cpu())
            
    slides_rep = torch.concat(rep_list).numpy()
    labels = torch.concat(label_list).numpy()
        
    return slides_rep, labels



def extract_features(model, dataloader, device, layer_idx=2):
    """
    Extract features from a specified classifier layer using forward hooks.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model.
    dataloader : DataLoader
        DataLoader providing (xb, yb, ..., ...) batches.
    device : torch.device
        Device to run the model on.
    layer_idx : int, optional (default=2)
        Index of the layer in `model.model.classifier` to hook.

    Returns
    -------
    all_feats : torch.Tensor
        Extracted features concatenated across all batches. Shape: [N, D]
    all_labels : torch.Tensor
        Labels concatenated across all batches. Shape: [N]
    """

    all_feats = []
    all_labels = []
    features = []

    # Hook function
    def grab_layer(_, __, output):
        features.append(output.detach().cpu())

    # Register hook
    handle = model.model.classifier[layer_idx].register_forward_hook(grab_layer)

    model.eval()
    with torch.no_grad():
        for xb, yb, *_ in dataloader:  # handles datasets with extra items
            features.clear()  # reset buffer per batch
            _ = model(xb.to(device).unsqueeze(0))
            all_feats.append(features[0])   # shape [B, D]
            all_labels.append(yb.cpu())

    # Remove hook
    handle.remove()

    # Concatenate results
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0).squeeze()

    return all_feats, all_labels

def plot_loss(train_loss, val_loss, save_path):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss", marker='o')
    plt.plot(range(1, len(train_loss) + 1), val_loss, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
            

def _uniform_sample(coords: np.ndarray, feats: np.ndarray, max_bag: int, seed: int = None, grid: int = 32) -> tuple:
    
    rng = np.random.default_rng(seed)
    
   
    # If fewer tiles than max_bag, just return them all
    N = len(coords)
    if N  <= max_bag:
        chosen = np.arange(N, dtype=int)
        return feats[chosen], coords[chosen], chosen
    
    # --- normal uniform sampling ---

    mins, maxs = coords.min(0), coords.max(0)

    norm   = (coords - mins) / (maxs - mins + 1e-8)

    bins   = np.floor(norm * grid).astype(np.int16)

    keys   = bins[:, 0] * grid+ bins[:, 1]

    #order  = np.random.permutation(len(keys))
    order  = rng.permutation(len(keys))  # use seeded RNG

    chosen, seen = [], set()

    for idx in order:

        k = keys[idx]

        if k not in seen:

            seen.add(k); chosen.append(idx)

            if len(chosen) == max_bag:

                break

    if len(chosen) < max_bag:        # pad if needed

        rest = np.setdiff1d(np.arange(len(keys)), chosen, assume_unique=True)
        
        extra = np.random.choice(rest, max_bag-len(chosen), replace=False)

        chosen = np.concatenate([chosen, extra])

    chosen = np.asarray(chosen, dtype=int)

    return feats[chosen], coords[chosen], chosen


import numpy as np

def _uniform_sample_with_cancer_prob(
    coords: np.ndarray,
    feats: np.ndarray,
    cancer_prob: np.ndarray,
    max_bag: int,
    seed: int = None,
    grid: int = 32,
    strength: float = 1.0,
    blend_uniform: float = 0.05,
) -> tuple:
    """
    Weighted spatial sampling:
      1) build GRID bins to encourage spatial diversity (<=1 pick/bin in the first pass)
      2) iterate tiles in a *probability-weighted* random order (higher prob first on average)
      3) pad remaining slots (if any) with weighted picks from the rest

    Args:
        coords: (N, 2) or (N, D) integer/float tile coordinates.
        feats:  (N, F) features aligned with coords.
        cancer_prob: (N,) or (N,1) per-tile probabilities in [0,1].
        max_bag: target number of tiles to sample.
        seed: RNG seed for reproducibility.
        grid: grid resolution used to enforce spatial diversity.
        strength: exponent on probabilities; >1 sharpens (more aggressive toward high prob),
                  <1 flattens. =0 becomes uniform (after blending).
        blend_uniform: small epsilon to ensure nonzero mass and keep some exploration.

    Returns:
        feats[chosen], coords[chosen], chosen_indices (np.int64)
    """
    rng = np.random.default_rng(seed)

    N = len(coords)
    if N == 0:
        return feats[:0], coords[:0], np.asarray([], dtype=np.int64)

    if N <= max_bag:
        chosen = np.arange(N, dtype=np.int64)
        return feats[chosen], coords[chosen], chosen

    # --- prep probabilities ---
    p = np.asarray(cancer_prob).reshape(-1)
    if p.shape[0] != N:
        raise ValueError(f"cancer_prob length {p.shape[0]} != N={N}")

    # sanitize: clip, replace NaNs, power 'strength', and blend with uniform so no zeros
    p = np.nan_to_num(p, nan=0.0)
    p = np.clip(p, 0.0, None)
    if strength != 1.0:
        # if all zeros, p**strength still zeros; blending below handles this.
        p = np.power(p, max(strength, 0.0))
    if not np.isfinite(p).all() or p.sum() == 0:
        # fallback to uniform
        p = np.ones_like(p, dtype=float)
    # blend with uniform mass
    if blend_uniform > 0:
        u = np.full_like(p, 1.0 / N)
        p = (1.0 - blend_uniform) * p + blend_uniform * u
    # normalize
    p = p / p.sum()

    # --- grid binning (same as your function, parametric grid) ---
    mins, maxs = coords.min(0), coords.max(0)
    norm = (coords - mins) / (maxs - mins + 1e-8)
    bins = np.floor(norm * grid).astype(np.int16)
    # allow coords with >2 dims: use first two for keys
    b0 = bins[:, 0]
    b1 = bins[:, 1] if bins.shape[1] > 1 else np.zeros_like(b0)
    keys = (b0 * grid + b1).astype(np.int32)

    # --- probability-weighted permutation without replacement ---
    # numpy supports p with replace=False (interpreted as successive draws proportional to p)
    order = rng.choice(N, size=N, replace=False, p=p)

    # --- first pass: one per bin, in weighted order ---
    chosen = []
    seen = set()
    for idx in order:
        k = int(keys[idx])
        if k not in seen:
            seen.add(k)
            chosen.append(idx)
            if len(chosen) == max_bag:
                break

    # --- pad if needed (still weighted, from the rest) ---
    if len(chosen) < max_bag:
        chosen = np.asarray(chosen, dtype=np.int64)
        mask = np.ones(N, dtype=bool)
        mask[chosen] = False
        rest_idx = np.nonzero(mask)[0]
        if rest_idx.size > 0:
            # renormalize probabilities over the remaining pool
            p_rest = p[rest_idx]
            p_rest = p_rest / p_rest.sum()
            extra = rng.choice(rest_idx, size=(max_bag - len(chosen)), replace=False, p=p_rest)
            chosen = np.concatenate([chosen, extra.astype(np.int64)], axis=0)
        else:
            # should be rare; just return what we have
            chosen = chosen.astype(np.int64)

    # stable dtype and order (optional shuffle to avoid deterministic bin sequence)
    # chosen = rng.permutation(chosen)  # uncomment if you want final order shuffled
    return feats[chosen], coords[chosen], chosen



import ast

def uniform_sample_all_samples(indata, incoords, max_bag = 100, grid = 32, sample_by_tf = True, plot = False):
    
    new_data_list = []
    for data_item, coord_item in zip(indata,incoords):            
        
        #get feature
        feats = data_item[0] #(N_tiles, 1536)
        label = data_item[1]
        tfs  = data_item[2]
        sl  = data_item[3]
        #get coordiantes
        coords = coord_item

        #uniform sampling
        if sample_by_tf == False:
            sampled_feats, sampled_coords, sampled_index = _uniform_sample(coords, feats, max_bag, grid = grid, seed = 1)
        else:
            sampled_feats, sampled_coords, sampled_index = _uniform_sample_with_cancer_prob(coords, feats, tfs, max_bag, 
                                                                                            seed = 1, 
                                                                                            grid = grid, 
                                                                                            strength = 1.0)

        if plot:
            # 3. Plot results
            plt.figure(figsize=(8, 8))
            plt.scatter(coords[:, 0], -coords[:, 1], alpha=0.3, label="All Tiles") #- for  flip Y
            plt.scatter(sampled_coords[:, 0], -sampled_coords[:, 1], color="red", label="Sampled Tiles") #- for  flip Y
            plt.legend()
            plt.title("Uniform Sampling with Grid Constraint")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
        
        
        sampled_tf = tfs[sampled_index]
        sampled_site_loc =sl[sampled_index]
        new_tuple = (sampled_feats,label, sampled_tf, sampled_site_loc)
        new_data_list.append(new_tuple)
        
        
    return new_data_list


from itertools import chain

def combine_all(split, keys=('test', 'train', 'val')):
    parts = [split[k] for k in keys]
    data  = list(chain.from_iterable(p[0] for p in parts))
    sp    = list(chain.from_iterable(p[1] for p in parts))
    pt    = list(chain.from_iterable(p[2] for p in parts))
    coh   = list(chain.from_iterable(p[3] for p in parts))
    return data, sp, pt, coh

def just_test(split):
    a, b, c, d, _ = split['test']
    return a, b, c, d



def train(train_loader, 
          model, 
          criterion, 
          optimizer, 
          model_name = 'Transfer_MIL' ,
          l2_coef = 1e-4, 
          logit_adjustments = None,
          logit_adj_train = True):
    """ Run one train epoch """
    
    total_loss = 0
    model.train()
    for data_it, data in enumerate(train_loader):
        
        #Get data
        x = data[0].to(device, dtype=torch.float32) #[1, N_Patches, N_FEATURE]
        y = data[1][0].long().view(-1).to(device)
        tf = data[2].to(device, dtype=torch.float32)            #(1, N_Patches]
                        
        #Run model     
        if model_name == 'TransMIL':
            results = model(data = x)
        elif model_name == "Transfer_MIL":
            results, _ = model(x)
        
        #Get output   
        output = results['logits']
        
        #Logit adjustment 
        if logit_adj_train == True:
            output = output + logit_adjustments.to(device)
        
        #Get loss
        loss = criterion(output, y)
        
        #L2 reg
        loss_r = 0
        for parameter in model.parameters():
            loss_r += torch.sum(parameter ** 2)
        
        loss = loss +  l2_coef * loss_r

        total_loss += loss    #Store total loss 
        
        optimizer.zero_grad() #zero grad, avoid grad acuum
        loss.backward()       #compute new gradients
        optimizer.step()      #update paraters
        
    
    avg_loss = total_loss/(len(train_loader)) 
    
    return avg_loss



def validate(test_data, sample_ids, model, criterion, logit_adjustments, 
             model_name = 'Transfer_MIL', logit_adj_infer = True, logit_adj_train = False,
             l2_coef = 0):
       
    model.eval()
    with torch.no_grad():
        total_loss = 0
        logits_list = []
        labels_list = []
        logits_list_adj = []
        for i in range(len(test_data)):
            x, y, tf, sl = test_data[i]        
            x = x.unsqueeze(0).to(device)      # add batch dim
            y = y.long().view(-1).to(device)
                            
            #Run model     
            if model_name == 'TransMIL':
                results = model(data = x)
            elif model_name == "Transfer_MIL":
                results, _ = model(x)
            
            #Get output   
            output = results['logits']
            logits_list.append(output.squeeze(0).detach().cpu()) #Get logit before posthoc adj
            loss = criterion(output, y) #get loss before post-hoc logit adust)
            
            #get adjust output post hoc
            if logit_adj_infer == True:
                adj_output = output - logit_adjustments.to(device) #get logit after posthoc adj
                logits_list_adj.append(adj_output.squeeze(0).detach().cpu()) #Get logit before posthoc adj
            else:
                logits_list_adj = None
                        
            #if logit adjustment used for training
            if logit_adj_train == True:
                loss = criterion(output + logit_adjustments.to(device), y)
                
            #L2 reg
            loss_r = 0
            for parameter in model.parameters():
                loss_r += torch.sum(parameter ** 2)
            loss = loss +  l2_coef * loss_r
                
            total_loss += loss #Store total loss 
            
            #get label
            labels_list.append(int(y.detach().cpu()))
        
        avg_loss = total_loss/(len(test_data)) 
        
    
    df = get_predict_df(logits_list, labels_list, sample_ids, logits_list_adj, thres = 0.5)
    
    return avg_loss, df


def get_predict_df(logits_list, labels_list, sample_ids, logits_list_adj = None, thres = 0.5):
    
    
    #Get logits and prob df
    logits = torch.stack(logits_list, dim=0)              # (N, C)
    probs  = torch.softmax(logits, dim=1)             # (N, C)
    
    num_classes = logits.size(1)
    logit_cols = [f"logit_{i}" for i in range(num_classes)]
    prob_cols  = [f"prob_{i}"  for i in range(num_classes)]
    
    df_logits = pd.DataFrame(logits.numpy(), columns=logit_cols)
    df_probs  = pd.DataFrame(probs.numpy(),  columns=prob_cols)
    df = pd.concat([df_logits, df_probs], axis=1)
    df["Pred_Class"] = (df["prob_1"] > thres).astype(int)
    
    
    #if post hoc logit adj
    if logits_list_adj is not None:
        logits_adj = torch.stack(logits_list_adj, dim=0)               # (N, C)
        probs_adj  = torch.softmax(logits_adj, dim=1)           # (N, C)
        
        num_classes = logits_adj.size(1)
        logit_cols = [f"adj_logit_{i}" for i in range(num_classes)]
        prob_cols  = [f"adj_prob_{i}"  for i in range(num_classes)]
        
        df_logits_adj = pd.DataFrame(logits_adj.numpy(), columns=logit_cols)
        df_probs_adj  = pd.DataFrame(probs_adj.numpy(),  columns=prob_cols)
        df_adj = pd.concat([df_logits_adj, df_probs_adj], axis=1)
        df_adj["Pred_Class_adj"] = (df_adj["adj_prob_1"] > thres).astype(int)
        
        #combine ther before logit adj
        df = pd.concat([df, df_adj], axis = 1)
            
    #add label and ids
    df["True_y"] = labels_list
    df['SAMPLE_ID'] = sample_ids


    if logits_list_adj is not None:
        #reorder
        df = df[['SAMPLE_ID','True_y',
                 'logit_0', 'logit_1','prob_0', 'prob_1','Pred_Class',
                 'adj_logit_0', 'adj_logit_1','adj_prob_0', 'adj_prob_1','Pred_Class_adj']]
    else:
        #reorder
        df = df[['SAMPLE_ID','True_y',
                 'logit_0', 'logit_1','prob_0', 'prob_1','Pred_Class']]
    
    return df


def run_eval(data, sp_ids, cohort, criterion, logit_adj_infer, logit_adj_train, l2_coef=0):
    avg_loss, pred_df = validate(
        data, sp_ids, model, criterion,
        logit_adjustments,
        model_name='Transfer_MIL',
        logit_adj_infer=logit_adj_infer,
        logit_adj_train=logit_adj_train,
        l2_coef = l2_coef,
    )

    if logit_adj_infer:
        y_true, prob, pred = pred_df['True_y'], pred_df['adj_prob_1'], pred_df['Pred_Class_adj']
    else:
        y_true, prob, pred = pred_df['True_y'], pred_df['prob_1'], pred_df['Pred_Class']

    print(compute_performance(y_true, prob, pred, cohort))
    return avg_loss, pred_df


import math
import copy

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.num_bad_epochs = 0
        self.best_state = None

    def step(self, metric, model):
        """
        Check if validation metric improved.
        Returns True if training should stop.
        """
        improved = (self.best - metric) > self.min_delta
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience

    def __repr__(self):
        return (f"EarlyStopper(patience={self.patience}, min_delta={self.min_delta}, "
                f"best={self.best}, num_bad_epochs={self.num_bad_epochs})")

    
    

def build_model(model_name: str, device, num_classes=2, n_feature=None):
    """
    Create and return a model based on the model_name.

    Args:
        model_name (str): Name of the model type ('Transfer_MIL' or 'TransMIL').
        device (torch.device): Device to move the model to.
        num_classes (int): Number of output classes.
        n_feature (int): Required only for TransMIL.

    Returns:
        model (nn.Module): The constructed model on the specified device.
    """
    if model_name == "Transfer_MIL":
        model = create_model('abmil.base.uni_v2.pc108-24k', num_classes=num_classes)
        model.to(device)

        # in_dim = model.model.classifier.in_features
        # model.model.classifier = nn.Sequential(
        #     nn.Dropout(p=0.3),
        #     nn.Linear(in_dim, 64),
        #     nn.Linear(64, 32),
        #     nn.Linear(32, num_classes),  # last layer matches num_classes
        # )
        # model.to(device)

    elif model_name == "TransMIL":
        if n_feature is None:
            raise ValueError("n_feature must be provided for TransMIL")
        model = TransMIL(n_classes=num_classes, in_dim=n_feature)
        model.to(device)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--tumor_frac', default= 0.0, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath, virchow2')
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='HR2', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--logit_adj_train', default=True, type=str2bool, help='Train with logit adjustment')
parser.add_argument('--logit_adj_infer', default=True, type=str2bool, help='Train with logit adjustment')

parser.add_argument('--out_folder', default= 'pred_out_092425', type=str, help='out folder name')

############################################################################################################
#     Model Para
############################################################################################################
parser.add_argument('--DIM_OUT', default=512, type=int, help='')
parser.add_argument('--droprate', default=0.01, type=float, help='drop out rate')
parser.add_argument('--lr', default = 3e-4, type=float, help='learning rate') #0.01 works for DA with union , OPX + TCGA
parser.add_argument('--train_epoch', default=10, type=int, help='')

            
if __name__ == '__main__':
    
    args = parser.parse_args()
    #fold_list = [0,1,2,3,4]
    fold_list = [0]

    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    
    ######################
    #Create output-dir
    ######################
    outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                           args.mutation,
                           'FOLD' + str(fold_list[0]))
    outdir1 =  outdir0  + "/saved_model/"
    outdir2 =  outdir0  + "/model_para/"
    outdir3 =  outdir0  + "/logs/"
    outdir4 =  outdir0  + "/predictions/"
    outdir5 =  outdir0  + "/perf/"
    outdir_list = [outdir0,outdir1,outdir2,outdir3,outdir4,outdir5]
    
    for out_path in outdir_list:
        create_dir_if_not_exists(out_path)
            
    
    
    ###################################
    #Load
    ###################################
    cohorts = [
        #"z_nostnorm_OPX",
        #"z_nostnorm_TCGA_PRAD",
        #"z_nostnorm_Neptune",
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
    
    # opx_ol100_nst = data['z_nostnorm_OPX_ol100']
    # opx_ol0_nst = data['z_nostnorm_OPX_ol0']
    # tcga_ol100_nst = data['z_nostnorm_TCGA_PRAD_ol100']
    # tcga_ol0_nst = data['z_nostnorm_TCGA_PRAD_ol0']
    # nep_ol100_nst = data['z_nostnorm_Neptune_ol100']
    # nep_ol0_nst = data['z_nostnorm_Neptune_ol0']
    
    opx_ol100 = data['OPX_ol100']
    opx_ol0 = data['OPX_ol0']
    tcga_ol100 = data['TCGA_PRAD_ol100']
    tcga_ol0 = data['TCGA_PRAD_ol0']
    nep_ol100 = data['Neptune_ol100']
    nep_ol0= data['Neptune_ol0']
    
        
    ##########################################################################################
    #Count tiles
    ##########################################################################################
    # outdir_0 = os.path.join(proj_dir, "intermediate_data","8_discrip_stats","tile_counts")
    # #Count AVG , MAX, MIN number of tiles
    # nep_count_df, nep_sample_count_df = count_num_tiles(nep_ol0,'Nep')
    # tcga_count_df, tcga_sample_count_df = count_num_tiles(tcga_ol0,'TCGA')
    # opx_count_df, opx_sample_count_df = count_num_tiles(opx_ol0,'OPX')
    # all_count_df = pd.concat([opx_count_df, tcga_count_df, nep_count_df])
    # all_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0.csv"))
    # nep_sample_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0_nep.csv"))
    # tcga_sample_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0_tcga.csv"))
    # opx_sample_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0_opx.csv"))
    
    # # Plot bar charts
    # plot_n_tiles_by_labels(opx_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL0_OPX.png"))
    # plot_n_tiles_by_labels(nep_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL0_NEP.png"))
    # plot_n_tiles_by_labels(tcga_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL0_TCGA.png"))

    
    # #Count AVG , MAX, MIN number of tiles
    # nep_count_df, nep_sample_count_df = count_num_tiles(nep_ol100,'Nep')
    # tcga_count_df, tcga_sample_count_df = count_num_tiles(tcga_ol100,'TCGA')
    # opx_count_df, opx_sample_count_df = count_num_tiles(opx_ol100,'OPX')
    # all_count_df = pd.concat([opx_count_df, tcga_count_df, nep_count_df])
    # all_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100.csv"))
    # nep_sample_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100_nep.csv"))
    # tcga_sample_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100_tcga.csv"))
    # opx_sample_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100_opx.csv"))
    # plot_n_tiles_by_labels(opx_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL100_OPX.png"))
    # plot_n_tiles_by_labels(nep_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL100_NEP.png"))
    # plot_n_tiles_by_labels(tcga_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL100_TCGA.png"))
    
    
    
    ##########################################################################################
    #Merge st norm and no st-norm
    ##########################################################################################

    # opx_union_ol100  = merge_data_lists(opx_ol100_nst, opx_ol100, merge_type = 'union')
    # opx_union_ol0    = merge_data_lists(opx_ol0_nst, opx_ol0, merge_type = 'union')
    # tcga_union_ol100 = merge_data_lists(tcga_ol100_nst, tcga_ol100, merge_type = 'union')
    # tcga_union_ol0   = merge_data_lists(tcga_ol0_nst, tcga_ol0, merge_type = 'union')
    # nep_union_ol100  = merge_data_lists(nep_ol100_nst, nep_ol100, merge_type = 'union')
    # nep_union_ol0    = merge_data_lists(nep_ol0_nst, nep_ol0, merge_type = 'union')
    
    #Combine
    comb_ol100 = opx_ol100 + tcga_ol100 
    comb_ol0   = opx_ol0 + tcga_ol0 

    for f in fold_list:
        f = 0
        ####################################
        #Load data
        ####################################    
        #get train test and valid
        opx_split    =  load_dataset_splits(opx_ol0, opx_ol0, f, args.mutation, concat_tf = False)
        tcga_split   =  load_dataset_splits(tcga_ol0, tcga_ol0, f, args.mutation, concat_tf = False)
        nep_split    =  load_dataset_splits(nep_ol0, nep_ol0, f, args.mutation, concat_tf = False)
        comb_splits  =  load_dataset_splits(comb_ol0, comb_ol0, f, args.mutation, concat_tf = False)
 

        train_data, train_sp_ids, train_pt_ids, train_cohorts, train_coords  = comb_splits['train']
        test_data, test_sp_ids, test_pt_ids, test_cohorts, _ = comb_splits['test']
        val_data, val_sp_ids, val_pt_ids, val_cohorts,_ = comb_splits['val']
        

        
        # OPX all
        test_data0, test_sp_ids0, test_pt_ids0, test_cohorts0 = combine_all(opx_split)

        # NEP all
        test_data2, test_sp_ids2, test_pt_ids2, test_cohorts2 = combine_all(nep_split)

        # TCGA all
        test_data3, test_sp_ids3, test_pt_ids3, test_cohorts3 = combine_all(tcga_split)

        # OPX / TCGA / NEP test only
        test_data4, test_sp_ids4, test_pt_ids4, test_cohorts4 = just_test(opx_split)
        test_data5, test_sp_ids5, test_pt_ids5, test_cohorts5 = just_test(tcga_split)
        test_data6, test_sp_ids6, test_pt_ids6, test_cohorts6 = just_test(nep_split)
        


        #samplling, sample could has less than 400, if original tile is <400
        train_data = uniform_sample_all_samples(train_data, train_coords, max_bag = 500, 
                                                grid = 32, sample_by_tf = True, plot = False)

        

        #UMAP
        # all_feature_train, all_labels_train, site_list_train = get_feature_label_site(train_data)
        # all_feature_test, all_labels_test, site_list_test = get_feature_label_site(test_data)
        # plot_umap(all_feature_train, all_labels_train, site_list_train, train_cohorts)

       
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
        
        # plot_umap(train_x, train_y, [], [])
        

        ####################################################
        #Select GPU
        ####################################################
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(device)
                
        
        #Feature and Label N
        N_FEATURE =  train_data[0][0].shape[1]
        N_LABELS  =  train_data[0][1].shape[1]
        
        
        #ABMILModel(in_dim=1024, num_classes=2, config = None)
        
        args.lr = 1e-3
        args.logit_adj_train = False
        args.l2_coef = 5e-4
        model_name = "Transfer_MIL"
        
        # construct the model from src and load the state dict from HuggingFace 
        model = build_model(model_name = model_name, 
                    device = device, 
                    num_classes=2, 
                    n_feature = N_FEATURE)
            
        loss_fn = FocalLoss(alpha=-1, gamma=0, reduction='mean')
 

        
        
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
        logit_adjustments, label_freq = compute_logit_adjustment(train_loader, tau = 2) #[-0.2093, -4.6176] The rarer class (1) gets a much more negative adjustment, which means during training its logits will be shifted down harder unless the model compensates.

        early_stopper = EarlyStopper(patience=5, min_delta=1e-4)  
        
                
        # avg_loss, pred_df = run_eval(train_data, train_sp_ids, "TRAIN",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data4, test_sp_ids4, "OPX_test",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data5, test_sp_ids5, "TCGA_test",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data6, test_sp_ids6, "NEP_test",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data0, test_sp_ids0, "OPX_All",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data3, test_sp_ids3, "TCGA_All",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data2, test_sp_ids2, "NEP_ALL",    loss_fn,  logit_adj_infer = False, logit_adj_train = args.logit_adj_train)


        train_loss = []
        val_loss =[]
        for epoch in range(100):
            
            avg_loss  =  train(train_loader, model, loss_fn, optimizer, 
                                     model_name = 'Transfer_MIL' ,l2_coef = args.l2_coef, 
                                     logit_adjustments = logit_adjustments,
                                     logit_adj_train = args.logit_adj_train)
            lr_scheduler.step()
            
            avg_loss_val, pred_df_val = validate(val_data, val_sp_ids, model, loss_fn, 
                                         logit_adjustments, 
                                         model_name = 'Transfer_MIL',
                                         logit_adj_infer = args.logit_adj_infer,
                                         logit_adj_train = args.logit_adj_train,
                                         l2_coef = args.l2_coef)
            
            
            
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
        
        
        
        
            
        
        avg_loss, pred_df = run_eval(train_data, train_sp_ids, "TRAIN",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data4, test_sp_ids4, "OPX_test",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data5, test_sp_ids5, "TCGA_test",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data6, test_sp_ids6, "NEP_test",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data0, test_sp_ids0, "OPX_All",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data3, test_sp_ids3, "TCGA_All",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data2, test_sp_ids2, "NEP_ALL",    loss_fn,  logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )

        

        ##############################################################################
        # DEcoupled training stage 2
        ##############################################################################
        mode_idxes = 27        
        checkpoint = torch.load(os.path.join(outdir1,'checkpoint'+ str(mode_idxes) + '.pth'))
        
        model = build_model(model_name = model_name, 
                    device = device, 
                    num_classes=2, 
                    n_feature = N_FEATURE)
        
        model.load_state_dict(checkpoint)
        
        args.logit_adj_train = False
        
        ###Freeze the feature extraction part
        freeze_feature = True
        if freeze_feature:
            if model_name == "Transfer_MIL":
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.model.classifier.parameters():
                    p.requires_grad = True
            elif model_name == "TransMIL":
                for p in model.parameters():
                    p.requires_grad = False
                for p in model._fc2.parameters():
                    p.requires_grad = True
            
        #double check
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
        
        # optimizer = torch.optim.AdamW(
        #     filter(lambda p: p.requires_grad, model.parameters()),
        #     lr=args.lr, #3e-4,
        # )
        
        loss_fn = FocalLoss(alpha=0.8, gamma=2, reduction='mean')
        
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr,
                            momentum=0.9,
                            weight_decay=1e-4,
                            nesterov=True)
        

        #down sampling
        pos_count = len([x for x in train_data if x[1] == 1])
        print(pos_count)
        samples = downsample(train_data, n_times=10, n_samples=pos_count, seed=42)
        
        #train loader
        for samp in samples:
            
            #Train loader
            train_loader = DataLoader(dataset=samp,batch_size=1, shuffle=False)
            
            # Scheduler 
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        
            
            # Set the network to training mode
            logit_adjustments, label_freq = compute_logit_adjustment(train_loader, tau = 2) #[-0.2093, -4.6176] The rarer class (1) gets a much more negative adjustment, which means during training its logits will be shifted down harder unless the model compensates.
    
    
            early_stopper = EarlyStopper(patience=5, min_delta=1e-4)  
            
            train_loss = []
            val_loss =[]
            for epoch in range(100):
                
                avg_loss  =  train(train_loader, model, loss_fn, optimizer, 
                                         model_name = 'Transfer_MIL' ,l2_coef = args.l2_coef, 
                                         logit_adjustments = logit_adjustments,
                                         logit_adj_train = args.logit_adj_train)
                lr_scheduler.step()
                
                avg_loss_val, pred_df_val = validate(val_data, val_sp_ids, model, loss_fn, 
                                             logit_adjustments, 
                                             model_name = 'Transfer_MIL',
                                             logit_adj_infer = args.logit_adj_infer,
                                             logit_adj_train = args.logit_adj_train,
                                             l2_coef = args.l2_coef)
                
                
                
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
                torch.save(model.state_dict(), outdir1 + "checkpoint_stage2_" + str(epoch) + '.pth')
            
                # Early stop check
                if early_stopper.step(avg_loss_val.item(), model):
                    print(f"Early stopping at epoch {epoch} (best val loss {early_stopper.best:.4f})")
                    break
        
        plot_loss(train_loss, val_loss, outdir5)
        
        avg_loss, pred_df = run_eval(train_data, train_sp_ids, "TRAIN",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data4, test_sp_ids4, "OPX_test",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data5, test_sp_ids5, "TCGA_test",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data6, test_sp_ids6, "NEP_test",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data0, test_sp_ids0, "OPX_All",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data3, test_sp_ids3, "TCGA_All",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        avg_loss, pred_df = run_eval(test_data2, test_sp_ids2, "NEP_ALL",    loss_fn,  logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )

        
        
       
        
        comb_data = test_data4 + test_data5 
        comb_ids  = test_sp_ids4 + test_sp_ids5 
        avg_loss, pred_df = run_eval(comb_data, comb_ids, "TCGA_OPX_NEP",    loss_fn,  logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )


        


    

import torch
from typing import List, Tuple

import torch
import pandas as pd
import numpy as np

def _norm(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v.float().reshape(-1), dim=0)

def _pair_cos(a_feat: torch.Tensor, b_feat: torch.Tensor, reduce: str = "flatten") -> float:
    """
    Cosine similarity between two feature tensors using your reduction rules:
    - try the requested `reduce`
    - if flatten sizes differ, fallback to 'mean'
    """
    va = _to_vec(a_feat, reduce=reduce)
    vb = _to_vec(b_feat, reduce=reduce)

    if reduce == "flatten" and va.numel() != vb.numel():
        # fallback to mean for both
        va = _to_vec(a_feat, reduce="mean")
        vb = _to_vec(b_feat, reduce="mean")

    # If sizes *still* differ (very rare unless D differs across items), truncate to min
    if va.numel() != vb.numel():
        L = min(va.numel(), vb.numel())
        va = va[:L]
        vb = vb[:L]

    va = _norm(va)
    vb = _norm(vb)
    return torch.dot(va, vb).item()

def compute_similarity_matrix(
    dataset,                    # List[Tuple[Tensor, Tensor, Tensor, Tensor]]
    ids,                        # List[str] aligned with dataset
    reduce: str = "flatten"     # or "mean"
):
    n = len(dataset)
    S = torch.empty((n, n), dtype=torch.float32)
    for i in range(n):
        ai = dataset[i][0]
        # diagonal is always 1.0 (after normalization), but compute for consistency
        for j in range(n):
            bj = dataset[j][0]
            S[i, j] = _pair_cos(ai, bj, reduce=reduce)
    # DataFrame for nice labeling/CSV
    df = pd.DataFrame(S.numpy(), index=ids, columns=ids)
    return S, df

# ---- Run it on your data ----
# Assumes `test_sp_ids4` is a list of IDs and `test_data4` is aligned (same order).
S_torch, S_df = compute_similarity_matrix(test_data4, test_sp_ids4, reduce="flatten")

# Save as CSV and NumPy (optional)
csv_path = "similarity_matrix.csv"
npy_path = "similarity_matrix.npy"
S_df.to_csv(csv_path)
np.save(npy_path, S_torch.numpy())

labels = [x[1].item() for x in test_data4]
label_df = pd.DataFrame({'ids': test_sp_ids4, args.mutation: labels})


