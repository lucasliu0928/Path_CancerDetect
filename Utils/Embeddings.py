#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 22:42:39 2025

@author: jliu6
"""

import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import os




#get average emebdding from the original data
def get_feature_label_site(indata: list = [], 
                           tf_threshold: float = 0.0,
                           agg_method: str = 'mean'):
    
    feature_list = []
    label_list = []
    site_list = []
    id_list = []
    ignored_id_list = []   # <── track skipped samples
    
    for data in indata:
        # get tile features
        features = data['x'].numpy()
        
        # get tumor fraction
        tf = data['tumor_fraction'].numpy()
        
        # select tiles >= threshold
        mask = tf >= tf_threshold
        if not np.any(mask):  
            # nothing passes threshold → skip this sample
            ignored_id_list.append(data['sample_id'])
            continue
        
        features = features[mask]
        
        # aggregate features
        if agg_method == 'mean':
            features_agg = features.mean(axis=0)
        elif agg_method == 'max':
            features_agg = features.max(axis=0)
        else:
            raise ValueError(f"Unsupported agg_method: {agg_method}")
        
        # get labels
        labels = data['y'].squeeze().numpy()  
        
        # get site
        site = data['site_location'].unique()[0].numpy()
        
        # get sample id
        sampled_id = data['sample_id']
        
        feature_list.append(features_agg)
        label_list.append(labels)
        site_list.append(site)
        id_list.append(sampled_id)
        
    # stack final results (only non-empty samples included)
    all_feature = np.stack(feature_list, axis=0)
    all_labels  = np.stack(label_list, axis=0)
    all_sites   = np.stack(site_list, axis=0)
    
    return all_feature, all_labels, all_sites, id_list, ignored_id_list


#Get slides meddbning from TransferMIL model
def get_slide_embedding(indata, net, device):
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



def plot_embeddings(all_feats, all_labels, method="pca", **kwargs):
            
    method = method.lower()
    
    if method == "pca":
        reducer = PCA(n_components=2, **kwargs)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, **kwargs)
    elif method == "umap":
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


    
if __name__ == '__main__':
    pass
    # #Example
    # train_x, train_y = get_slide_embedding(train_data, model)
    # test_x, test_y = get_slide_embedding(test_data, model)
    
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