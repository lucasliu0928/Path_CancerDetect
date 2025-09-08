#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:37:19 2025

@author: jliu6
"""

import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot

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