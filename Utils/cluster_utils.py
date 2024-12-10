#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 03:27:29 2024

@author: jliu6
"""

import sys
import os
import numpy as np
import cv2
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import xml.etree.ElementTree as ET
from xml.dom import minidom
import geojson
import argparse
import matplotlib.pyplot as plt
import fastai
from fastai.vision.all import *
import PIL
matplotlib.use('Agg')
import pandas as pd
import datetime
from skimage import draw, measure, morphology, filters
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
from shapely.ops import cascaded_union, unary_union
import json
import shapely
import warnings
import datetime
import random
import torch
from PIL import ImageCms, Image
warnings.filterwarnings("ignore")
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Get clustering data
def get_cluster_data(feature_list, label_list, id_list, selected_labels):
    feature_list = [pd.DataFrame(x) for x in feature_list]
    label_list = [y.squeeze() for y in label_list]
    
    for i,x in enumerate(feature_list):
        x['ID'] = id_list[i]
        for j,l in enumerate(selected_labels):
            x[l] = int(label_list[i][j])
    feature_df = pd.concat(feature_list)

    #Change feature tumor frac name
    feature_df.rename(columns = {2048: 'TUMOR_PIXEL_PERC'}, inplace = True)
    
    return feature_df


def get_cluster_label(feature_df, cluster_centers, cluster_features):
    r'''
    Get Cluster label by compute dist between test/valid embedding to the center of kmeans
    '''
    distances = cdist(cluster_centers, feature_df[cluster_features].to_numpy(), 'euclidean')
    closest_indices = np.argmin(distances, axis=0)

    cluster_labels  = closest_indices

    return cluster_labels
    
def get_updated_feature(input_df, selected_ids, selected_feature):
    feature_list = []
    ct = 0 
    for pt in selected_ids:
        if ct % 10 == 0 : print(ct)

        cur_df = input_df.loc[input_df['ID'] == pt]

        #Extract feature, label and tumor info
        cur_feature = cur_df[selected_feature].values

        feature_list.append(cur_feature)
        ct += 1

    return feature_list


def get_pca_components(feature_data, n_components = 2):
    pca = PCA(n_components = n_components)
    pcs = pca.fit_transform(feature_data)
    explained_variance = pca.explained_variance_
    print("Explained variance by each component:", explained_variance)

    return pcs




def plot_cluster_distribution(info_df, selected_label, out_path, cohort_name):
    #Plot scatter cluster plot for each outcome,
    for plot_outcome in selected_label:
        plot_data = info_df[['PC1','PC2', 'Cluster'] + [plot_outcome]]
            
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate dots by Cluster but color by Outcome
        for cluster in plot_data['Cluster'].unique():
            subset = plot_data[plot_data['Cluster'] == cluster]
            ax.scatter(subset['PC1'], subset['PC2'], 
                       s=np.where(subset[plot_outcome] == 1, 20, 0.001), 
                       c=['steelblue' if outcome == 0 else 'darkred' for outcome in subset[plot_outcome]], 
                       alpha=0.6,
                       linewidth=1.5, label=f'Cluster {cluster}',
                       zorder=3 if (subset[plot_outcome] == 1).any() else 2)
        
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(plot_outcome)
        plt.grid(True)
        plt.savefig(out_path +  '/cluster_scatter_' + plot_outcome + '_' + cohort_name + '.png')
        plt.close()
        
    # Plot distribution of outcome by cluster (stacked bar plot)   
    for plot_outcome in selected_label:
        plot_data = info_df[['PC1','PC2', 'Cluster'] + [plot_outcome]]
     
        # Create a crosstab to count the occurrences of each outcome per cluster
        crosstab = pd.crosstab(plot_data['Cluster'], plot_data[plot_outcome])
        
        # Calculate the percentage of each outcome per cluster
        percentage_crosstab = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        
        # Plot the stacked bar chart
        percentage_crosstab.plot(kind='bar', stacked=True, color=['steelblue', 'darkred'])
        plt.xlabel('Cluster')
        plt.ylabel('Percentage')
        plt.title('Bar Chart of ' + plot_outcome + ' per Cluster')
        plt.legend(title=plot_outcome, loc='center left', bbox_to_anchor=(1.0, 0.5))
        
        plt.tight_layout()
        plt.savefig(out_path + "/outcome_distribution_" + '_' + cohort_name + '.png')
        plt.close()
    
    # Plot combined distribution of outcome by cluster 
    for plot_outcome in selected_label:
        plot_data = info_df[['PC1','PC2', 'Cluster'] + [plot_outcome]]
    
        # Create a crosstab to count the occurrences of each outcome per cluster
        crosstab = pd.crosstab(plot_data[plot_outcome],plot_data['Cluster'])
    
        # Calculate the percentage of each outcome per cluster
        percentage_crosstab = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        
        # Plot the stacked bar chart
        percentage_crosstab.plot(kind='bar', stacked=False, color=['#440154','#3b528b','#5ec962','#fde725'])
        plt.xlabel(plot_outcome)
        plt.ylabel('Percentage')
        plt.title('Bar Chart of Clusters Per Outcome')
        plt.legend(title='Cluster', loc='center left', bbox_to_anchor=(1.0, 0.5))
    
        plt.tight_layout()
        plt.savefig(out_path  + '/cluster_distribution_' + '_' + cohort_name + '.png')
        plt.close()




def load_alltile_tumor_info(input_path, patient_id, selected_labels,info_df):
    all_tumor_info_df = pd.read_csv(os.path.join(input_path, patient_id, "ft_model/", patient_id + "_TILE_TUMOR_PERC.csv"))
    selected_tumor_info_df = info_df.loc[info_df['SAMPLE_ID'] == patient_id]
    
    comb_df = all_tumor_info_df.merge(selected_tumor_info_df, on = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP',
                                                          'LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE',
                                                          'TISSUE_COVERAGE', 'pred_map_location', 'TUMOR_PIXEL_PERC'], how = 'left')
    #Fill NA with the mutation label for the sample, they are the same arcoss all tiles
    cols = selected_labels + ['Bx Type','Anatomic site','Notes','SITE_LOCAL']
    for label in cols:
        if len(comb_df[label].dropna().unique()) > 0:
            comb_df[label].fillna(comb_df[label].dropna().unique()[0], inplace=True)

    return comb_df
