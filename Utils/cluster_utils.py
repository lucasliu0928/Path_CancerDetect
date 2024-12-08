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
