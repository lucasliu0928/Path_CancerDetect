#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:41:17 2024

@author: jliu6
"""

from torch.utils.data import Dataset
import PIL
import torch
import pandas as pd
import numpy as np

class pull_tiles(Dataset):
    def __init__(self, tile_info, deepzoom_tiles, tile_levels):
        super().__init__()
        self.tile_info = tile_info
        self.deepzoom_tiles = deepzoom_tiles
        self.tile_levels = tile_levels
        self.mag_extract = list(set(tile_info['MAG_EXTRACT']))[0]
        self.save_image_size = list(set(tile_info['SAVE_IMAGE_SIZE']))[0]


    def __getitem__(self, idx):
        #Get x, y index
        tile_ind = self.tile_info['TILE_XY_INDEXES'].iloc[idx].strip("()").split(", ")
        x ,y = int(tile_ind[0]) , int(tile_ind[1])

        #Pull tiles
        tile_pull = self.deepzoom_tiles.get_tile(self.tile_levels.index(self.mag_extract), (x, y))
        tile_pull = tile_pull.resize(size=(self.save_image_size, self.save_image_size),resample=PIL.Image.LANCZOS) #resize

        return tile_pull

# Function to convert coordinate string to dictionary
def convert_to_dict(row):
    coords = tuple(map(int, row['pred_map_location'].strip('()').split(', ')))
    return {'coords': coords, 'att': row['ATT']}



class ModelReadyData_MT_V2(Dataset):
    def __init__(self,
                 feature_df,
                 label_df,
                ):
        
        self.x = torch.FloatTensor(feature_df)
        
        # Get the Y labels
        self.y = torch.FloatTensor(label_df)
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x = self.x[index]
        y = self.y[index]
        
        return x, y

def get_feature_label_array(input_path, feature_folder, selected_ids,selected_labels, selected_feature):

    feature_label_list = []
    for pt in selected_ids:
        input_dir = input_path + pt + '/' + 'features/' + 'train_features_' + feature_folder + '.h5'
        cur_feature_df = pd.read_hdf(input_dir, key='feature')
        cur_feature_df.columns = cur_feature_df.columns.astype(str)
        cur_label_df = pd.read_hdf(input_dir, key='tile_info')[["SAMPLE_ID"] + selected_labels]
        cur_comb_df = pd.concat([cur_feature_df,cur_label_df], axis = 1) #Add ID and label to feature df
        feature_label_list.append(cur_comb_df)
        
    full_df = pd.concat(feature_label_list)
    
    
    #Get label numpy
    label_df = full_df[["SAMPLE_ID"] + selected_labels]
    label_df = label_df.drop_duplicates(subset= ['SAMPLE_ID'])
    label_array = label_df.drop(columns='SAMPLE_ID').values.astype('float32')
    
    #get feature numpy
    feature_df = full_df[["SAMPLE_ID"] + selected_feature]
    # Convert DataFrame to 3-D tensor
    feature_list = []    
    for group in list(label_df['SAMPLE_ID']):
        group_data = feature_df[feature_df['SAMPLE_ID'] == group].drop(columns='SAMPLE_ID').values
        feature_list.append(group_data)
    feature_3d = np.stack(feature_list)

    return feature_3d, label_array