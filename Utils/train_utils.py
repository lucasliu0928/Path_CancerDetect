#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:41:17 2024

@author: jliu6
"""

from torch.utils.data import Dataset
from torch.utils.data import Subset
import PIL
import torch
import pandas as pd
import numpy as np
import os
from misc_utils import convert_img
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import random

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False



        
def get_feature_idexes(method, include_tumor_fraction = True):
    
    if method == 'retccl':
        selected_feature = [str(i) for i in range(0,2048)] 
    elif method == 'uni1': 
        selected_feature = [str(i) for i in range(0,1024)] 
    elif method == 'uni2' or method == 'prov_gigapath':
        selected_feature = [str(i) for i in range(0,1536)] 

    if include_tumor_fraction == True:
        selected_feature = selected_feature + ['TUMOR_PIXEL_PERC'] 
        
    return selected_feature

def has_seven_csv_files(folder_path):
    # List all files ending with .csv
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and os.path.isfile(os.path.join(folder_path, f))]
    return len(csv_files) == 6

def get_selected_labels(mutation_name, train_cohort):
    
    all_labels = ["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI"]    
    
    if mutation_name == 'MT':
        if 'TCGA' in train_cohort:
            selected = ["AR" , "HR2", "PTEN","RB1","TP53","MSI"]   #without TMB
        else:
            selected = ["AR",  "HR2", "PTEN","RB1","TP53","TMB","MSI"]   
    else:
            selected = mutation_name.split('_')

    #get label index in all labels
    selected_index = []
    for l in selected:
        selected_index.append(all_labels.index(l))
    
    return selected, selected_index

    
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
        tile_pull = convert_img(tile_pull)

        #Get ATT info
        tile_att = self.tile_info['ATT'].iloc[idx]

        #Get Map info
        tile_coord = self.tile_info['pred_map_location'].iloc[idx]


        return tile_pull, tile_att, tile_coord

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


def get_sample_feature(folder_name, feature_path, input_file_name):
    #Input dir
    input_dir = os.path.join(feature_path, folder_name, 'features', 'features_alltiles_' + input_file_name + '.h5')
    
    #feature
    feature_df = pd.read_hdf(input_dir, key='feature')
    feature_df.columns = feature_df.columns.astype(str)
    feature_df.reset_index(drop = True, inplace = True)

    #Tile ID (Do not use the labels in this, only use the tile info, because the label has been updated in all_tile_info_df from 3_otherinfo)
    cols_to_keep = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 
                    'PIXEL_OVERLAP','LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 
                    'WHITE_SPACE','TISSUE_COVERAGE']
    id_df = pd.read_hdf(input_dir, key='tile_info')[cols_to_keep]
    id_df.reset_index(drop = True, inplace = True)

    #Combine feature and IDs
    feature_df = pd.concat([id_df,feature_df], axis = 1)

    return feature_df
    
def get_sample_label(sample_id, all_label_data, id_col = 'SAMPLE_ID'):
    label_df = all_label_data.loc[all_label_data[id_col] == sample_id]
    label_df.reset_index(drop = True, inplace = True)
    return label_df

def combine_feature_and_label(feature_df,label_df):
    cols_to_map = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 
                'PIXEL_OVERLAP','LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 
                'WHITE_SPACE','TISSUE_COVERAGE']
    if feature_df.shape[0] == label_df.shape[0]:
        comb_df = feature_df.merge(label_df, on = cols_to_map)
    else:
        print("N Tiles does not match")

    return comb_df

def compare_modelreadydata(d1, d2):
    if len(d1) != len(d2):
        print(f"Different lengths: {len(d1)} vs {len(d2)}")
        return False

    all_match = True

    for i in range(len(d1)):
        if not torch.equal(d1.x[i], d2.x[i]):
            print(f"[x] mismatch at index {i}")
            all_match = False
        if not torch.equal(d1.y[i], d2.y[i]):
            print(f"[y] mismatch at index {i}")
            all_match = False
        if not torch.equal(d1.tf[i], d2.tf[i]):
            print(f"[tf] mismatch at index {i}")
            all_match = False
        if not d1.other_info[i].equals(d2.other_info[i]):
            print(f"[other_info] mismatch at index {i}")
            all_match = False

    return all_match

# def extract_feature_label_tumorinfo_np(selected_df, selected_feature, selected_labels):
#     #Extract feature, label and tumor info
#     feature_np = selected_df[selected_feature].values #np array
#     label_np   = selected_df[selected_labels].drop_duplicates().values.astype('float32') #numpy array
#     info_np    = selected_df[['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP', 'LIMIT_BOUNDS', 
#                                'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE', 'TISSUE_COVERAGE',
#                               'SITE_LOCAL', 'pred_map_location', 'TUMOR_PIXEL_PERC']]
#     tf_info_np = selected_df['TUMOR_PIXEL_PERC'].values

#     return feature_np, label_np, info_np, tf_info_np


# def get_feature_label_array_dynamic(feature_path, tumor_info_path, feature_name, selected_ids,selected_labels, selected_feature, tumor_fraction_thres = 0):
#     r'''
#     #if test, no tumor tiles, select all other tiles
#     #if train, no tumor tiles, do not include in the train list
#     '''
#     feature_list = []
#     label_list = []
#     info_list = []
#     tumor_info_list = []
#     id_list = []
#     ct = 0 
#     for pt in selected_ids:
#         if ct % 10 == 0 : print(ct)

#         #Combined feature label and tumor info
#         #TODELETE
#         cur_comb_df = combine_feature_label_tumorinfo(pt, feature_path, tumor_info_path, feature_name)
        
#         #Select tumor fraction > X tiles
#         cur_comb_df_tumor = cur_comb_df.loc[cur_comb_df['TUMOR_PIXEL_PERC'] >= tumor_fraction_thres].copy()
#         cur_comb_df_tumor = cur_comb_df_tumor.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
#         cur_n_tumor_tiles = cur_comb_df_tumor.shape[0] # N of tumor tiles
        


#         if tumor_fraction_thres == 0: #select all tiles
#             cur_selected_df =  cur_comb_df 
#         elif tumor_fraction_thres > 0: #select tumor tiles based on the threshold
#             cur_selected_df =  cur_comb_df_tumor 
#         cur_selected_df = cur_selected_df.reset_index(drop = True)
        
#         if cur_selected_df is not None :
#             #Extract feature, label and tumor info
#             cur_feature, cur_label, cur_info, cur_tf_info =  extract_feature_label_tumorinfo_np(cur_selected_df, selected_feature, selected_labels)
#             feature_list.append(cur_feature)
#             label_list.append(cur_label)
#             info_list.append(cur_info)
#             tumor_info_list.append(cur_tf_info)
#             id_list.append(pt)
#             ct += 1
            
#     return feature_list, label_list, info_list, tumor_info_list, id_list




def combine_feature_label_tumorinfo_tma(patient_id, feature_path, tumor_info_path, input_file_name, selected_labels):

    #Input dir
    input_dir = feature_path + patient_id + '/' + 'features/' + input_file_name + '.h5'

    #feature
    feature_df = pd.read_hdf(input_dir, key='feature')
    feature_df.columns = feature_df.columns.astype(str)
    
    #Label
    label_df = pd.read_hdf(input_dir, key='tile_info')
    label_df.reset_index(drop = True, inplace = True)
    #add lacking labels as nan to fit the input format in the model
    labels_notintma = [x for x in selected_labels if x not in label_df.columns]
    for label in labels_notintma:
        label_df[label] = np.nan
    
    #Add tumor info to label
    tumor_info_df = pd.read_csv(os.path.join(tumor_info_path, patient_id, "ft_model/", patient_id + "_TILE_TUMOR_PERC.csv"))
    tumor_info_df.reset_index(drop = True, inplace = True)
    label_df = label_df.merge(tumor_info_df, on = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP',
                                                   'LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE',
                                                   'TISSUE_COVERAGE'])
    
    #Combine feature and label and tumor info
    comb_df = pd.concat([feature_df,label_df], axis = 1)

    return comb_df

def extract_feature_label_tumorinfo_np_tma(selected_df, selected_feature, selected_labels):
    #Extract feature, label and tumor info
    feature_np = selected_df[selected_feature].values #np array
    label_np   = selected_df[selected_labels].drop_duplicates().values.astype('float32') #numpy array
    info_np    = selected_df[['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP', 'LIMIT_BOUNDS', 
                               'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE', 'TISSUE_COVERAGE', 'SITE_LOCAL', 'pred_map_location', 'TUMOR_PIXEL_PERC']]
    tf_info_np = selected_df['TUMOR_PIXEL_PERC'].values

    return feature_np, label_np, info_np, tf_info_np


def get_feature_label_array_dynamic_tma(feature_path, tumor_info_path, feature_name, selected_ids,selected_labels, selected_feature, tumor_fraction_thres = 0):
    
    feature_list = []
    label_list = []
    info_list = []
    tumor_info_list = []
    id_list = []
    ct = 0 
    for pt in selected_ids:
        if ct % 100 == 0 : print(ct)
    
        #Combined feature label and tumor info
        cur_comb_df = combine_feature_label_tumorinfo_tma(pt, feature_path, tumor_info_path, feature_name, selected_labels)
        
        #Select tumor fraction > X tiles
        cur_comb_df_tumor = cur_comb_df.loc[cur_comb_df['TUMOR_PIXEL_PERC'] >= tumor_fraction_thres].copy()
        cur_comb_df_tumor = cur_comb_df_tumor.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
        cur_n_tumor_tiles = cur_comb_df_tumor.shape[0] # N of tumor tiles
    
        if tumor_fraction_thres == 0: #select all tiles
            cur_selected_df =  cur_comb_df 
        elif tumor_fraction_thres > 0: #select tumor tiles based on the threshold
            cur_selected_df =  cur_comb_df_tumor 
        cur_selected_df = cur_selected_df.reset_index(drop = True)
    
        if cur_selected_df is not None :
            #Extract feature, label and tumor info
            cur_feature, cur_label, cur_info, cur_tf_info =  extract_feature_label_tumorinfo_np_tma(cur_selected_df, selected_feature, selected_labels)
            feature_list.append(cur_feature)
            label_list.append(cur_label)
            info_list.append(cur_info)
            tumor_info_list.append(cur_tf_info)
            id_list.append(pt)
            ct += 1
        
    return feature_list, label_list, info_list, tumor_info_list, id_list


def extract_before_third_hyphen(id_string):
    parts = id_string.split('-')
    return '-'.join(parts[:3])
    
def combine_feature_label_tumorinfo_TCGA(patient_id, label_dict, feature_path, tumor_info_path, input_file_name):

    #Input dir
    input_dir = feature_path + patient_id + '/' + 'features/' + input_file_name + '.h5'

    #feature
    feature_df = pd.read_hdf(input_dir, key='feature')
    feature_df.columns = feature_df.columns.astype(str)
    
    #Label
    label_df = pd.read_hdf(input_dir, key='tile_info')
    label_df.reset_index(drop = True, inplace = True)
    
    
    #Add tumor info to label
    cur_slides_name = [f for f in os.listdir(os.path.join(tumor_info_path, patient_id, "ft_model/")) if '.csv' in f][0].replace('_TILE_TUMOR_PERC.csv','')
    tumor_info_df = pd.read_csv(os.path.join(tumor_info_path, patient_id, "ft_model/", cur_slides_name + "_TILE_TUMOR_PERC.csv"))
    tumor_info_df.reset_index(drop = True, inplace = True)
    label_df = label_df.merge(tumor_info_df, on = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP',
                                                   'LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE',
                                                   'TISSUE_COVERAGE'])    
    #Combine feature and label and tumor info
    comb_df = pd.concat([feature_df,label_df], axis = 1)
    comb_df['PATIENT_ID'] = comb_df['SAMPLE_ID'].apply(extract_before_third_hyphen)
    cur_patient = list(set(comb_df['PATIENT_ID']))[0]

    #Update label
    for k in label_dict.keys():
        if cur_patient in label_dict[k]:
            comb_df[k] = 1
        else:
            comb_df[k] = 0

    return comb_df


def get_feature_label_array_dynamic_TCGA(feature_path, label_dict, tumor_info_path, feature_name, selected_ids,selected_labels, selected_feature, tumor_fraction_thres = 0):
    r'''
    #if test, no tumor tiles, select all other tiles
    #if train, no tumor tiles, do not include in the train list
    '''
    feature_list = []
    label_list = []
    info_list = []
    tumor_info_list = []
    id_list = []
    ct = 0 
    for pt in selected_ids:
        if ct % 10 == 0 : print(ct)

        #Combined feature label and tumor info
        cur_comb_df = combine_feature_label_tumorinfo_TCGA(pt, label_dict, feature_path, tumor_info_path, feature_name)
        
        #Select tumor fraction > X tiles
        cur_comb_df_tumor = cur_comb_df.loc[cur_comb_df['TUMOR_PIXEL_PERC'] >= tumor_fraction_thres].copy()
        cur_comb_df_tumor = cur_comb_df_tumor.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
        cur_n_tumor_tiles = cur_comb_df_tumor.shape[0] # N of tumor tiles
        


        if tumor_fraction_thres == 0: #select all tiles
            cur_selected_df =  cur_comb_df 
        elif tumor_fraction_thres > 0: #select tumor tiles based on the threshold
            cur_selected_df =  cur_comb_df_tumor 
        cur_selected_df = cur_selected_df.reset_index(drop = True)
        
        if cur_selected_df is not None :
            #Extract feature, label and tumor info
            cur_feature, cur_label, cur_info, cur_tf_info =  extract_feature_label_tumorinfo_np(cur_selected_df, selected_feature, selected_labels)
            feature_list.append(cur_feature)
            label_list.append(cur_label)
            info_list.append(cur_info)
            tumor_info_list.append(cur_tf_info)
            id_list.append(pt)
            ct += 1
            
    return feature_list, label_list, info_list, tumor_info_list, id_list

class ModelReadyData_diffdim_V2(Dataset):
    def __init__(self,
                 tile_info_list,
                 selected_features,
                 selected_labels,
                ):

        #Get feature
        self.x =  [torch.FloatTensor(df[selected_features].to_numpy()) for df in tile_info_list]
        
        # Get the Y labels
        self.y =  [torch.FloatTensor(df[selected_labels].drop_duplicates().to_numpy()) for df in tile_info_list]

        #Get tumor fraction
        self.tf = [torch.FloatTensor(df['TUMOR_PIXEL_PERC'].to_numpy()) for df in tile_info_list]

        #Get other info
        self.other_info = [df.drop(columns = selected_features + selected_labels + ['TUMOR_PIXEL_PERC']) for df in tile_info_list]
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x  = self.x[index]
        y  = self.y[index]
        tf = self.tf[index]
        of = self.other_info[index]
        sp_id = of['SAMPLE_ID'].unique().item()
        pt_id = of['PATIENT_ID'].unique().item()
        
        return x, y, tf, of, sp_id, pt_id
        
class ModelReadyData_diffdim(Dataset):
    def __init__(self,
                 feature_list,
                 label_list,
                 tumor_info_list,
                 include_tumor_fraction = False,
                 include_cluster = False,
                 feature_name = 'retccl',
                ):

        #Feature order: list(range(0,2048)) + ['TUMOR_PIXEL_PERC','Cluster']

        if feature_name == 'retccl':
            n_total_f = 2048 + 2
        elif feature_name == 'uni1':
            n_total_f = 1024 + 2 
        elif feature_name == 'uni2':
            n_total_f = 1536 + 2 
            
        feature_indexes = list(range(0,n_total_f))
        if include_tumor_fraction == False and include_cluster == False:
            feature_indexes.remove(n_total_f - 2)
            feature_indexes.remove(n_total_f - 1)
        elif include_tumor_fraction == True and include_cluster == False:
            feature_indexes.remove(n_total_f -1)
        elif include_tumor_fraction == False and include_cluster == True:
            feature_indexes.remove(n_total_f - 2)
        elif include_tumor_fraction == True and include_cluster == True:
            feature_indexes = feature_indexes
            
        self.x =[torch.FloatTensor(feature[:,feature_indexes]) for feature in feature_list] 
        
        # Get the Y labels
        self.y = [torch.FloatTensor(label) for label in label_list] 

        self.tf = [torch.FloatTensor(tf) for tf in tumor_info_list] 
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x = self.x[index]
        y = self.y[index]
        tf= self.tf[index]
        
        return x, y, tf

class ModelReadyData_diffdim_withclusterinfo(Dataset):
    def __init__(self,
                 feature_list,
                 label_list,
                 tumor_info_list,
                 cluster_list,
                ):
        
        self.x =[torch.FloatTensor(feature) for feature in feature_list] 
        
        # Get the Y labels
        self.y = [torch.FloatTensor(label) for label in label_list] 

        self.tf = [torch.FloatTensor(tf) for tf in tumor_info_list] 

        self.c = [torch.FloatTensor(c) for c in cluster_list] 
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x = self.x[index]
        y = self.y[index]
        tf= self.tf[index]
        c= self.c[index]
        
        return x, y, tf,c

class ModelReadyData_Instance_based(Dataset):
    def __init__(self,
                 feature_data,
                 label_data,
                ):
            
        self.x = feature_data
        
        # Get the Y labels
        self.y = label_data
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x = self.x[index]
        y = self.y[index]
        
        return x, y

def modify_to_instance_based(indata):
    feature_list = []
    label_list = []
    for data_it, data in enumerate(indata):
        cur_labels = data[1]
        label_list.append(cur_labels.repeat(data[0].shape[0], 1))
        feature_list.append(data[0].squeeze())
    
    feature_data = torch.concat(feature_list, dim = 0) #torch.Size([N_TILES, 1536])
    label_data = torch.concat(label_list, dim = 0) #torch.Size([N_TILES, 7])

    indata_instance_based = ModelReadyData_Instance_based(feature_data, label_data)

    return indata_instance_based
    
class add_tile_xy(Dataset):
    def __init__(self, original_dataset, additional_component):
        self.original_dataset = original_dataset
        self.additional_component = additional_component

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_sample = self.original_dataset[idx]
        tile_xy = torch.tensor(list(self.additional_component[idx]['TILE_XY_INDEXES'].apply(self.str_to_tuple)))

        # Assuming original_sample is a tuple
        return original_sample + (tile_xy,)

    # Function to convert string coordinates to tuples
    def str_to_tuple(self,coord_str):
        coord_str = coord_str.strip('()')
        x, y = map(int, coord_str.split(','))
        return (x, y)



def prediction(in_dataloader, in_model, n_label, loss_function, device, mutation_type, all_selected_label, attention = True):
    in_model.eval()
    with torch.no_grad():
        running_loss = 0

        pred_prob_list = []
        y_true_list = []
        att_list = []
        for x,y,tf in in_dataloader:

            #predict
            if attention == True:
                yhat_list, att_score = in_model(x.to(device))
                att_list.append(att_score.squeeze().cpu().numpy())
            else:
                yhat_list  = in_model(x.to(device))
                
                
            pred_prob_list.append(torch.concat(yhat_list, axis = 1).squeeze().detach().cpu().numpy())
            y_true_list.append(y.squeeze().detach().cpu().numpy())
            

            #Compute loss
            loss_list = []
            for i in range(0,n_label):
                if mutation_type == "MT":
                    label_index = i
                else:
                    label_index = all_selected_label.index(mutation_type)
                cur_loss = loss_function(yhat_list[i],y[:,:,label_index].to(device))  #compute loss
                loss_list.append(cur_loss) 
            loss = sum(loss_list)
            running_loss += loss.detach().item() 

        #average loss across all sample
        avg_loss = running_loss/len(in_dataloader) 
        
    return pred_prob_list, y_true_list, att_list, avg_loss



def prediction_sepatt(in_dataloader, in_model, n_label, loss_function, device, mutation_type, all_selected_label, attention = True):
    in_model.eval()
    with torch.no_grad():
        running_loss = 0

        pred_prob_list = []
        y_true_list = []
        att_list = []
        for x,y,tf in in_dataloader:

            #predict
            
            if attention == True:
                yhat_list, att_score = in_model(x.to(device))
                att_each_out = [] 
                for i in range(0,len(att_score)):
                    att_each_out.append(att_score[i].squeeze().cpu().numpy())
                att_list.append(att_each_out)
            else:
                yhat_list  = in_model(x.to(device))
                
                
            pred_prob_list.append(torch.concat(yhat_list, axis = 1).squeeze().detach().cpu().numpy())
            y_true_list.append(y.squeeze().detach().cpu().numpy())
            

            #Compute loss
            loss_list = []
            for i in range(0,n_label):
                if mutation_type == "MT":
                    label_index = i
                else:
                    label_index = all_selected_label.index(mutation_type)
                cur_loss = loss_function(yhat_list[i],y[:,:,label_index].to(device))  #compute loss
                loss_list.append(cur_loss) 
            loss = sum(loss_list)
            running_loss += loss.detach().item() 

        #average loss across all sample
        avg_loss = running_loss/len(in_dataloader) 
        
    return pred_prob_list, y_true_list, att_list, avg_loss

#prediction for moransI
def prediction_m(in_dataloader, in_model, n_label, loss_function, device, mutation_type, all_selected_label, attention = True):
    in_model.eval()
    with torch.no_grad():
        running_loss = 0
    
        pred_prob_list = []
        y_true_list = []
        att_list = []
        for x,y,tf,coor in in_dataloader:
    
            #predict
            if attention == True:
                yhat_list, att_score = in_model(x.to(device),coor.to(device))
                att_list.append(att_score.cpu().detach().numpy())
            else:
                yhat_list  = in_model(x.to(device))
                
                
            pred_prob_list.append(torch.concat(yhat_list, axis = 0).squeeze().detach().cpu().numpy())
            y_true_list.append(y.squeeze().detach().cpu().numpy())
    
            #Compute loss
            loss_list = []
            for i in range(0,n_label):
                if mutation_type == "MT":
                    label_index = i
                else:
                    label_index = all_selected_label.index(mutation_type)
                cur_loss = loss_function(yhat_list[i],y[:,:,label_index][0].to(device))  #compute loss
                loss_list.append(cur_loss) 
            loss = sum(loss_list)
            running_loss += loss.detach().item() 
    
        #average loss across all sample
        avg_loss = running_loss/len(in_dataloader) 
        
    return pred_prob_list, y_true_list, att_list, avg_loss
    
def prediction_one_mute(in_dataloader, in_model, n_label, loss_function, device,attention = True):
    in_model.eval()
    with torch.no_grad():
        running_loss = 0

        pred_prob_list = []
        y_true_list = []
        att_list = []
        for x,y in in_dataloader:

            #predict
            if attention == True:
                yhat, att_score = in_model(x.to(device))
                att_list.append(att_score.squeeze().cpu().numpy())
            else:
                yhat  = in_model(x.to(device))
            pred_prob_list.append(yhat.squeeze().detach().cpu().numpy())
            y_true_list.append(y.squeeze().detach().cpu().numpy())
            


            
            #Compute loss
            loss = loss_func(yhat.squeeze(),y.squeeze().to(device))  #compute loss
            running_loss += loss.detach().item() 

        #average loss across all sample
        avg_loss = running_loss/len(in_dataloader) 
        
    return pred_prob_list, y_true_list, att_list, avg_loss



# def BCE_WithRegularization(output, target, lambda_coef, reg_type, model, class_weight,reduction = 'mean'):
    

#     #Compute loss for each sample
#     loss = - ( target * torch.log(output) + (1-target)*torch.log(1-output))
    
#     #Weight loss for each class
#     pos_idex = torch.where(target == 1)[0] #index of pos
#     neg_idex = torch.where(target == 0)[0] #index of neg
    
#     loss[neg_idex] =  loss[neg_idex]*class_weight[0] #neg class weight * correponding loss
#     loss[pos_idex] =  loss[pos_idex]*class_weight[1]

    
#     if reduction == 'mean':
#         loss = loss.mean()
#     #Regularization
#     l1_regularization = 0
#     l2_regularization = 0
#     for param in model.parameters():
#         l1_regularization += param.abs().sum()
#         l2_regularization += param.square().sum()
#     if reg_type == "L1":
#         loss = loss + lambda_coef*l1_regularization    
#     elif reg_type == "L2":
#         loss = loss + lambda_coef*l2_regularization 
#     else:
#         loss = loss 
       
#     return loss



class BCE_Weighted_Reg(nn.Module):
    def __init__(self, lambda_coef, reg_type, model, reduction = "mean", att_reg_flag = False):
        super(BCE_Weighted_Reg, self).__init__()
        self.lambda_coef = lambda_coef
        self.reg_type = reg_type
        self.model = model
        self.reduction = reduction
        self.att_reg_flag = att_reg_flag 

        self.att_reg_loss = nn.MSELoss()

    def forward(self, output, target, class_weight, tumor_fractions, attention_scores):

        #Compute BCE
        loss = - (target * torch.log(output) + (1-target)*torch.log(1-output))
        
        #Weight loss for each class
        pos_idex = torch.where(target == 1)[0] #index of pos
        neg_idex = torch.where(target == 0)[0] #index of neg
        
        loss[neg_idex] =  loss[neg_idex]*class_weight[0] #neg class weight * corresponding loss
        loss[pos_idex] =  loss[pos_idex]*class_weight[1]


        if self.reduction == 'mean':
            loss = loss.mean()

        if self.att_reg_flag == True:
            loss = loss + self.att_reg_loss(tumor_fractions, attention_scores)
        
        #Regularization
        l1_regularization = 0
        l2_regularization = 0
        for param in self.model.parameters():
            l1_regularization += param.abs().sum()
            l2_regularization += param.square().sum()
        if self.reg_type == "L1":
            loss = loss + self.lambda_coef*l1_regularization    
        elif self.reg_type == "L2":
            loss = loss + self.lambda_coef*l2_regularization 
        else:
            loss = loss 

        return loss

class BCE_Weighted_Reg_focal(nn.Module):
    def __init__(self, lambda_coef, reg_type, model, gamma = 2,reduction = "mean", att_reg_flag = False):
        super(BCE_Weighted_Reg_focal, self).__init__()
        self.lambda_coef = lambda_coef
        self.reg_type = reg_type
        self.model = model
        self.reduction = reduction
        self.att_reg_flag = att_reg_flag 
        self.gamma = gamma

        self.att_reg_loss = nn.MSELoss()

    def forward(self, output, target, class_weight, tumor_fractions, attention_scores):

        #Compute Focal Loss
        #https://medium.com/visionwizard/understanding-focal-loss-a-quick-read-b914422913e7
        loss = - (target * ((1 - output)**self.gamma) * torch.log(output) + (1-target)* (output**self.gamma) *torch.log(1-output))
        
        #Weight loss for each class
        pos_idex = torch.where(target == 1)[0] #index of pos
        neg_idex = torch.where(target == 0)[0] #index of neg
        
        loss[neg_idex] =  loss[neg_idex]*class_weight[0] #neg class weight * corresponding loss
        loss[pos_idex] =  loss[pos_idex]*class_weight[1]


        if self.reduction == 'mean':
            loss = loss.mean()

        if self.att_reg_flag == True:
            loss = loss + self.att_reg_loss(tumor_fractions, attention_scores)
        
        #Regularization
        l1_regularization = 0
        l2_regularization = 0
        for param in self.model.parameters():
            l1_regularization += param.abs().sum()
            l2_regularization += param.square().sum()
        if self.reg_type == "L1":
            loss = loss + self.lambda_coef*l1_regularization    
        elif self.reg_type == "L2":
            loss = loss + self.lambda_coef*l2_regularization 
        else:
            loss = loss 

        return loss

def compute_loss_for_all_labels(predicted_list, target_list, weight_list, loss_func_name, loss_function, device, tumor_fractions, attention_scores, mutation_type, all_selected_label):    
    loss_list = []
    for i in range(0,len(predicted_list)):
        if mutation_type == "MT":
            label_index = i
        else:
            label_index = all_selected_label.index(mutation_type)

        if loss_func_name == "BCE_Weighted_Reg":
            cur_loss = loss_function(predicted_list[i],target_list[:,:,label_index].to(device),weight_list[label_index],tumor_fractions.squeeze().to(device), attention_scores.squeeze())
        elif loss_func_name == "BCE_Weighted_Reg_focal":
            cur_loss = loss_function(predicted_list[i],target_list[:,:,label_index].to(device),weight_list[label_index],tumor_fractions.squeeze().to(device), attention_scores.squeeze())
        elif loss_func_name == "BCELoss": 
            cur_loss = loss_function(predicted_list[i],target_list[:,:,label_index].to(device)) 
        loss_list.append(cur_loss) #compute loss
    #Sum loss for all labels
    loss = sum(loss_list)

    return loss



def compute_loss_for_all_labels_sepatt(predicted_list, target_list, weight_list, loss_func_name, loss_function, device, tumor_fractions, attention_scores, mutation_type, all_selected_label):    
    loss_list = []
    for i in range(0,len(predicted_list)):
        if mutation_type == "MT":
            label_index = i
        else:
            label_index = all_selected_label.index(mutation_type)

        if loss_func_name == "BCE_Weighted_Reg":
            cur_loss = loss_function(predicted_list[i],target_list[:,:,label_index].to(device),weight_list[label_index],tumor_fractions.squeeze().to(device), attention_scores[i].squeeze())
        elif loss_func_name == "BCE_Weighted_Reg_focal":
            cur_loss = loss_function(predicted_list[i],target_list[:,:,label_index].to(device),weight_list[label_index],tumor_fractions.squeeze().to(device), attention_scores[i].squeeze())
        elif loss_func_name == "BCELoss": 
            cur_loss = loss_function(predicted_list[i],target_list[:,:,label_index].to(device)) 
        loss_list.append(cur_loss) #compute loss
    #Sum loss for all labels
    loss = sum(loss_list)

    return loss


class FocalLoss_withATT(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss_withATT, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.att_reg_flag = True
        self.att_reg_loss = nn.MSELoss()

    def forward(self, inputs, targets, tumor_fractions, attention_scores):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            F_loss =  F_loss.mean()
        elif self.reduction == 'sum':
            F_loss =  F_loss.sum()

        if self.att_reg_flag == True:
            attention_scores_mean = torch.softmax(attention_scores, dim=-1).mean(dim = 1) #Take the mean across all braches
            F_loss = F_loss + self.att_reg_loss(tumor_fractions, attention_scores)

        return F_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, reduction='mean'):
        r'''
        if alpha = -1, gamma = 0, then it is = CE loss
        '''
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits, target):
        
        if not (0 <= self.alpha <= 1) and self.alpha != -1:
            raise ValueError(f"Invalid alpha value: {self.alpha}. alpha must be in the range [0,1] or -1 for ignore.")

        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pred = pred_logits.sigmoid()
        pt = torch.where(target == 1, pred, 1 - pred)
        loss =  ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha != -1:
            alpha_t = target*self.alpha + (1.0 - target)*(1.0 - self.alpha)
            loss = alpha_t*loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

#This one is for muti-class, where pt = exp(...)
class FocalLossv2(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
        
def get_partial_data(indata, selected_ids):
    r'''
    Input:   Model Ready data pool, List of selected_ids
    Returns: Model ready data for selected Ids
    -------
    '''
    #Get ordered patient IDs in indata
    ids_pool  = [x[-1] for x in indata] #The 2nd to the last one is sample ID, the last one is patient ID

    #Find all index of the selected ids
    inc_idx = [i for x in selected_ids for i, val in enumerate(ids_pool) if val == x]

    #Subsets
    indata_subset = Subset(indata, inc_idx)
    
    #Get final 
    sp_ids_order =  [x[-2] for x in indata_subset] #sample IDs
    pt_ids_order =  [x[-1] for x in indata_subset] #patient IDs

    return indata_subset,sp_ids_order, pt_ids_order



def get_train_test_val_data(data_pool_train, data_pool_test, id_df, fold):

    #Get train, test IDs
    train_ids_pt = list(id_df.loc[id_df['FOLD' + str(fold)] == 'TRAIN', 'PATIENT_ID'])
    test_ids_pt  = list(id_df.loc[id_df['FOLD' + str(fold)] == 'TEST', 'PATIENT_ID'])
    val_ids_pt   = list(id_df.loc[id_df['FOLD' + str(fold)] == 'VALID', 'PATIENT_ID'])
   
    train_data_final, train_sp_ids_final, train_pt_ids_final = get_partial_data(data_pool_train, train_ids_pt)
    test_data_final, test_sp_ids_final, test_pt_ids_final = get_partial_data(data_pool_test, test_ids_pt)
    val_data_final, val_sp_ids_final, val_pt_ids_final = get_partial_data(data_pool_train, val_ids_pt)
    
    print(f'Sample N: Train: {len(set(train_sp_ids_final))}; Test: {len(set(test_sp_ids_final))}; Val: {len(set(val_sp_ids_final))}')
    print(f'Patient N: Train: {len(set(train_pt_ids_final))}; Test: {len(set(test_pt_ids_final))}; Val: {len(set(val_pt_ids_final))}')
    print(f'dataset N: Train: {len(train_data_final)}; Test: {len(test_data_final)}; Val: {len(val_data_final)}')

    return (train_data_final, train_sp_ids_final), (val_data_final, val_sp_ids_final), (test_data_final, test_sp_ids_final)

    



def update_to_agg_feature(indata):
            
    indata_final = [(x[0].mean(dim = 0).unsqueeze(0), x[1], x[2], x[3],x[4],x[5]) for x in indata]
    
    return indata_final
        

def concate_agg_feature(indata, cohort_name):
            
    indata_feature = [(x[0].mean(dim = 0).unsqueeze(0)) for x in indata]
    indata_final = np.array(torch.concat(indata_feature))
    indata_y = np.array([cohort_name] * indata_final.shape[0])
    
    return indata_final, indata_y

class ModelReadyData_clustering(Dataset):
    def __init__(self,
                 matrix_list,
                 label_list,
                 sp_id_list,
                 pt_id_list
                ):

        #Get feature
        self.x =  [torch.FloatTensor(m) for m in matrix_list]
        
        # Get the Y labels
        self.y =  [torch.FloatTensor(m) for m in label_list]
        
        # Get the IDs
        self.sp_id =  [m for m in sp_id_list]
        self.pt_id =  [m for m in pt_id_list]

        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x  = self.x[index]
        y  = self.y[index]

        sp_id = self.sp_id[index]
        pt_id = self.pt_id[index]
        
        return x, y, sp_id, pt_id


   

def plot_cluster_image(cluster_matrix, plot_tile, plot_outdir, colorbar = True):
    # --- Define discrete colormap ---
    colors = ['whitesmoke', 'silver', 'steelblue', 'darkorange', 'tab:red']  # -2, -1, 0, 1, 2
    cmap = ListedColormap(colors)
    bounds = [-2.5,-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # --- Plot ---
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cluster_matrix.T, cmap=cmap, norm=norm, origin='lower')
    
    # --- Colorbar with labels ---
    if colorbar:
        cbar = plt.colorbar(im, ticks=[-2,-1, 0, 1, 2])
        cbar.ax.set_yticklabels(['Non-Tissue', 'Non-Cancer' ,'Cluster 0', 'Cluster 1', 'Cluster 2'])
        cbar.set_label('Cluster Label')
    
    plt.title('Cluster Heatmap')
    plt.xlabel('Tile X')
    plt.ylabel('Tile Y')
    plt.grid(False)
    plt.savefig(plot_outdir + plot_tile + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def get_cluster_image_feature(cluster_assigment_df):
    # Extract X and Y directly
    cluster_assigment_df['X'] = cluster_assigment_df['TILE_XY_INDEXES'].apply(lambda s: int(s.strip('()').split(',')[0]))
    cluster_assigment_df['Y'] = cluster_assigment_df['TILE_XY_INDEXES'].apply(lambda s: int(s.strip('()').split(',')[1]))
    

    # Determine matrix size
    max_x = cluster_assigment_df['X'].max()
    max_y = cluster_assigment_df['Y'].max()
    
    # Create matrix and fill with cluster labels
    cluster_matrix = np.full((max_x + 1, max_y + 1), -2)  # or use np.nan
    
    for _, row in cluster_assigment_df.iterrows():
        x, y = row['X'], row['Y']
        cluster_matrix[x, y] = row['CLUSTER']
        
    return cluster_matrix
        

def get_list_for_modelreadydata(targetd_data, targeted_ids, selected_labels, tumor_frac):
    matrix_list = []
    label_list = []
    sp_id_list = []
    pt_id_list = []
    ct = 0
    for pt in targeted_ids:
        if ct % 10 == 0: print(ct)
        ct += 1
        cur_df = targetd_data.loc[targetd_data['SAMPLE_ID'] == pt].copy()
        #change the cluster for non-tumor tiles code the cluster as -1
        cur_df.loc[cur_df['TUMOR_PIXEL_PERC']< tumor_frac,'CLUSTER'] = -1
        cur_matrix = get_cluster_image_feature(cur_df)
        matrix_list.append(cur_matrix)
        
        #Get label
        cur_label = cur_df[selected_labels].drop_duplicates().to_numpy()
        label_list.append(cur_label)
        
        #Get ids
        sp_id_list.append(cur_df['SAMPLE_ID'].unique().item())
        pt_id_list.append(cur_df['PATIENT_ID'].unique().item())
    
    return matrix_list, label_list, sp_id_list, pt_id_list
            

def predict_clustercnn(net, data_loader, criterion, device, n_task = 7):    
    y_pred = []
    y_true = []
    y_pred_prob = []
    # Set the network to evaluation mode
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels, sp_id, pt_id) in enumerate(data_loader, 1):
            images, labels = images.to(device), labels.to(device)

            slide_preds = net(images) 
            
            #Compute loss for each task, then sum
            pred_list = []
            pred_prob_list = []
            loss = 0
            for i in range(len(slide_preds)):
                loss += criterion(slide_preds[i], labels[:,:,i])
                pred_prob = torch.sigmoid(slide_preds[i]) #[BS, 1]
                pred = pred_prob.round() ##[BS, 1]
                pred_list.append(pred.squeeze().cpu().numpy())
                pred_prob_list.append(pred_prob.squeeze().cpu().numpy())
        
            y_pred.append(np.array(pred_list))
            y_true.append(list(labels.squeeze().cpu().numpy()))
            y_pred_prob.append(np.array(pred_prob_list))
            
        
        y_predprob_task = []
        y_pred_tasks = []
        y_true_tasks = []
        for k in range(7):
            y_pred_tasks.append([p[k] for p in y_pred])
            y_predprob_task.append([p[k] for p in y_pred_prob])
            y_true_tasks.append([p[k] for p in y_true])
            
    
    
    return y_pred_tasks, y_predprob_task, y_true_tasks



def update_label(indata, select_label_index):
    r'''
    Input:   Model Ready data
    Returns: Model ready data, List of ids
    -------
    '''
    
    #Update label
    indata_final = [(x[0], x[1][:,select_label_index], x[2], x[3],x[4],x[5]) for x in indata]

    print(f'DS: {len(indata_final)}')

    return indata_final



def get_train_test_val_data_singlecohort(data_dir, cohort_name , model_data, tumor_frac, s_fold):
    
    #data_dir = proj_dir + 'intermediate_data/3B_Train_TEST_IDS'
    
    #Load ID split data
    d_path =  os.path.join(data_dir, cohort_name ,'TFT' + str(tumor_frac))
    train_test_val_id_df = pd.read_csv(os.path.join(d_path, "train_test_split.csv"))
    train_test_val_id_df.rename(columns = {'TMB_HIGHorINTERMEDITATE': 'TMB'}, inplace = True)
    
    
    #Load data
    train_model_data = model_data['OL100']
    test_model_data  = model_data['OL0']

    #get train test data
    (train_data, train_ids), (val_data, val_ids), (test_data, test_ids) = get_train_test_val_data(train_model_data, test_model_data, train_test_val_id_df, s_fold)
    
    #add domain label:
    train_data = [item[:3] + (torch.tensor(1.0),) + item[3:] for item in train_data] #1 for OPX, biopsy, #0 for TCGA, surgical
    
    return (train_data, train_ids), (val_data, val_ids), (test_data, test_ids)





def random_sample_tiles(indata, k = 1000, random_seed = 42):
    random.seed(random_seed)          
    torch.manual_seed(random_seed)   
    
    for i in range(len(indata)):
        # Unpack the tuple
        features, labels, tf, domain_label, other_info, sample_id, patient_id = indata[i]
        

        num_tiles = features.size(0)
        sample_indices = random.sample(range(num_tiles), min(k, num_tiles)) # Ensure k does not exceed number of rows
        sample_indices.sort()
        sampled_feature = features[sample_indices]  
        sampled_tileinfo = other_info.iloc[sample_indices,].reset_index(drop=True)
        sampled_tf = tf[sample_indices]
        
        # Recreate the tuple with updated features
        indata[i] = (
            sampled_feature,
            labels,
            sampled_tf,
            sampled_tileinfo,
            sample_id,
            patient_id
        )
        
        
def get_larger_tumor_fraction_tile(tf1, tf2, match_index1, match_index2):
    
    tensor1 = tf1[match_index1]
    tensor2 = tf2[match_index2]
    
    
    mask_tensor1 = tensor1 >= tensor2
    
    # Get indices where tensor1 "wins"
    indices_tensor1 = mask_tensor1.nonzero(as_tuple=True)[0]
    
    # Get indices where tensor2 "wins"
    indices_tensor2 = (~mask_tensor1).nonzero(as_tuple=True)[0]
    
    final_idx1  = [match_index1[i] for i in indices_tensor1.tolist()]
    final_idx2  = [match_index2[i] for i in indices_tensor2.tolist()]
    
    return final_idx1, final_idx2


def get_matching_tile_index(indata, common_tiles_xy):
    # Get row indices in df1
    match_mask = indata['TILE_COOR_ATLV0'].isin(common_tiles_xy)
    df_match = indata[match_mask]
    df_nomatch = indata[~match_mask]
    match_index = df_match.index.tolist()
    nomatch_index = df_nomatch.index.tolist()
    
    return  match_index, nomatch_index




def combine_data_from_stnorm_and_nostnorm(indata_stnorm, indata_stnorm_no, method = 'union'):
    
    # indata_stnorm = data_ol100_opx_stnorm0
    # indata_stnorm_no = data_ol100_opx_stnorm1
    
    comb_data_list = []
    #get the data with more IDs
    if len(indata_stnorm) >= len(indata_stnorm_no):
        main_data = indata_stnorm
        nomain_data = indata_stnorm_no
    else:
        main_data = indata_stnorm_no
        nomain_data = indata_stnorm

    for i in range(len(main_data)):

        # Unpack the tuple
        features1, labels1, tf1, other_info1, sample_id1, patient_id1 = main_data[i]
        
        #find the data in indata2
        index = next((i for i, entry in enumerate(nomain_data) if entry[-2] == sample_id1), None)
        
    
        
        if index is not None: #if the ID is in both data, take the union or combine
            features2, labels2, tf2, other_info2, sample_id2, patient_id2 = nomain_data[index]
            if method == 'combine_all':
                comb_data = (
                    features1,
                    labels1,
                    tf1,
                    other_info1,
                    sample_id1,
                    patient_id1,
                    
                    features2,
                    labels2,
                    tf2,
                    other_info2,
                    sample_id2,
                    patient_id2,
                )
            elif method == 'union':
                #Intersect
                common_tiles = set(other_info1['TILE_COOR_ATLV0']).intersection(other_info2['TILE_COOR_ATLV0'])
                match_index1, nomatch_index1 = get_matching_tile_index(other_info1, common_tiles)
                match_index2, nomatch_index2 = get_matching_tile_index(other_info2, common_tiles)
                match_index1_tokeep, match_index2_tokeep = get_larger_tumor_fraction_tile(tf1, tf2,match_index1, match_index2)
                
                final_idx_tokeep1 = match_index1_tokeep + nomatch_index1
                final_idx_tokeep2 = match_index2_tokeep + nomatch_index2
                
                
                #Get updated info
                feature = torch.concat([features1[final_idx_tokeep1], features2[final_idx_tokeep2]])
                tf = torch.concat([tf1[final_idx_tokeep1], tf2[final_idx_tokeep2]])
                labels = labels1 
                other_info = pd.concat([other_info1.iloc[final_idx_tokeep1], other_info2.iloc[final_idx_tokeep2]])
                sample_id = sample_id1
                patient_id = patient_id1
                
                comb_data = (
                    feature,
                    labels,
                    tf,
                    other_info,
                    sample_id,
                    patient_id
                )

        else: #only take all the things in main data
            if method == 'combine_all':
                comb_data = (
                    features1,
                    labels1,
                    tf1,
                    other_info1,
                    sample_id1,
                    patient_id1,
                    [],
                    [],
                    [],
                    [],
                    [],
                    []
                    
                )
            elif method == 'union':
                 comb_data = (
                     features1,
                     labels1,
                     tf1,
                     other_info1,
                     sample_id1,
                     patient_id1
                 )
                 
        comb_data_list.append(comb_data)
        
    return comb_data_list   




def load_data(data_path):
    loaded = torch.load(data_path)
    data = loaded['data']
    ids =  loaded['id']
    return data, ids    


def load_model_ready_data(data_path, cohort, pixel_overlap, fe_method, tumor_frac):
    
    feature_path =  os.path.join(data_path, 
                                 cohort, 
                                 "IMSIZE250_OL" + str(pixel_overlap), 
                                 'feature_' + fe_method, 
                                 'TFT' + str(tumor_frac))
    try:
        model_ready_data = torch.load(os.path.join(feature_path, cohort + '_data.pth'))
    
    except FileNotFoundError:
        model_ready_data = None
    
    return model_ready_data

def get_cohort_data(data_path, cohort_name, fe_method, tumor_frac):
    
    #Load data
    data_ol100 = load_model_ready_data(data_path, cohort_name, 100, fe_method, tumor_frac) #overlap 100
    data_ol0   = load_model_ready_data(data_path, cohort_name, 0, fe_method, tumor_frac) #overlap 0
    
    #Combine
    data_comb = {'OL100': data_ol100, 'OL0': data_ol0}
    
    return data_comb


def get_final_split_data(id_data_dir, train_cohort1, model_data1, train_cohort2, model_data2, tumor_frac, s_fold):
    

    (train_data1, train_ids1), (val_data1, val_ids1), (test_data1, test_ids1) = get_train_test_val_data_singlecohort(id_data_dir, 
                                                                                                               train_cohort1 ,
                                                                                                               model_data = model_data1, 
                                                                                                               tumor_frac = tumor_frac, 
                                                                                                               s_fold = s_fold)
    
    
    if train_cohort2 is not None:
        (train_data2, train_ids2), (val_data2, val_ids2), (test_data2, test_ids2) = get_train_test_val_data_singlecohort(id_data_dir, 
                                                                                                                   train_cohort2 ,
                                                                                                                   model_data = model_data2, 
                                                                                                                   tumor_frac = tumor_frac, 
                                                                                                                   s_fold = s_fold)
    
    else:
        (train_data2, train_ids2), (val_data2, val_ids2), (test_data2, test_ids2) = (None, None), (None, None), (None, None)
    
    
    train_data = train_data1 + train_data2
    train_ids = train_ids1 + train_ids2
    
    val_data = val_data1 + val_data2
    val_ids = val_ids1 + val_ids2
    
    test_data = test_data1 + test_data2 
    test_ids = test_ids1 + test_ids2
    
    return {
            "train": (train_data, train_ids),
            "val": (val_data, val_ids),
            "test": (test_data, test_ids),
            "test1": (test_data1, test_ids1),
            "test2": (test_data2, test_ids2)
            }



def get_final_model_data(data_dir, id_data_dir, train_cohort, mutation, fe_method, tumor_frac, s_fold):
    #OPX data
    data_opx_stnorm0 = get_cohort_data(data_dir, "z_nostnorm_OPX", fe_method, tumor_frac)
    data_opx_stnorm1 = get_cohort_data(data_dir, "OPX", fe_method, tumor_frac) #TODO: Check OPX_001 was removed beased no cancer detected in stnormed


    #TCGA data
    data_tcga_stnorm0 = get_cohort_data(data_dir, "z_nostnorm_TCGA_PRAD", fe_method, tumor_frac)
    data_tcga_stnorm1 = get_cohort_data(data_dir, "TCGA_PRAD", fe_method, tumor_frac)

    
    #Neptune
    data_nep_stnorm0 = get_cohort_data(data_dir, "z_nostnorm_Neptune", fe_method, tumor_frac)
    data_nep_stnorm1 = get_cohort_data(data_dir, "Neptune", fe_method, tumor_frac)
    

    ################################################
    # Combine stnorm and nostnorm 
    ################################################
    #OPX
    data_ol100_opx_union = combine_data_from_stnorm_and_nostnorm(data_opx_stnorm0['OL100'], data_opx_stnorm1['OL100'], method = 'union')
    data_ol0_opx_union   = combine_data_from_stnorm_and_nostnorm(data_opx_stnorm0['OL0'], data_opx_stnorm1['OL0'], method = 'union')
    data_opx_stnorm10_union = {'OL100': data_ol100_opx_union, 'OL0': data_ol0_opx_union}

    data_ol100_opx_comb = combine_data_from_stnorm_and_nostnorm(data_opx_stnorm0['OL100'],  data_opx_stnorm1['OL100'], method = 'combine_all')
    data_ol0_opx_comb = combine_data_from_stnorm_and_nostnorm(data_opx_stnorm0['OL0'], data_opx_stnorm1['OL0'], method = 'combine_all')
    data_opx_stnorm10_comb = {'OL100': data_ol100_opx_comb, 'OL0': data_ol0_opx_comb}

    #TCGA
    data_ol100_tcga_union = combine_data_from_stnorm_and_nostnorm(data_tcga_stnorm0['OL100'], data_tcga_stnorm1['OL100'], method = 'union')
    data_ol0_tcga_union = combine_data_from_stnorm_and_nostnorm(data_tcga_stnorm0['OL0'], data_tcga_stnorm0['OL0'], method = 'union')
    data_tcga_stnorm10_union = {'OL100': data_ol100_tcga_union, 'OL0': data_ol0_tcga_union}

    data_ol100_tcga_comb = combine_data_from_stnorm_and_nostnorm(data_tcga_stnorm0['OL100'], data_tcga_stnorm1['OL100'], method = 'combine_all')
    data_ol0_tcga_comb = combine_data_from_stnorm_and_nostnorm(data_tcga_stnorm0['OL0'], data_tcga_stnorm0['OL0'], method = 'combine_all')
    data_tcga_stnorm10_comb = {'OL100': data_ol100_tcga_comb, 'OL0': data_ol0_tcga_comb}
    
    #NEP
    data_ol0_nep_union = combine_data_from_stnorm_and_nostnorm(data_nep_stnorm0['OL0'], data_nep_stnorm1['OL0'], method = 'union')
    data_nep_stnorm10_union = {'OL100': None, 'OL0': data_ol0_nep_union}
    data_ol0_nep_comb = combine_data_from_stnorm_and_nostnorm(data_nep_stnorm0['OL0'], data_nep_stnorm1['OL0'], method = 'combine_all')
    data_nep_stnorm10_comb = {'OL100': None, 'OL0': data_ol0_nep_comb}
    
    ################################################
    #Get Train, test, val data
    ################################################    
    train_cohort_map = {
        'OPX': {
            'train_cohort1': 'OPX',
            'model_data1': data_opx_stnorm1,
            'train_cohort2': None,
            'model_data2': None
        },
        'TCGA': {
            'train_cohort1': 'TCGA_PRAD',
            'model_data1': data_tcga_stnorm1,
            'train_cohort2': None,
            'model_data2': None
        },
        'z_nostnorm_OPX': {
            'train_cohort1': 'OPX',
            'model_data1': data_opx_stnorm0,
            'train_cohort2': None,
            'model_data2': None
        },
        'z_nostnorm_TCGA_PRAD': {
            'train_cohort1': 'TCGA_PRAD',
            'model_data1': data_tcga_stnorm0,
            'train_cohort2': None,
            'model_data2': None
        },
        'z_nostnorm_OPX_TCGA': {
            'train_cohort1': 'z_nostnorm_OPX',
            'model_data1': data_opx_stnorm0,
            'train_cohort2': 'z_nostnorm_TCGA_PRAD',
            'model_data2': data_tcga_stnorm0
        },
        'OPX_TCGA': {
            'train_cohort1': 'OPX',
            'model_data1': data_opx_stnorm1,
            'train_cohort2': 'TCGA_PRAD',
            'model_data2': data_tcga_stnorm1
        },
        'union_STNandNSTN_OPX_TCGA': {
            'train_cohort1': 'OPX',
            'model_data1': data_opx_stnorm10_union,
            'train_cohort2': 'TCGA_PRAD',
            'model_data2': data_tcga_stnorm10_union
        },
        'comb_STNandNSTN_OPX_TCGA': {
            'train_cohort1': 'OPX',
            'model_data1': data_opx_stnorm10_comb,
            'train_cohort2': 'TCGA_PRAD',
            'model_data2': data_tcga_stnorm10_comb
        }
    }
    
    if train_cohort in train_cohort_map:
        config = train_cohort_map[train_cohort]
        train_cohort1 = config['train_cohort1']
        model_data1 = config['model_data1']
        train_cohort2 = config['train_cohort2']
        model_data2 = config['model_data2']
    else:
        raise ValueError(f"Unknown training cohort: {train_cohort}")

    
    
    #Get Train, validation and test
    split_data = get_final_split_data(id_data_dir,
                                    train_cohort1,
                                    model_data1, 
                                    train_cohort2, 
                                    model_data2, 
                                    tumor_frac, 
                                    s_fold)
    
    train_data, _ = split_data["train"]
    val_data, _ = split_data["val"]
    test_data, _ = split_data["test"]
    test_data1, _ = split_data["test1"]
    test_data2, _ = split_data["test2"]
    
    ext_data_st0 =  data_nep_stnorm0['OL0']
    ext_data_st1 =  data_nep_stnorm1['OL0']
    ext_data_union =  data_nep_stnorm10_union['OL0']
    
    
    #Get sanple ID:
    train_ids =  [x[-2] for x in train_data]
    val_ids =  [x[-2] for x in val_data]
    test_ids =  [x[-2] for x in test_data]
    test_ids1 =  [x[-2] for x in test_data1]
    test_ids2 =  [x[-2] for x in test_data2]
    nep_ids0 =  [x[-2] for x in ext_data_st0]
    nep_ids1 =  [x[-2] for x in ext_data_st1]
    nep_ids =  [x[-2] for x in ext_data_union]
    
    
    #Update labels
    selected_labels, selected_label_index = get_selected_labels(mutation, train_cohort)
    print(selected_labels)
    print(selected_label_index)

    train_data = update_label(train_data, selected_label_index)
    val_data = update_label(val_data, selected_label_index)
    test_data = update_label(test_data, selected_label_index)
    test_data1 = update_label(test_data1, selected_label_index)
    test_data2 = update_label(test_data2, selected_label_index)
    ext_data_st0 = update_label(ext_data_st0, selected_label_index)
    ext_data_st1 = update_label(ext_data_st1, selected_label_index)
    ext_data_union = update_label(ext_data_union, selected_label_index)
    
    
    #update item for model
            
    if train_cohort == 'comb_STNandNSTN_OPX_TCGA': 
        #Keep feature1, label, tf,1 dlabel, feature2, tf2
        train_data = [(item[0], item[1], item[2], item[3], item[7], item[9]) for item in train_data]
        test_data1 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data1]
        test_data2 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data2]
        test_data = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data]
        val_data = [(item[0], item[1], item[2],  item[6], item[8]) for item in val_data]
    else:
        #Exclude tile info data, sample ID, patient ID, do not needed it for training
        train_data = [item[:-3] for item in train_data]
        test_data1 = [item[:-3] for item in test_data1]   
        test_data2 = [item[:-3] for item in test_data2]   
        test_data = [item[:-3] for item in test_data]   
        val_data = [item[:-3] for item in val_data]
    
    ext_data_st0 = [item[:-3] for item in ext_data_st0] #no st norm
    ext_data_st1 = [item[:-3] for item in ext_data_st1] #st normed
    ext_data_union = [item[:-3] for item in ext_data_union]

    
    return {'train': (train_data, train_ids),
            'val': (val_data, val_ids),
            'test': (test_data, test_ids),
            'test1': (test_data1, test_ids1),
            'test2': (test_data2, test_ids2),
            'ext_data_st0': (ext_data_st0, nep_ids0),
            'ext_data_st1': (ext_data_st1, nep_ids1),
            'ext_data_union': (ext_data_union, nep_ids)
        }, selected_labels