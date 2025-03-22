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
import os
from PIL import ImageCms, Image
from Utils import convert_img
import torch.nn as nn
import torch.nn.functional as F

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


def get_sample_feature(patient_id, feature_path, input_file_name):
    #Input dir
    input_dir = os.path.join(feature_path, patient_id, 'features', 'features_alltiles_' + input_file_name + '.h5')
    
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
    
def get_sample_label(patient_id, all_label_data):
    label_df = all_label_data.loc[all_label_data['SAMPLE_ID'] == patient_id]
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
            raise ValueError(f"Invalid alpha value: {alpha}. alpha must be in the range [0,1] or -1 for ignore.")

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


