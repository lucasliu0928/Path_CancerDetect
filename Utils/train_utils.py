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




def combine_feature_label_tumorinfo(patient_id, input_path, input_file_name):

    #Input dir
    input_dir = input_path + patient_id + '/' + 'features/' + input_file_name + '.h5'

    #feature
    feature_df = pd.read_hdf(input_dir, key='feature')
    feature_df.columns = feature_df.columns.astype(str)
    
    #Label
    label_df = pd.read_hdf(input_dir, key='tile_info')
    label_df.reset_index(drop = True, inplace = True)
    
    
    #Add tumor info to label
    tumor_info_df = pd.read_csv(os.path.join(input_path, patient_id, "ft_model/", patient_id + "_TILE_TUMOR_PERC.csv"))
    tumor_info_df.reset_index(drop = True, inplace = True)
    label_df = label_df.merge(tumor_info_df, on = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP',
                                                   'LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE',
                                                   'TISSUE_COVERAGE'])
    
    #Combine feature and label and tumor info
    comb_df = pd.concat([feature_df,label_df], axis = 1)

    return comb_df


def extract_feature_label_tumorinfo_np(selected_df, selected_feature, selected_labels):
    #Extract feature, label and tumor info
    feature_np = selected_df[selected_feature].values #np array
    label_np   = selected_df[selected_labels].drop_duplicates().values.astype('float32') #numpy array
    info_np    = selected_df[['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP', 'LIMIT_BOUNDS', 
                               'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE', 'TISSUE_COVERAGE',
                               'Bx Type', 'Anatomic site', 'Notes', 'SITE_LOCAL', 'pred_map_location', 'TUMOR_PIXEL_PERC']]
    tf_info_np = selected_df['TUMOR_PIXEL_PERC'].values

    return feature_np, label_np, info_np, tf_info_np


def get_feature_label_array_dynamic(input_path, feature_name, selected_ids,selected_labels, selected_feature, train_or_test, train_sample_size = 'ALL_TUMOR_TILES', tumor_fraction_thres = 0):
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
        cur_comb_df = combine_feature_label_tumorinfo(pt, input_path, feature_name)
        
        #Select tumor fraction > X tiles
        cur_comb_df_tumor = cur_comb_df.loc[cur_comb_df['TUMOR_PIXEL_PERC'] > tumor_fraction_thres].copy()
        cur_comb_df_tumor = cur_comb_df_tumor.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
        cur_n_tumor_tiles = cur_comb_df_tumor.shape[0] # N of tumor tiles

        
        if train_or_test == 'Test':
            if cur_n_tumor_tiles > 0: #if any tumor tiles
                cur_selected_df =  cur_comb_df_tumor  #select all tumor tiles
            else:
                cur_selected_df =  cur_comb_df #select all other tiles
            cur_selected_df = cur_selected_df.reset_index(drop = True)
        
        elif train_or_test == 'Train':
            if cur_n_tumor_tiles > 0: #only include samples has tumor tiles                
                if train_sample_size == 'ALL_TUMOR_TILES':
                    cur_selected_df  = cur_comb_df_tumor #select all tumor tiles
                elif train_sample_size > 0:
                    if cur_n_tumor_tiles > train_sample_size:
                        cur_selected_df = cur_comb_df_tumor.iloc[0:train_sample_size,] #select top tumor tiles
                    else:
                        cur_selected_df = cur_comb_df_tumor #select all tumor tiles
                cur_selected_df = cur_selected_df.reset_index(drop = True)
            else:
                cur_selected_df = None



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




class ModelReadyData_diffdim(Dataset):
    def __init__(self,
                 feature_list,
                 label_list,
                 tumor_info_list,
                ):
        
        self.x =[torch.FloatTensor(feature) for feature in feature_list] 
        
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


def prediction(in_dataloader, in_model, n_label, loss_function, device, attention = True):
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
                cur_loss = loss_function(yhat_list[i],y[:,:,i].to(device))  #compute loss
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



def compute_loss_for_all_labels(predicted_list, target_list, weight_list, loss_func_name, loss_function, device, tumor_fractions, attention_scores):    
    loss_list = []
    for i in range(0,len(predicted_list)):f
        if loss_func_name == "BCE_Weighted_Reg":
            cur_loss = loss_function(predicted_list[i],target_list[:,:,i].to(device),weight_list[i],tumor_fractions.squeeze().to(device), attention_scores.squeeze())
        elif loss_func_name == "BCELoss": 
            cur_loss = loss_function(predicted_list[i],target_list[:,:,i].to(device), tumor_fractions.squeeze().to(device), attention_scores.squeeze()) 
        loss_list.append(cur_loss) #compute loss
    #Sum loss for all labels
    loss = sum(loss_list)

    return loss
