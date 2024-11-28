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


def convert_img(in_img):
    srgb_profile = ImageCms.createProfile("sRGB")
    converted_img = ImageCms.profileToProfile(in_img, srgb_profile, srgb_profile)

    return converted_img

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



def get_feature_label_array_dynamic(input_path, feature_name, selected_ids,selected_labels, selected_feature, tumor_fraction_thres = 0, sample_size = 0, upsampling = False):
    feature_list = []
    label_list = []
    info_list = []
    tumor_info_list = []
    ct = 0 
    for pt in selected_ids:
        if ct % 10 == 0 : print(ct)
        input_dir = input_path + pt + '/' + 'features/' + feature_name + '.h5'
        
        #feature
        cur_feature_df = pd.read_hdf(input_dir, key='feature')
        cur_feature_df.columns = cur_feature_df.columns.astype(str)
        
        # #Label
        cur_label_df = pd.read_hdf(input_dir, key='tile_info')
        cur_label_df.reset_index(drop = True, inplace = True)
        
        #Add tumor info to label
        cur_tumor_info_df = pd.read_csv(os.path.join(input_path,pt,"ft_model/",pt + "_TILE_TUMOR_PERC.csv"))
        cur_tumor_info_df.reset_index(drop = True, inplace = True)
        cur_label_df = cur_label_df.merge(cur_tumor_info_df, on = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP',
                                                                   'LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE',
                                                                   'TISSUE_COVERAGE'])
        
        #Combine feature and label and tumor info
        cur_comb_df = pd.concat([cur_feature_df,cur_label_df], axis = 1)
        
        #Select tumor fraction > X tiles
        cur_comb_df = cur_comb_df.loc[cur_comb_df['TUMOR_PIXEL_PERC'] > tumor_fraction_thres]
        cur_comb_df = cur_comb_df.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
        
        #if non-tumor_fraction tiles, exclude the sample 
        if cur_comb_df.shape[0] != 0:
            #random select
            if sample_size == 0:
                cur_comb_df  = cur_comb_df
            elif sample_size > 0:
                if cur_comb_df.shape[0] > sample_size:
                    cur_comb_df = cur_comb_df.iloc[0:sample_size,]
                else:
                    if upsampling == True:
                        cur_comb_df = cur_comb_df.sample(n=sample_size, replace=True, random_state=42)
                    else:
                        cur_comb_df = cur_comb_df
            cur_comb_df = cur_comb_df.reset_index(drop = True)
    
            #Extract feature, label and tumor info
            cur_feature = cur_comb_df[selected_feature].values #np array
            cur_label   = cur_comb_df[selected_labels].drop_duplicates().values.astype('float32') #numpy array
            cur_info  =  cur_comb_df[['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP', 'LIMIT_BOUNDS', 
                                      'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE', 'TISSUE_COVERAGE',
                                      'Bx Type', 'Anatomic site', 'Notes', 'SITE_LOCAL', 'pred_map_location', 'TUMOR_PIXEL_PERC']]
            cur_tf_info = cur_comb_df['TUMOR_PIXEL_PERC'].values
        
            feature_list.append(cur_feature)
            label_list.append(cur_label)
            info_list.append(cur_info)
            tumor_info_list.append(cur_tf_info)
            ct += 1
        

    return feature_list, label_list, info_list, tumor_info_list



def get_feature_label_array_dynamic_nonoverlaptest(input_path, feature_name, selected_ids,selected_labels, selected_feature):
    feature_list = []
    label_list = []
    info_list = []
    for pt in selected_ids:
        input_dir = input_path + pt + '/' + 'features/' + feature_name + '.h5'
    
        #feature
        cur_feature_df = pd.read_hdf(input_dir, key='feature')
        cur_feature_df.columns = cur_feature_df.columns.astype(str)
        cur_feature = cur_feature_df.values #np array
    
        #Label
        cur_label_info_df =  pd.read_hdf(input_dir, key='tile_info')
        cur_label_df = cur_label_info_df[["SAMPLE_ID"] + selected_labels]
        cur_label = cur_label_df.drop_duplicates(subset= ['SAMPLE_ID'])
        cur_label = cur_label.drop(columns='SAMPLE_ID').values.astype('float32') #numpy array
        cur_label.shape

        #Info
        cur_info  =  cur_label_info_df[['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP', 'LIMIT_BOUNDS', 
                                        'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE', 'TISSUE_COVERAGE',
                                        'Bx Type', 'Anatomic site', 'Notes', 'SITE_LOCAL']]
    
        feature_list.append(cur_feature)
        label_list.append(cur_label)
        info_list.append(cur_info)

    return feature_list, label_list, info_list




class ModelReadyData_diffdim(Dataset):
    def __init__(self,
                 feature_list,
                 label_list,
                ):
        
        self.x =[torch.FloatTensor(feature) for feature in feature_list] 
        
        # Get the Y labels
        self.y = [torch.FloatTensor(label) for label in label_list] 
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x = self.x[index]
        y = self.y[index]
        
        return x, y


def prediction(in_dataloader, in_model, n_label, loss_function, device, attention = True):
    in_model.eval()
    with torch.no_grad():
        running_loss = 0

        pred_prob_list = []
        y_true_list = []
        att_list = []
        for x,y in in_dataloader:

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



def BCE_WithRegularization(output, target, lambda_coef, reg_type, model, class_weight,reduction = 'mean'):
    

    #Compute loss for each sample
    loss = - ( target * torch.log(output) + (1-target)*torch.log(1-output))
    
    #Weight loss for each class
    pos_idex = torch.where(target == 1)[0] #index of pos
    neg_idex = torch.where(target == 0)[0] #index of neg
    
    loss[neg_idex] =  loss[neg_idex]*class_weight[0] #neg class weight * correponding loss
    loss[pos_idex] =  loss[pos_idex]*class_weight[1]

    
    if reduction == 'mean':
        loss = loss.mean()
    #Regularization
    l1_regularization = 0
    l2_regularization = 0
    for param in model.parameters():
        l1_regularization += param.abs().sum()
        l2_regularization += param.square().sum()
    if reg_type == "L1":
        loss = loss + lambda_coef*l1_regularization    
    elif reg_type == "L2":
        loss = loss + lambda_coef*l2_regularization 
    else:
        loss = loss 
       
    return loss
