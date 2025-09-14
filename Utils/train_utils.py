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
from misc_utils import convert_img
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import random
from data_loader import load_model_ready_data
import re
 
def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False



        


def has_seven_csv_files(folder_path):
    # List all files ending with .csv
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and os.path.isfile(os.path.join(folder_path, f))]
    return len(csv_files) == 6



    
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
        self.alpha = alpha #alpha is for the class = 1
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits, target):
        """
        pred_logits: [N, C] raw logits (e.g., [N, 2])
        target: [N] or [N,1] with values in {0,...,C-1}
        """
        
        if not (0 <= self.alpha <= 1) and self.alpha != -1:
            raise ValueError(f"Invalid alpha value: {self.alpha}. alpha must be in the range [0,1] or -1 for ignore.")

        # if target.dim() > 1:
        #     target = target.squeeze(1)  # [N,1] -> [N]
        if target.dtype != torch.long:
            target = target.long()              # floats -> int class
    
        ce_loss = F.cross_entropy(pred_logits, target, reduction="none")
        pt = torch.exp(-ce_loss)  # pt = softmax prob of the true class
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


class FocalLoss_logitadj(nn.Module):
    def __init__(self, alpha=1, gamma=1, prior_prob = 0.04, tau = 2.0, reduction='mean'):
        r'''
        if alpha = -1, gamma = 0, then it is = CE loss
        '''
        super(FocalLoss_logitadj, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.prior_prob = prior_prob
        self.tau = tau

    def forward(self, pred_logits, target):
        
        if not (0 <= self.alpha <= 1) and self.alpha != -1:
            raise ValueError(f"Invalid alpha value: {self.alpha}. alpha must be in the range [0,1] or -1 for ignore.")
        
        # if target.dim() > 1:
        #     target = target.squeeze(1)  # [N,1] -> [N]
        if target.dtype != torch.long:
            target = target.long()              # floats -> int class
            
        # Compute logit adjustment term
        adjustment = self.tau * torch.log(torch.tensor(self.prior_prob))
        pred_logits_adjusted = pred_logits + (-adjustment)
        
        ce_loss = F.cross_entropy(pred_logits_adjusted, target, reduction="none")
        pt = torch.exp(-ce_loss)  # pt = softmax prob of the true class
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
        super(FocalLossv2, self).__init__()
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













def random_sample_tiles(indata, k = 1000, random_seed = 42):
    random.seed(random_seed)          
    torch.manual_seed(random_seed)   
    
    for i in range(len(indata)):
        # Unpack the tuple
        #features, labels, tf, domain_label, other_info, sample_id, patient_id = indata[i]
        features, labels, tf , sl = indata[i]

        num_tiles = features.size(0)
        sample_indices = random.sample(range(num_tiles), min(k, num_tiles)) # Ensure k does not exceed number of rows
        sample_indices.sort()
        sampled_feature = features[sample_indices]  
        #sampled_tileinfo = other_info.iloc[sample_indices,].reset_index(drop=True)
        sampled_tf = tf[sample_indices]
        sampled_sl = sl[sample_indices]
        
        # Recreate the tuple with updated features
        indata[i] = (
            sampled_feature,
            labels,
            sampled_tf,
            sampled_sl
            #sampled_tileinfo,
            #sample_id,
            #patient_id
        )
        
        











def load_data(data_path):
    loaded = torch.load(data_path)
    data = loaded['data']
    ids =  loaded['id']
    return data, ids    










    

