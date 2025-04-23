#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#NOTE: use paimg9 env
import sys
import os
import pandas as pd
import warnings
import torch
import argparse

sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists, count_label, set_seed
from train_utils import ModelReadyData_diffdim_V2, get_feature_idexes, get_sample_feature, get_sample_label, combine_feature_and_label
warnings.filterwarnings("ignore")



#Run: python3 -u 5_get_modelreadydata_OPX_TCGA.py --pixel_overlap 100 --cohort_name TCGA_PRAD

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--pixel_overlap', default=100, type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--save_image_size', default=250, type=int, help='the size of extracted tiles')
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--cohort_name', default='OPX', type=str, help='data set name: TAN_TMA_Cores or OPX or TCGA_PRAD')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')

args = parser.parse_args()

####################################
#Select label and feature index
####################################
SELECTED_LABEL = ["AR","HR1","HR2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]
SELECTED_FEATURE = get_feature_idexes(args.fe_method,include_tumor_fraction = False)

##################
###### DIR  ######
##################
folder_name = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.pixel_overlap)
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
info_path =   os.path.join(proj_dir,'intermediate_data','3A_otherinfo', args.cohort_name, folder_name)
feature_path = os.path.join(proj_dir,'intermediate_data','4_tile_feature', args.cohort_name, folder_name)


################################################
#Create output dir
################################################
outdir =  os.path.join(proj_dir + 'intermediate_data/5_model_ready_data',
                       args.cohort_name, 
                       folder_name, 
                       'feature_' + args.fe_method, 
                       'TFT' + str(args.TUMOR_FRAC_THRES))
create_dir_if_not_exists(outdir)

##################
#Select GPU
##################
device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')
print(device)
set_seed(0)


############################################################################################################
#Load all tile info df
#This file contains all tiles before cancer fraction exclusion 
#and  has tissue membership > 0.9, white space < 0.9 (non white space > 0.1)
############################################################################################################
all_tile_info_df = pd.read_csv(os.path.join(info_path, "all_tile_info.csv"))

if args.cohort_name == 'OPX':
    id_col = 'SAMPLE_ID'
elif args.cohort_name == 'TCGA_PRAD':
    id_col = 'TCGA_FOLDER_ID'

selected_ids = list(all_tile_info_df[id_col].unique())    
selected_ids.sort()

# info_path2 =   os.path.join(proj_dir,'intermediate_data','Old_5_model_ready_data', cohort_name, folder_name, "feature_retccl", "TFT0.9")
# all_tile_info_list = torch.load(info_path2 + "/OPX_info.pth")
# all_tile_info_df2 = pd.concat(all_tile_info_list)
# check_ids = list(all_tile_info_df2['SAMPLE_ID'].unique())

# check_df2 = all_tile_info_df.loc[all_tile_info_df['SAMPLE_ID'].isin(check_ids)]
# len(set(check_df2['SAMPLE_ID']))
# check_df3 = all_tile_info_df2.loc[all_tile_info_df2['SAMPLE_ID'].isin(list(set(check_df2['SAMPLE_ID'])))]

# check_df2 = check_df2.loc[check_df2['TUMOR_PIXEL_PERC'] > 0.9] #306566

#TODO: Double check this two, why not match
# print(all_tile_info_thres.shape) #930297 #This
# check = np.concatenate(info)     #927717 #tumor info list
# check.shape

############################################################################################################
#Get model ready data
############################################################################################################
comb_df_list = []
ct = 0 
for pt in selected_ids:
    if ct % 10 == 0 : print(ct)
    #Get feature
    feature_df = get_sample_feature(pt, feature_path, args.fe_method)    
    #Get label
    label_df = get_sample_label(pt,all_tile_info_df, id_col = id_col)
    
    #Merge feature and label
    comb_df = combine_feature_and_label(feature_df,label_df)
    
    #Select tumor fraction > X tiles
    comb_df = comb_df.loc[comb_df['TUMOR_PIXEL_PERC'] >= args.TUMOR_FRAC_THRES].copy()
    comb_df = comb_df.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False)
    comb_df = comb_df.sort_values(by = ['TILE_XY_INDEXES'], ascending = True)
    comb_df.reset_index(inplace = True, drop = True)
    comb_df_list.append(comb_df)
    ct += 1

all_comb_df = pd.concat(comb_df_list)

#Get model ready data
data = ModelReadyData_diffdim_V2(comb_df_list, SELECTED_FEATURE, SELECTED_LABEL)
torch.save(data, os.path.join(outdir, args.cohort_name + '_data.pth'))


############################################################################################################
#Count Distribution
############################################################################################################
#Tile level
counts1 = count_label(all_comb_df, SELECTED_LABEL, args.cohort_name + "_TILE")
sample_level_comb_df = all_comb_df.drop_duplicates(subset = ['SAMPLE_ID'])
counts2 = count_label(sample_level_comb_df, SELECTED_LABEL, args.cohort_name + "_SAMPLE")
patient_level_comb_df = all_comb_df.drop_duplicates(subset = ['PATIENT_ID'])
counts3 = count_label(patient_level_comb_df, SELECTED_LABEL, args.cohort_name + "_PTS")
#print(counts2)
counts = counts2.merge(counts1, left_index = True, right_index = True)
counts = counts.merge(counts3, left_index = True, right_index = True)
counts.to_csv(os.path.join(outdir, args.cohort_name + '_counts.csv'))


print(all_comb_df['TUMOR_PIXEL_PERC'].shape)