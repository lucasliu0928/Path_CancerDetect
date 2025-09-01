#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import os
import pandas as pd
import warnings
import torch
import argparse
import time

sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists
from Utils import set_seed
from train_utils import ModelReadyData_diffdim_V2, get_feature_idexes, combine_feature_and_label_listids
warnings.filterwarnings("ignore")



# source ~/.bashrc
# conda activate paimg9
# python3 -u 5_get_combined_data.py --cohort_name z_nostnorm_TCGA_PRAD --pixel_overlap 0

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Model ready data")
parser.add_argument('--pixel_overlap', default=100, type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--save_image_size', default=250, type=int, help='the size of extracted tiles')
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.9, type=float, help='tile tumor fraction threshold')
parser.add_argument('--cohort_name', default='z_nostnorm_Neptune', type=str, help='data set name: TAN_TMA_Cores or OPX or TCGA_PRAD or Neptune')
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
info_path =    os.path.join(proj_dir,'intermediate_data','3A_otherinfo',   args.cohort_name, folder_name, "TFT" + str(args.TUMOR_FRAC_THRES))
feature_path = os.path.join(proj_dir,'intermediate_data','4_tile_feature', args.cohort_name, folder_name)

################################################
#Create output dir
################################################
outdir =  os.path.join(proj_dir + 'intermediate_data/5_combined_data',
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

if 'OPX' in args.cohort_name or 'Neptune' in args.cohort_name:
    id_col = 'SAMPLE_ID'
elif 'TCGA_PRAD' in args.cohort_name:
    id_col = 'FOLDER_ID'

selected_ids = list(all_tile_info_df[id_col].unique())    
selected_ids.sort()

############################################################################################################
#Get model ready data
############################################################################################################
start_time = time.time()
comb_df, comb_df_list = combine_feature_and_label_listids(selected_ids, all_tile_info_df, feature_path, id_col, args.fe_method, args.TUMOR_FRAC_THRES)

#Get model ready data
data = ModelReadyData_diffdim_V2(comb_df_list, SELECTED_FEATURE, SELECTED_LABEL)
torch.save(data, os.path.join(outdir, args.cohort_name + '_data.pth'))

elapsed_time = (time.time() - start_time)/60
print(elapsed_time, "min")
    
    
# ############################################################################################################
# #Count Distribution
# ############################################################################################################
# #Tile level
# counts1 = count_label(all_comb_df, SELECTED_LABEL, args.cohort_name + "_TILE")
# sample_level_comb_df = all_comb_df.drop_duplicates(subset = ['SAMPLE_ID'])
# counts2 = count_label(sample_level_comb_df, SELECTED_LABEL, args.cohort_name + "_SAMPLE")
# patient_level_comb_df = all_comb_df.drop_duplicates(subset = ['PATIENT_ID'])
# counts3 = count_label(patient_level_comb_df, SELECTED_LABEL, args.cohort_name + "_PTS")
# #print(counts2)
# counts = counts2.merge(counts1, left_index = True, right_index = True)
# counts = counts.merge(counts3, left_index = True, right_index = True)
# counts.to_csv(os.path.join(outdir, args.cohort_name + '_counts.csv'))


# print(all_comb_df['TUMOR_PIXEL_PERC'].shape)
