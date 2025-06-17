#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:03:04 2025

@author: jliu6
"""

import os
import pandas as pd

def load_all_tile_df(full_path):
    
    all_data = []
    for folder in os.listdir(full_path):
        folder_path = os.path.join(full_path, folder, 'ft_model')
        
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith("_TILE_TUMOR_PERC.csv"):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)
                    df['source_folder'] = folder  # Optional: keep track of origin
                    all_data.append(df)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    
    tiles_counts_df = pd.DataFrame(combined_df['SAMPLE_ID'].value_counts())
    tiles_counts_df.reset_index(drop=False, inplace=True)
    tiles_counts_df.rename(columns = {'count': 'tile_n'}, inplace = True)
    
    return tiles_counts_df


# Define the root directory containing all the GAMMA_* folders
proj_dir = "/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/"
tile_dir_opx = proj_dir + '3A_otherinfo/z_nostnorm_OPX/IMSIZE250_OL0/'
tile_dir_tcga = proj_dir + '3A_otherinfo/z_nostnorm_TCGA_PRAD/IMSIZE250_OL0/'
tile_dir_nep = proj_dir + '3A_otherinfo/Neptune/IMSIZE250_OL0/'

all_tile_dir_opx = proj_dir + '2_cancer_detection/OPX/IMSIZE250_OL0/'
all_tile_dir_tcga = proj_dir + '2_cancer_detection/TCGA_PRAD/IMSIZE250_OL0/'
all_tile_dir_nep = proj_dir + '2_cancer_detection/Neptune/IMSIZE250_OL0/'


root_dir = proj_dir  + "/pred_out_050625/trainCohort_z_nostnorm_TCGA_OPXGRLFalse/acmil/uni2/TrainOL100_TestOL0_TFT0.9/"
root_dir = root_dir + 'FOLD4/MTHR_TYPEHR2/'
pred_dir = root_dir  + 'predictions/GAMMA_6_ALPHA_0.2/'

# #read all tile info
# all_tile_df_opx = load_all_tile_df(all_tile_dir_opx)
# all_tile_df_tcga = load_all_tile_df(all_tile_dir_tcga)
# all_tile_df_nep = load_all_tile_df(all_tile_dir_nep)




#read tile info
def tile_counts(full_path):
    
    file_path = full_path + 'all_tile_info.csv'
    indata =  pd.read_csv(file_path)
    
    cancer_tiles_counts_df = pd.DataFrame(indata['SAMPLE_ID'].value_counts())
    cancer_tiles_counts_df.reset_index(drop=False, inplace=True)
    cancer_tiles_counts_df.rename(columns = {'count': 'tile_n'}, inplace = True)
    
    return cancer_tiles_counts_df

tile_df_opx = pd.read_csv(tile_dir_opx + 'all_tile_info.csv')
tile_df_opx["TUMOR_PIXEL_PERC"] = pd.to_numeric(tile_df_opx["TUMOR_PIXEL_PERC"], errors="coerce")
tile_df_tcga = pd.read_csv(tile_dir_tcga + 'all_tile_info.csv')
tile_df_tcga["TUMOR_PIXEL_PERC"] = pd.to_numeric(tile_df_tcga["TUMOR_PIXEL_PERC"], errors="coerce")
tile_df_nep = pd.read_csv(tile_dir_nep + 'all_tile_info.csv')
tile_df_nep["TUMOR_PIXEL_PERC"] = pd.to_numeric(tile_df_nep["TUMOR_PIXEL_PERC"], errors="coerce")

# Convert column to numeric (just in case there are strings)
cancer_tile_counts_df_opx = pd.DataFrame(tile_df_opx[tile_df_opx["TUMOR_PIXEL_PERC"] > 0.9].groupby("PATIENT_ID").size())
cancer_tile_counts_df_opx.reset_index(drop=False, inplace=True)
cancer_tile_counts_df_opx.rename(columns = {0: 'cancer_tile_n'}, inplace = True)
    
cancer_tile_counts_df_tcga = pd.DataFrame(tile_df_tcga[tile_df_tcga["TUMOR_PIXEL_PERC"] > 0.9].groupby("PATIENT_ID").size())
cancer_tile_counts_df_tcga.reset_index(drop=False, inplace=True)
cancer_tile_counts_df_tcga.rename(columns = {0: 'cancer_tile_n'}, inplace = True)

cancer_tile_counts_df_nep = pd.DataFrame(tile_df_nep[tile_df_nep["TUMOR_PIXEL_PERC"] > 0.9].groupby("PATIENT_ID").size())
cancer_tile_counts_df_nep.reset_index(drop=False, inplace=True)
cancer_tile_counts_df_nep.rename(columns = {0: 'cancer_tile_n'}, inplace = True)

tile_counts_df_opx = tile_counts(tile_dir_opx)
tile_counts_df_tcga = tile_counts(tile_dir_tcga)
tile_counts_df_nep = tile_counts(tile_dir_nep)

##################################################################
merged_df_opx = pd.merge(tile_counts_df_opx, cancer_tile_counts_df_opx, 
                     left_on='SAMPLE_ID', right_on='PATIENT_ID')
merged_df_opx['cancer_tile_percentage'] = (merged_df_opx['cancer_tile_n'] / merged_df_opx['tile_n']) * 100

merged_df_tcga = pd.merge(tile_counts_df_opx, cancer_tile_counts_df_opx, 
                     left_on='SAMPLE_ID', right_on='PATIENT_ID')
merged_df_tcga['cancer_tile_percentage'] = (merged_df_tcga['cancer_tile_n'] / merged_df_tcga['tile_n']) * 100


##################################################################
#Read pred 
##################################################################
df_nep = pd.read_csv(pred_dir + 'n_token3_EXT_pred_df.csv')

df_opx_tcga = pd.read_csv(pred_dir + 'n_token3_TEST_COMB_pred_df.csv')



