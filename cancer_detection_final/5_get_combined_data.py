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
from misc_utils import set_seed
from data_loader import ModelReadyData_diffdim_V2, get_feature_idexes
from data_loader import combine_features_label_allsamples
from data_loader import write_h5_dataset
warnings.filterwarnings("ignore")



# source ~/.bashrc
# conda activate paimg9
#python3 -u 5_get_combined_data.py --pixel_overlap 100 --cohort_name Neptune --TUMOR_FRAC_THRES 0.0 --fe_method uni1

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Model ready data")
parser.add_argument('--pixel_overlap', default=100, type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--save_image_size', default=250, type=int, help='the size of extracted tiles')
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.0, type=float, help='tile tumor fraction threshold')
parser.add_argument('--cohort_name', default='TCGA_PRAD', type=str, help='data set name: TAN_TMA_Cores or OPX or TCGA_PRAD or Neptune or z_nostnorm_Neptune')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath,virchow2')
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
label_path =    os.path.join(proj_dir,'intermediate_data','3C_labels_train_test', args.cohort_name.replace("z_nostnorm_", ""), "TFT" + str(args.TUMOR_FRAC_THRES))
feature_path = os.path.join(proj_dir,'intermediate_data','4_tile_feature', args.cohort_name, folder_name)
cancer_info_path = os.path.join(proj_dir,'intermediate_data','2_cancer_detection', args.cohort_name, folder_name)

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
#Load labels df
############################################################################################################
all_sp_label_df = pd.read_csv(os.path.join(label_path, "train_test_split.csv"))


if "TCGA_PRAD" in args.cohort_name:
    id_col = "FOLDER_ID"
else:
    id_col = "SAMPLE_ID"
    
selected_ids = list(all_sp_label_df[id_col].unique())    
selected_ids.sort()


############################################################################################################
#Get model ready data
############################################################################################################
start_time = time.time()
file_path = os.path.join(outdir, args.cohort_name + '_data.h5')

if os.path.exists(file_path):
    print(file_path + " already exists.")
else:
    comb_df, comb_df_list = combine_features_label_allsamples(selected_ids, 
                                                       feature_path,
                                                       args.fe_method, 
                                                       args.TUMOR_FRAC_THRES, 
                                                       all_sp_label_df,
                                                       id_col = id_col,
                                                       cancer_info_path = cancer_info_path)
    #Get model ready data
    data = ModelReadyData_diffdim_V2(comb_df_list, SELECTED_FEATURE, SELECTED_LABEL)
    write_h5_dataset(file_path, data)
    #torch.save(data, os.path.join(outdir, args.cohort_name + '_data.pth'))
elapsed_time = (time.time() - start_time)/60
print(elapsed_time, "min")


# #Check h5 and pth matches
# from data_loader import H5Cases
# from data_loader import compare_case
# import h5py
# start_time = time.time()
# ds = H5Cases(os.path.join(outdir, args.cohort_name + '_data.h5'))
# print(len(ds))        # number of cases stored
# elapsed_time = (time.time() - start_time)/60
# print(elapsed_time, "min")


# start_time = time.time()
# ds2 = torch.load(os.path.join(outdir, args.cohort_name + '_data.pth'))
# print(len(ds2))        # number of cases stored
# elapsed_time = (time.time() - start_time)/60
# print(elapsed_time, "min")



# MAX_MISMATCH_TO_PRINT = 20  # stop early if tons of mismatches
# PTH_PATH = os.path.join(outdir, args.cohort_name + '_data.pth')
# H5_PATH = os.path.join(outdir, args.cohort_name + '_data.h5')

# ds_pth = torch.load(PTH_PATH, map_location="cpu")
# f = h5py.File(H5_PATH, "r", swmr=True)
# cases = f["cases"]

# n_pth = len(ds_pth)
# n_h5 = len(cases)
# if n_pth != n_h5:
#     print(f"COUNT MISMATCH: .pth has {n_pth} items, .h5 has {n_h5} groups")
# N = min(n_pth, n_h5)

# total_bad = 0
# for i in range(N):
#     g = cases[str(i)]
#     mm = compare_case(i, ds_pth, g)
#     if mm:
#         total_bad += 1
#         print(f"[i={i}] ❌ {len(mm)} mismatch(es):")
#         for m in mm:
#             print("    -", m)
#         if total_bad >= MAX_MISMATCH_TO_PRINT:
#             print(f"Stopping after {MAX_MISMATCH_TO_PRINT} failing cases…")
#             break
#     else:
#         print(f"[i={i}] ✅ match")

# print(f"\nDone. Compared {N} items. {total_bad} case(s) had mismatches.")
# f.close()
    



