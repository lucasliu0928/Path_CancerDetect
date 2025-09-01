#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 17:09:04 2025

@author: jliu6
"""

#NOTE: use python env acmil in ACMIL folder
import sys
import os
import matplotlib
import time
import argparse
matplotlib.use('Agg')
import warnings
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists, set_seed
from data_loader import combine_cohort_data
from data_loader import load_model_ready_data
import torch
warnings.filterwarnings("ignore")

#source /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/ACMIL-main/acmil/bin/activate

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='HR2', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_overlap', default=0, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')


            
            
if __name__ == '__main__':
    
    args = parser.parse_args()

    
    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    id_data_dir = os.path.join(proj_dir, 'intermediate_data', "3B_Train_TEST_IDS")
    
    
    opx_ol100 = load_model_ready_data(data_dir, "OPX", 100, args.fe_method, args.tumor_frac) #overlap 100
    opx_ol100[0]
    
    
    start_time = time.time()
    opx = combine_cohort_data(data_dir, id_data_dir, "OPX" , args.fe_method, args.tumor_frac)
    elapsed_time = time.time() - start_time
    print(elapsed_time/60)
