#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:45:28 2025

@author: jliu6
#NOTE: use python env acmil in ACMIL folder
#source /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/ACMIL-main/acmil/bin/activate
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')
import warnings
import torch
import argparse
import gc
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists
from train_utils import get_final_split_data
from train_utils import get_cohort_data, combine_data_from_stnorm_and_nostnorm, get_final_model_data
warnings.filterwarnings("ignore")


#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--s_fold', default=0, type=int, help='select fold')
parser.add_argument('--loss_method', default='', type=str, help='ATTLOSS or ''')
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--mutation', default='MT', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_overlap', default=100, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--train_cohort', default= 'union_STNandNSTN_OPX_TCGA', type=str, help='TCGA or OPX or OPX_TCGA or z_nostnorm_OPX_TCGA or union_STNandNSTN_OPX_TCGA or comb_STNandNSTN_OPX_TCGA')
parser.add_argument('--sample_training_n', default= 1000, type=int, help='random sample K tiles')


            
            
if __name__ == '__main__':
    
    args = parser.parse_args()
    
    fold_list = [0,1,2,3,4]
    for f in fold_list:
        
        args.s_fold = f
        
        ##################
        ###### DIR  ######
        ##################
        proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred'    
        data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
        id_data_dir =os.path.join(proj_dir, 'intermediate_data', "3B_Train_TEST_IDS")
        outdir = os.path.join(proj_dir,
                               "intermediate_data", 
                               "7_model_ready_data",
                               'trainCohort_' + args.train_cohort,
                               'TrainOL' + str(args.train_overlap) +  '_TestOL' + str(args.test_overlap) + '_' + args.fe_method + '_TFT' + str(args.tumor_frac), 
                               'FOLD' + str(args.s_fold))

        create_dir_if_not_exists(outdir)
    
            
        ################################################
        #     Get combined feature data
        ################################################
        loaded_data = get_final_model_data(data_dir, id_data_dir, args.train_cohort, args.fe_method, args.tumor_frac, args.s_fold)
        train_data, train_ids = loaded_data['train']
        val_data = loaded_data['val']
        test_data = loaded_data['test']
        test_data1 = loaded_data['test1']
        test_data2 = loaded_data['test2']
        ext_data_st0 = loaded_data['ext_data_st0']
        ext_data_st1 = loaded_data['ext_data_st1']
        ext_data_union = loaded_data['ext_data_union']



        


        

        
        

        




        
        
        

            
    
