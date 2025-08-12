#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:45:28 2025

@author: jliu6
"""

#!/usr/bin/env python
# coding: utf-8

#NOTE: use python env acmil in ACMIL folder
import sys
import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import warnings

sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists, set_seed
from train_utils import get_final_model_data
warnings.filterwarnings("ignore")
import argparse

#source /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/ACMIL-main/acmil/bin/activate
#Run: python3 -u 7_train_ACMIL_mixed_0618.py --mutation MT --GRL False --train_cohort OPX_TCGA --train_flag True --batchsize 1 --use_sep_cri False --sample_training_n 1000 --out_folder pred_out_061825_sample1000tiles_trainOPX_TCGA_GRLFALSE --f_alpha 0.2 --f_gamma 6 

#Train n tmux train1
#python3 -u 7_train_ACMIL_mixed_0727_singlemutation.py  --sample_training_n 0 --out_folder pred_out_063025 --f_alpha 0.9 --f_gamma 6 --mutation MT --GRL False --train_cohort union_STNandNSTN_OPX_TCGA --train_flag True --batchsize 1  --batch_train False --use_sep_cri False

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--s_fold', default=0, type=int, help='select fold')
parser.add_argument('--loss_method', default='', type=str, help='ATTLOSS or ''')
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--learning_method', default='acmil', type=str, help=': e.g., acmil, abmil')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='HR2', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_cohort', default= 'union_STNandNSTN_OPX_TCGA', type=str, help='TCGA or OPX or OPX_TCGA or z_nostnorm_OPX_TCGA or union_STNandNSTN_OPX_TCGA or comb_STNandNSTN_OPX_TCGA')



            
            
if __name__ == '__main__':
    
    args = parser.parse_args()
    args.train_flag = True
    fold_list = [0,1,2,3,4]

    
    #fold_list = [0,1,2,3,4]
    for f in fold_list:
        
        args.s_fold = f
        
        ##################
        ###### DIR  ######
        ##################
        proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
        data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
        id_data_dir = os.path.join(proj_dir, 'intermediate_data', "3B_Train_TEST_IDS")
        outdir = os.path.join(proj_dir, 'intermediate_data', "7_model_data", args.train_cohort)
        create_dir_if_not_exists(outdir)

        
        ####################################
        #Load data
        ####################################            
        loaded_data, selected_label = get_final_model_data(data_dir, id_data_dir, args.train_cohort, args.mutation, args.fe_method, args.tumor_frac, args.s_fold)
        train_data, train_ids = loaded_data['train']
        val_data, val_ids = loaded_data['val']
        test_data, test_ids = loaded_data['test']
        test_data1, test_ids1 = loaded_data['test1']
        test_data2, test_ids2 = loaded_data['test2']
        ext_data_st0, nep_ids0  = loaded_data['ext_data_st0']
        ext_data_st1, nep_ids1 = loaded_data['ext_data_st1']
        ext_data_union, nep_ids = loaded_data['ext_data_union']
        
        torch.save(data, os.path.join(outdir, args.cohort_name + '_data.pth'))

        
 