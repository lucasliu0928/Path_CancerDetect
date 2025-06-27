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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists, set_seed
from Eval import boxplot_predprob_by_mutationclass, get_performance, plot_roc_curve
from Eval import bootstrap_ci_from_df, calibrate_probs_isotonic, get_performance_alltask
from Eval import output_pred_perf
from train_utils import FocalLoss, get_feature_idexes, get_selected_labels,has_seven_csv_files, get_train_test_val_data, update_label, load_model_ready_data
from train_utils import str2bool, clean_data, get_train_test_val_data_cohort, random_sample_tiles
from train_utils import get_larger_tumor_fraction_tile, get_matching_tile_index, combine_data_from_stnorm_and_nostnorm
from ACMIL import ACMIL_GA_MultiTask,ACMIL_GA_MultiTask_DA, train_one_epoch_multitask, train_one_epoch_multitask_minibatch, evaluate_multitask, get_emebddings
from ACMIL import train_one_epoch_multitask_minibatch_randomSample,evaluate_multitask_randomSample

warnings.filterwarnings("ignore")


#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
from utils.utils import save_model, Struct
import yaml
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse
from architecture.transformer import ACMIL_GA #ACMIL_GA
from architecture.transformer import ACMIL_MHA
import wandb

#Running in tmux train2
#source /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/ACMIL-main/acmil/bin/activate
#Run: python3 -u 7_train_ACMIL_mixed_0618.py --mutation MT --GRL False --train_cohort OPX_TCGA --train_flag True --batchsize 1 --use_sep_cri False --sample_training_n 1000 --out_folder pred_out_061825_sample1000tiles_trainOPX_TCGA_GRLFALSE --f_alpha 0.2 --f_gamma 6 
############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--learning_method', default='acmil', type=str, help=': e.g., acmil, abmil')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='MT', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--hr_type', default= "HR2", type=str, help='HR version 1 or 2 (2 only include 3 genes)')
parser.add_argument('--GRL', type=str2bool, default=False, help='Enable Gradient Reserse Layer for domain prediciton (yes/no, true/false)')
parser.add_argument('--train_overlap', default=100, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--train_cohort', default= 'union_stnormAndnostnorm_OPX_TCGA', type=str, help='TCGA_PRAD or OPX or z_nostnorm_OPX_TCGA or union_stnormAndnostnorm_OPX_TCGA or comb_stnormAndnostnorm_OPX_TCGA')

parser.add_argument('--proj_dir', default= '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/', type=str, help='model folder name')
parser.add_argument('--data_folder', default= '5_model_ready_data', type=str, help='model folder name')
parser.add_argument('--id_folder', default= '3B_Train_TEST_IDS', type=str, help='ID folder')
parser.add_argument('--model_folder', default= 'pred_out_061825_union_1000tiles_trainOPX_TCGA_GRLFALSE/trainCohort_union_stnormAndnostnorm_OPX_TCGA_GRLFalse/acmil/uni2/TrainOL100_TestOL0_TFT0.9/FOLD0/MTHR_TYPEHR2/', type=str, help='model folder name')
parser.add_argument('--focal_para_folder', default= 'GAMMA_6.0_ALPHA_0.9', type=str, help='focal parameter folder')





############################################################################################################
#     Model Para
############################################################################################################
parser.add_argument('--arch', default='ga_mt', type=str, help='e.g., ga_mt, or ga')
parser.add_argument('--use_sep_cri', type=str2bool, default=False, help='use seperate focal parameters for each mutation')


            
            
if __name__ == '__main__':
    
    args = parser.parse_args()    
    fold_list = [0,1,2,3,4]
    for f in fold_list:
        f = 0
        args.s_fold = f
    
        ####################################
        ######      USERINPUT       ########
        ####################################
        #Feature
        SELECTED_FEATURE = get_feature_idexes(args.fe_method, include_tumor_fraction = False)
        N_FEATURE = len(SELECTED_FEATURE)
        
    
        #Label
        SELECTED_LABEL, selected_label_index = get_selected_labels(args.mutation, args.hr_type, args.train_cohort)
        print(SELECTED_LABEL)
        print(selected_label_index)
        
        
        ##################
        ###### DIR  ######
        ##################
        outdir1 =  os.path.join(args.proj_dir, args.model_folder , "saved_model")
        outdir2 =  os.path.join(args.proj_dir, args.model_folder , "model_para")
        outdir3 =  os.path.join(args.proj_dir, args.model_folder , "logs")
        outdir4 =  os.path.join(args.proj_dir, args.model_folder , "predictions")
        outdir5 =  os.path.join(args.proj_dir, args.model_folder , "perf")
        
        
        ##################                
        #Model Config
        ##################
        with open(outdir2 + '/' + args.focal_para_folder + '/final_config.yml', "r") as ymlfile:
            conf = yaml.unsafe_load(ymlfile)
            #conf = Struct(**c)
        
        print(conf.train_epoch)
        print(conf.D_feat)
        print(conf.D_inner)
        print(conf.n_class)
        print(conf.lr)
        print(conf.n_task)
        print(conf.sample_training_n)
        print(conf.batchsize)
        print(conf.n_token)
        print(conf.mask_drop)
        print(conf.n_masked_patch)

    
        ##################
        #Select GPU
        ##################
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(device)

            
        ################################################
        #     Model ready data 
        ################################################
        data_path = os.path.join(args.proj_dir, args.data_folder)
        
        #OPX data
        data_ol100_opx_stnorm0, _ = clean_data(data_path, 'z_nostnorm_OPX',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol100_opx_stnorm1, _ = clean_data(data_path, 'OPX',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol0_opx_stnorm0, _ = clean_data(data_path, 'z_nostnorm_OPX',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol0_opx_stnorm1, _ = clean_data(data_path, 'OPX',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_opx_stnorm0 = {'OL100': data_ol100_opx_stnorm0, 'OL0': data_ol0_opx_stnorm0}
        data_opx_stnorm1 = {'OL100': data_ol100_opx_stnorm1, 'OL0': data_ol0_opx_stnorm1}
    
        #TCGA data
        data_ol100_tcga_stnorm0, _ = clean_data(data_path, 'z_nostnorm_TCGA_PRAD',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol100_tcga_stnorm1, _ = clean_data(data_path, 'TCGA_PRAD',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol0_tcga_stnorm0, _ = clean_data(data_path, 'z_nostnorm_TCGA_PRAD',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol0_tcga_stnorm1, _ = clean_data(data_path, 'TCGA_PRAD',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_tcga_stnorm0 = {'OL100': data_ol100_tcga_stnorm0, 'OL0': data_ol0_tcga_stnorm0}
        data_tcga_stnorm1 = {'OL100': data_ol100_tcga_stnorm1, 'OL0': data_ol0_tcga_stnorm1}
        
        #Neptune
        data_ol0_nep_stnorm0, nep_ids0   = clean_data(data_path, 'z_nostnorm_Neptune', args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol0_nep_stnorm1, nep_ids1   = clean_data(data_path, 'Neptune', args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
    
        #Combine stnorm and nostnorm
        data_ol100_opx_union = combine_data_from_stnorm_and_nostnorm(data_ol100_opx_stnorm0, data_ol100_opx_stnorm1, method = 'union')
        data_ol100_opx_comb = combine_data_from_stnorm_and_nostnorm(data_ol100_opx_stnorm0, data_ol100_opx_stnorm1, method = 'combine_all')
        
        data_ol0_opx_union = combine_data_from_stnorm_and_nostnorm(data_ol0_opx_stnorm0, data_ol0_opx_stnorm1, method = 'union')
        data_ol0_opx_comb = combine_data_from_stnorm_and_nostnorm(data_ol0_opx_stnorm0, data_ol0_opx_stnorm1, method = 'combine_all')
        
        data_opx_stnorm10_union = {'OL100': data_ol100_opx_union, 'OL0': data_ol0_opx_union}
        data_opx_stnorm10_comb = {'OL100': data_ol100_opx_comb, 'OL0': data_ol0_opx_comb}

        
        data_ol100_tcga_union = combine_data_from_stnorm_and_nostnorm(data_ol100_tcga_stnorm0, data_ol100_tcga_stnorm1, method = 'union')
        data_ol100_tcga_comb = combine_data_from_stnorm_and_nostnorm(data_ol100_tcga_stnorm0, data_ol100_tcga_stnorm1, method = 'combine_all')
        
        data_ol0_tcga_union = combine_data_from_stnorm_and_nostnorm(data_ol0_tcga_stnorm0, data_ol0_tcga_stnorm1, method = 'union')
        data_ol0_tcga_comb = combine_data_from_stnorm_and_nostnorm(data_ol0_tcga_stnorm0, data_ol0_tcga_stnorm1, method = 'combine_all')
        
        data_tcga_stnorm10_union = {'OL100': data_ol100_tcga_union, 'OL0': data_ol0_tcga_union}
        data_tcga_stnorm10_comb = {'OL100': data_ol100_tcga_comb, 'OL0': data_ol0_tcga_comb}
        
        
        data_ol0_nep_union = combine_data_from_stnorm_and_nostnorm(data_ol0_nep_stnorm0, data_ol0_nep_stnorm1, method = 'union')
        nep_id = [entry[-2] for i, entry in enumerate(data_ol0_nep_union)]
        data_ol0_nep_comb = combine_data_from_stnorm_and_nostnorm(data_ol0_nep_stnorm0, data_ol0_nep_stnorm1, method = 'combine_all')
        nep_id = [entry[-2] for i, entry in enumerate(data_ol0_nep_comb)]
    
        ################################################
        #Get Train, test, val data
        ################################################    
        
        
        if args.train_cohort == 'z_nostnorm_OPX_TCGA':
            train_cohort1 = 'z_nostnorm_OPX'
            model_data1 = data_opx_stnorm0
            train_cohort2 = 'z_nostnorm_TCGA_PRAD'
            model_data2 = data_tcga_stnorm0
            
        elif args.train_cohort == 'OPX_TCGA':
            train_cohort1 = 'OPX'
            model_data1 = data_opx_stnorm1
            train_cohort2 = 'TCGA_PRAD'
            model_data2 = data_tcga_stnorm1
        
        elif args.train_cohort == 'union_stnormAndnostnorm_OPX_TCGA':
            train_cohort1 = 'OPX' 
            model_data1 = data_opx_stnorm10_union
            train_cohort2 = 'TCGA_PRAD'
            model_data2 = data_tcga_stnorm10_union
        elif args.train_cohort == 'comb_stnormAndnostnorm_OPX_TCGA':
            train_cohort1 = 'OPX'
            model_data1 = data_opx_stnorm10_comb
            train_cohort2 = 'TCGA_PRAD'
            model_data2 = data_tcga_stnorm10_comb
            
        ################################################################################################
        #For training and test data, take the union of tiles from stained normed and nostained normed tiles
        ################################################################################################
        id_data_dir = os.path.join(args.proj_dir, args.id_folder)
        
        (train_data1, train_ids1), (val_data1, val_ids1), (test_data1, test_ids1) = get_train_test_val_data_cohort(id_data_dir, 
                                                                                                                   train_cohort1 ,
                                                                                                                   model_data = model_data1, 
                                                                                                                   tumor_frac = args.tumor_frac, 
                                                                                                                   s_fold = args.s_fold)
        
        (train_data2, train_ids2), (val_data2, val_ids2), (test_data2, test_ids2) = get_train_test_val_data_cohort(id_data_dir, 
                                                                                                                   train_cohort2 ,
                                                                                                                   model_data = model_data2, 
                                                                                                                   tumor_frac = args.tumor_frac, 
                                                                                                                   s_fold = args.s_fold)
        
        


            
    
        ################################################################################
        #Get Final train and test and val data
        ################################################################################        
        val_data = val_data1 + val_data2
        val_ids = val_ids1 + val_ids2
        
        test_data = test_data1 + test_data2 #put two test together
        test_ids = test_ids1 + test_ids2
        

        if args.train_cohort != 'comb_stnormAndnostnorm_OPX_TCGA':
            #Exclude tile info data, sample ID, patient ID, do not needed it for training
            test_data1 = [item[:-3] for item in test_data1]   
            test_data2 = [item[:-3] for item in test_data2]   
            test_data = [item[:-3] for item in test_data]   
            val_data = [item[:-3] for item in val_data]
        else:
            #Keep feature1, label, tf,1 dlabel, feature2, tf2
            test_data1 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data1]
            test_data2 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data2]
            test_data = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data]
            val_data = [(item[0], item[1], item[2],  item[6], item[8]) for item in val_data]
        
        
        if args.train_cohort == 'union_stnormAndnostnorm_OPX_TCGA':
            #The following two  are the same
            ext_data_nep_st0 = [item[:-3] for item in data_ol0_nep_union]
            ext_data_nep_st1 = [item[:-3] for item in data_ol0_nep_union]
            nep_ids0 = nep_id
            nep_ids1 = nep_id
        elif args.train_cohort == 'comb_stnormAndnostnorm_OPX_TCGA':
            #The following two  are the same
            ext_data_nep_st0 = [(item[0], item[1], item[2], item[6], item[8]) for item in data_ol0_nep_comb]            
            ext_data_nep_st1 = [(item[0], item[1], item[2], item[6], item[8]) for item in data_ol0_nep_comb]
            nep_ids0 = nep_id
            nep_ids1 = nep_id
        else:
            ext_data_nep_st0 = [item[:-3] for item in data_ol0_nep_stnorm0]
            ext_data_nep_st1 = [item[:-3] for item in data_ol0_nep_stnorm1]


    
        
        ####################################################
        #Dataloader for training
        ####################################################
        test_loader1  = DataLoader(dataset=test_data1, batch_size=1, shuffle=False)
        test_loader2  = DataLoader(dataset=test_data2, batch_size=1, shuffle=False)
        test_loader  = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        val_loader   = DataLoader(dataset=val_data,  batch_size=1, shuffle=False)            
        ext_loader_st0   = DataLoader(dataset=ext_data_nep_st0,  batch_size=1, shuffle=False)
        ext_loader_st1   = DataLoader(dataset=ext_data_nep_st1,  batch_size=1, shuffle=False)   
        
        
        ####################################################
        # define network
        ####################################################
        if args.arch == 'ga':
            model = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop)
        elif args.arch == 'ga_mt':
            if args.GRL == False:
                model = ACMIL_GA_MultiTask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
            else:
                model = ACMIL_GA_MultiTask_DA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
        else:
            model = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
        model.to(device)
        
        
        if args.GRL == False:
            criterion_da = None 
        else:
            criterion_da = nn.BCEWithLogitsLoss()
            

        ###################################################
        #  TEST
        ###################################################        
        # Load the checkpoint
        #checkpoint = torch.load(ckpt_dir + 'checkpoint-best.pth')
        mode_idxes = conf.train_epoch-1
        checkpoint = torch.load(os.path.join(outdir1, args.focal_para_folder ,'checkpoint_epoch'+ str(mode_idxes) + '.pth'))
        model.load_state_dict(checkpoint['model'])
        
        
        out_path_pred = os.path.join(outdir4, args.focal_para_folder)
        out_path_pref = os.path.join(outdir5, args.focal_para_folder)
        

        
        # VAL
        output_pred_perf(model, val_loader, val_ids, SELECTED_LABEL, conf, "VAL", out_path_pred, out_path_pref, criterion_da, device)
         
        
        # Test 1
        output_pred_perf(model, test_loader1, test_ids1, SELECTED_LABEL, conf, train_cohort1, out_path_pred, out_path_pref, criterion_da, device)
        
        # Test 2
        output_pred_perf(model, test_loader2, test_ids2, SELECTED_LABEL, conf, train_cohort2 + "2", out_path_pred, out_path_pref, criterion_da, device)
        
        
        #Test Comb
        output_pred_perf(model, test_loader, test_ids, SELECTED_LABEL, conf, "TEST_COMB" + "2", out_path_pred, out_path_pref, criterion_da, device)

        
        #External Validation 1 (z_nostnorm_nep)
        output_pred_perf(model, ext_loader_st0, nep_ids0, SELECTED_LABEL, conf, "z_nostnorm_NEP" + "2", out_path_pred, out_path_pref, criterion_da, device)

        
        #External Validation 2 (normed nep)
        output_pred_perf(model, ext_loader_st1, nep_ids1, SELECTED_LABEL, conf, "NEP" + "2", out_path_pred, out_path_pref, criterion_da, device)


        #TODO
        # #Get embeddings
        # embedding_opx = get_emebddings(model2, test_loader11, device, criterion_da = criterion_da)
        # embedding_tcga = get_emebddings(model2, test_loader22, device, criterion_da = criterion_da)
        # embedding_nep = get_emebddings(model2, ext_loader2, device, criterion_da = criterion_da)

    

            
