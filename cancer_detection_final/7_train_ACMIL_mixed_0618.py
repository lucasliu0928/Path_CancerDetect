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
from Eval import predict_v2, predict_v2_sp_nost_andst, output_pred_perf
from train_utils import FocalLoss, get_feature_idexes, get_selected_labels,has_seven_csv_files, get_train_test_val_data, update_label, load_model_ready_data
from train_utils import str2bool, clean_data, get_train_test_val_data_cohort, random_sample_tiles
from train_utils import get_larger_tumor_fraction_tile, get_matching_tile_index, combine_data_from_stnorm_and_nostnorm
from train_utils import load_data
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


#Train with sp from st and nost
#python3 -u 7_train_ACMIL_mixed_0618.py  --sample_training_n 0 --out_folder pred_out_061825_comb_alltiles_trainOPX_TCGA_GRLFALSE --f_alpha 0.9 --f_gamma 6 --mutation MT --GRL False --train_cohort comb_stnormAndnostnorm_OPX_TCGA --train_flag True --batchsize 1  --batch_train False --use_sep_cri False

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
parser.add_argument('--mutation', default='MT', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_overlap', default=100, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--train_cohort', default= 'union_stnormAndnostnorm_OPX_TCGA', type=str, help='TCGA_PRAD or OPX or z_nostnorm_OPX_TCGA or union_stnormAndnostnorm_OPX_TCGA or comb_stnormAndnostnorm_OPX_TCGA')
parser.add_argument('--f_alpha', default= 0.9, type=float, help='focal alpha')
parser.add_argument('--f_gamma', default= 6, type=float, help='focal gamma')
parser.add_argument('--GRL', type=str2bool, default=False, help='Enable Gradient Reserse Layer for domain prediciton (yes/no, true/false)')
parser.add_argument('--train_flag', type=str2bool, default=False, help='train flag')
parser.add_argument('--sample_training_n', default= 1000, type=int, help='random sample K tiles')
parser.add_argument('--train_with_samplingSTandNOST', type=str2bool, default=False, help='train flag')
parser.add_argument('--out_folder', default= 'pred_out_061825_new2', type=str, help='out folder name')
#parser.add_argument('--hr_type', default= "HR2", type=str, help='HR version 1 or 2 (2 only include 3 genes)')


############################################################################################################
#     Model Para
############################################################################################################
parser.add_argument('--batch_train', type=str2bool, default=False,  help='if use batch train')
parser.add_argument('--batchsize', default=32, type=int, help='training batch size')
#parser.add_argument('--DROPOUT', default=0, type=int, help='drop out rate')
parser.add_argument('--DIM_OUT', default=128, type=int, help='')
parser.add_argument('--train_epoch', default=50, type=int, help='')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--arch', default='ga_mt', type=str, help='e.g., ga_mt, or ga')
parser.add_argument('--use_sep_cri', type=str2bool, default=False, help='use seperate focal parameters for each mutation')




            
            
if __name__ == '__main__':
    
    args = parser.parse_args()
    #args.GRL = False
    #args.s_fold  = 0
    args.train_flag = True
    #args.mutation = 'MT'
    #args.train_epoch = 10
    
    fold_list = [0,1,2,3,4]
    for f in fold_list:
        
        args.s_fold = f
    
    
        ####################################
        ######      USERINPUT       ########
        ####################################
        #Feature
        SELECTED_FEATURE = get_feature_idexes(args.fe_method, include_tumor_fraction = False)
        N_FEATURE = len(SELECTED_FEATURE)
        
    
        #Label
        SELECTED_LABEL, selected_label_index = get_selected_labels(args.mutation, args.train_cohort)
        print(SELECTED_LABEL)
        print(selected_label_index)

                        
        #Model Config
        config_dir = "myconf.yml"
        with open(config_dir, "r") as ymlfile:
            c = yaml.load(ymlfile, Loader=yaml.FullLoader)
            conf = Struct(**c)
        conf.train_epoch = args.train_epoch
        conf.D_feat = N_FEATURE
        conf.D_inner = args.DIM_OUT
        conf.n_class = 1
        conf.wandb_mode = 'disabled'
        conf.lr = args.lr
        conf.n_task = len(SELECTED_LABEL)
        conf.sample_training_n = args.sample_training_n
        conf.batchsize = args.batchsize
        conf.learning_method = args.learning_method
        conf.arch = args.arch
        conf.GRL = args.GRL
       
        
        if args.learning_method == 'abmil':
            conf.n_token = 1
            conf.mask_drop = 0
            conf.n_masked_patch = 0
        elif args.learning_method == 'acmil':
            conf.n_token = 3
            conf.mask_drop = 0.6
            conf.n_masked_patch = 0
            
        # Print all key-value pairs in the conf object
        for key, value in conf.__dict__.items():
            print(f"{key}: {value}")
            

            
        ##################
        ###### DIR  ######
        ##################
        proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'        
    
        ######################
        #Create output-dir
        ######################
        folder_name1 = args.fe_method + '/TrainOL' + str(args.train_overlap) +  '_TestOL' + str(args.test_overlap) + '_TFT' + str(args.tumor_frac)  + "/" 
        outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                               'trainCohort_' + args.train_cohort + '_GRL' + str(args.GRL),
                               args.learning_method,
                               folder_name1,
                               'FOLD' + str(args.s_fold),
                               args.mutation)
        outdir1 =  outdir0  + "/saved_model/"
        outdir2 =  outdir0  + "/model_para/"
        outdir3 =  outdir0  + "/logs/"
        outdir4 =  outdir0  + "/predictions/"
        outdir5 =  outdir0  + "/perf/"
        outdir_list = [outdir0,outdir1,outdir2,outdir3,outdir4,outdir5]
        
        for out_path in outdir_list:
            create_dir_if_not_exists(out_path)
    
        ##################
        #Select GPU
        ##################
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(device)

            
        # ################################################
        # #     Model ready data 
        # ################################################
        # data_path = proj_dir + 'intermediate_data/5_combined_data'
        
        # #OPX data
        # data_ol100_opx_stnorm0, _ = clean_data(data_path, 'z_nostnorm_OPX',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        # data_ol100_opx_stnorm1, _ = clean_data(data_path, 'OPX',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        # data_ol0_opx_stnorm0, _ = clean_data(data_path, 'z_nostnorm_OPX',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        # data_ol0_opx_stnorm1, _ = clean_data(data_path, 'OPX',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        # data_opx_stnorm0 = {'OL100': data_ol100_opx_stnorm0, 'OL0': data_ol0_opx_stnorm0}
        # data_opx_stnorm1 = {'OL100': data_ol100_opx_stnorm1, 'OL0': data_ol0_opx_stnorm1}
    
        # #TCGA data
        # data_ol100_tcga_stnorm0, _ = clean_data(data_path, 'z_nostnorm_TCGA_PRAD',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        # data_ol100_tcga_stnorm1, _ = clean_data(data_path, 'TCGA_PRAD',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        # data_ol0_tcga_stnorm0, _ = clean_data(data_path, 'z_nostnorm_TCGA_PRAD',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        # data_ol0_tcga_stnorm1, _ = clean_data(data_path, 'TCGA_PRAD',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        # data_tcga_stnorm0 = {'OL100': data_ol100_tcga_stnorm0, 'OL0': data_ol0_tcga_stnorm0}
        # data_tcga_stnorm1 = {'OL100': data_ol100_tcga_stnorm1, 'OL0': data_ol0_tcga_stnorm1}
        
        # #Neptune
        # data_ol0_nep_stnorm0, nep_ids0   = clean_data(data_path, 'z_nostnorm_Neptune', args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        # data_ol0_nep_stnorm1, nep_ids1   = clean_data(data_path, 'Neptune', args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
    
        # #Combine stnorm and nostnorm
        # data_ol100_opx_union = combine_data_from_stnorm_and_nostnorm(data_ol100_opx_stnorm0, data_ol100_opx_stnorm1, method = 'union')
        # data_ol100_opx_comb = combine_data_from_stnorm_and_nostnorm(data_ol100_opx_stnorm0, data_ol100_opx_stnorm1, method = 'combine_all')
        
        # data_ol0_opx_union = combine_data_from_stnorm_and_nostnorm(data_ol0_opx_stnorm0, data_ol0_opx_stnorm1, method = 'union')
        # data_ol0_opx_comb = combine_data_from_stnorm_and_nostnorm(data_ol0_opx_stnorm0, data_ol0_opx_stnorm1, method = 'combine_all')
        
        # data_opx_stnorm10_union = {'OL100': data_ol100_opx_union, 'OL0': data_ol0_opx_union}
        # data_opx_stnorm10_comb = {'OL100': data_ol100_opx_comb, 'OL0': data_ol0_opx_comb}

        
        # data_ol100_tcga_union = combine_data_from_stnorm_and_nostnorm(data_ol100_tcga_stnorm0, data_ol100_tcga_stnorm1, method = 'union')
        # data_ol100_tcga_comb = combine_data_from_stnorm_and_nostnorm(data_ol100_tcga_stnorm0, data_ol100_tcga_stnorm1, method = 'combine_all')
        
        # data_ol0_tcga_union = combine_data_from_stnorm_and_nostnorm(data_ol0_tcga_stnorm0, data_ol0_tcga_stnorm1, method = 'union')
        # data_ol0_tcga_comb = combine_data_from_stnorm_and_nostnorm(data_ol0_tcga_stnorm0, data_ol0_tcga_stnorm1, method = 'combine_all')
        
        # data_tcga_stnorm10_union = {'OL100': data_ol100_tcga_union, 'OL0': data_ol0_tcga_union}
        # data_tcga_stnorm10_comb = {'OL100': data_ol100_tcga_comb, 'OL0': data_ol0_tcga_comb}
        
        
        # data_ol0_nep_union = combine_data_from_stnorm_and_nostnorm(data_ol0_nep_stnorm0, data_ol0_nep_stnorm1, method = 'union')
        # nep_id = [entry[-2] for i, entry in enumerate(data_ol0_nep_union)]
        # data_ol0_nep_comb = combine_data_from_stnorm_and_nostnorm(data_ol0_nep_stnorm0, data_ol0_nep_stnorm1, method = 'combine_all')
        # nep_id = [entry[-2] for i, entry in enumerate(data_ol0_nep_comb)]
    
        # #TODO Actual: Check OPX_001 was removed beased no cancer detected in stnormed
        # ################################################
        # #Get Train, test, val data
        # ################################################    
        # id_data_dir = proj_dir + 'intermediate_data/3B_Train_TEST_IDS'
        
        # if args.train_cohort == 'z_nostnorm_OPX_TCGA':
        #     train_cohort1 = 'z_nostnorm_OPX'
        #     model_data1 = data_opx_stnorm0
        #     train_cohort2 = 'z_nostnorm_TCGA_PRAD'
        #     model_data2 = data_tcga_stnorm0
            
        # elif args.train_cohort == 'OPX_TCGA':
        #     train_cohort1 = 'OPX'
        #     model_data1 = data_opx_stnorm1
        #     train_cohort2 = 'TCGA_PRAD'
        #     model_data2 = data_tcga_stnorm1
            
        # elif args.train_cohort == 'union_stnormAndnostnorm_OPX_TCGA':
        #     train_cohort1 = 'OPX'
        #     model_data1 = data_opx_stnorm10_union
        #     train_cohort2 = 'TCGA_PRAD'
        #     model_data2 = data_tcga_stnorm10_union
        # elif args.train_cohort == 'comb_stnormAndnostnorm_OPX_TCGA':
        #     train_cohort1 = 'OPX'
        #     model_data1 = data_opx_stnorm10_comb
        #     train_cohort2 = 'TCGA_PRAD'
        #     model_data2 = data_tcga_stnorm10_comb
            

            
        # ################################################################################################
        # #For training and test data, take the union of tiles from stained normed and nostained normed tiles
        # ################################################################################################
        
        # (train_data1, train_ids1), (val_data1, val_ids1), (test_data1, test_ids1) = get_train_test_val_data_cohort(id_data_dir, 
        #                                                                                                            train_cohort1 ,
        #                                                                                                            model_data = model_data1, 
        #                                                                                                            tumor_frac = args.tumor_frac, 
        #                                                                                                            s_fold = args.s_fold)
        
        # (train_data2, train_ids2), (val_data2, val_ids2), (test_data2, test_ids2) = get_train_test_val_data_cohort(id_data_dir, 
        #                                                                                                            train_cohort2 ,
        #                                                                                                            model_data = model_data2, 
        #                                                                                                            tumor_frac = args.tumor_frac, 
        #                                                                                                            s_fold = args.s_fold)
        
        


            
    
        # ################################################################################
        # #Get Final train and test and val data
        # ################################################################################
        # train_data = train_data1 + train_data2
        # train_ids = train_ids1 + train_ids2
        
        # val_data = val_data1 + val_data2
        # val_ids = val_ids1 + val_ids2
        
        # test_data = test_data1 + test_data2 #put two test together
        # test_ids = test_ids1 + test_ids2
        
        # if args.train_cohort != 'comb_stnormAndnostnorm_OPX_TCGA':
        #     if conf.sample_training_n > 0:
        #         #Random Sample 1000 tiles or oriingal N tiles (if total number is < 1000) for training data
        #         random_sample_tiles(train_data, k = conf.sample_training_n, random_seed = 42)

        # if args.train_cohort == 'comb_stnormAndnostnorm_OPX_TCGA': 
        #     #Keep feature1, label, tf,1 dlabel, feature2, tf2
        #     train_data = [(item[0], item[1], item[2], item[3], item[7], item[9]) for item in train_data]
        #     test_data1 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data1]
        #     test_data2 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data2]
        #     test_data = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data]
        #     val_data = [(item[0], item[1], item[2],  item[6], item[8]) for item in val_data]
        # else:
        #     #Exclude tile info data, sample ID, patient ID, do not needed it for training
        #     train_data = [item[:-3] for item in train_data]
        #     test_data1 = [item[:-3] for item in test_data1]   
        #     test_data2 = [item[:-3] for item in test_data2]   
        #     test_data = [item[:-3] for item in test_data]   
        #     val_data = [item[:-3] for item in val_data]
        

        
        # ext_data_nep_st0 = [item[:-3] for item in data_ol0_nep_stnorm0] #no st norm
        # ext_data_nep_st1 = [item[:-3] for item in data_ol0_nep_stnorm1] #st normed
        # ext_data_nep_union = [item[:-3] for item in data_ol0_nep_union]
        
        
        # #write to pt
        # torch.save({"data": train_data, "id": train_ids}, 'train_data.pt')
        # torch.save({"data": test_data1, "id": test_ids1}, 'test_data1.pt')
        # torch.save({"data": test_data2, "id": test_ids2}, 'test_data2.pt')
        # torch.save({"data": test_data, "id": test_ids}, 'test_data.pt')
        # torch.save({"data": val_data, "id": val_ids}, 'val_data.pt')
        # torch.save({"data": ext_data_nep_st0, "id": nep_ids0}, 'ext_data_nep_st0.pt')
        # torch.save({"data": ext_data_nep_st1, "id": nep_ids1}, 'ext_data_nep_st1.pt')
        # torch.save({"data": ext_data_nep_union, "id": nep_id}, 'ext_data_nep_union.pt')
        
        
        
        #Load data
        if args.train_cohort == 'z_nostnorm_OPX_TCGA':
            train_cohort1 = 'z_nostnorm_OPX'
            train_cohort2 = 'z_nostnorm_TCGA_PRAD'
            
        else:
            train_cohort1 = 'OPX'
            train_cohort2 = 'TCGA_PRAD'

            
        train_data, train_ids = load_data("train_data.pt")
        test_data1, test_ids1 = load_data("test_data1.pt")
        test_data2, test_ids2 = load_data("test_data2.pt")
        test_data, test_ids = load_data("test_data.pt")
        val_data, val_ids = load_data("val_data.pt")
        ext_data_nep_st0, nep_ids0 = load_data("ext_data_nep_st0.pt")
        ext_data_nep_st1, nep_ids1 = load_data("ext_data_nep_st1.pt")
        ext_data_nep_union, nep_id = load_data("ext_data_nep_union.pt")



        
        
        

            
    
        
        ####################################################
        #Dataloader for training
        ####################################################
        train_loader = DataLoader(dataset=train_data,batch_size=1, shuffle=False)
        test_loader1  = DataLoader(dataset=test_data1, batch_size=1, shuffle=False)
        test_loader2  = DataLoader(dataset=test_data2, batch_size=1, shuffle=False)
        test_loader  = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        val_loader   = DataLoader(dataset=val_data,  batch_size=1, shuffle=False)            
        ext_loader_st0   = DataLoader(dataset=ext_data_nep_st0,  batch_size=1, shuffle=False)
        ext_loader_st1   = DataLoader(dataset=ext_data_nep_st1,  batch_size=1, shuffle=False) 
        ext_loader_union   = DataLoader(dataset=ext_data_nep_union,  batch_size=1, shuffle=False)
        

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
            
    
        # Define different values for alpha and gamma
        if args.use_sep_cri:
            alpha_values = [0.9, 0.9, 0.8, 0.9, 0.7, 0.9, 0.9]  # Example alpha values
            gamma_values = [2,    2,    2,   2,  2,   2,  2]   # Example gamma values
            
            focal_gamma = 'use_sep_cri'
            focal_alpha = 'use_sep_cri'
            outdir11 = outdir1 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir22 = outdir2 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir33 = outdir3 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir44 = outdir4 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir55 = outdir5 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir_list = [outdir11,outdir22,outdir33,outdir44,outdir55]
            for out_path in outdir_list:
                create_dir_if_not_exists(out_path)
                
            # Create a list of FocalLoss criteria with different alpha and gamma
            criterion = [FocalLoss(alpha=a, gamma=g, reduction='mean') for a, g in zip(alpha_values, gamma_values)]
                
            
            ####################################
            #Ouput hyper para
            ####################################
            conf.focal_alpha = alpha_values
            conf.focal_gamma = gamma_values
            with open(outdir22 + 'final_config.yml', 'w') as file:
                yaml.dump(conf, file, sort_keys=False)
                    
            if args.train_flag == True:
                ####################################################
                #            Train 
                ####################################################
                set_seed(0)
                # define optimizer, lr not important at this point
                optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=conf.lr, weight_decay=conf.wd)
                best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
                train_epoch = conf.train_epoch
                for epoch in range(train_epoch):
                    
                    if args.train_with_samplingSTandNOST:
                        train_one_epoch_multitask_minibatch_randomSample(model, criterion, train_loader, optimizer0, device, epoch, conf, 
                                                            batch_train = args.batch_train, 
                                                            accum_steps = args.batchsize, 
                                                            print_every = 100,
                                                            loss_method = args.loss_method, 
                                                            use_sep_criterion = args.use_sep_cri, 
                                                            criterion_da = criterion_da)
                        val_auc, val_acc, val_f1, val_loss = evaluate_multitask_randomSample(model, criterion, val_loader, device, conf, 'Val', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                        test_auc1, test_acc1, test_f11, test_loss1 = evaluate_multitask_randomSample(model, criterion, test_loader1, device, conf, 'Test', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)

                    else:
                        #train_one_epoch_multitask(model, criterion, train_loader, optimizer0, device, epoch, conf, args.loss_method, use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                        train_one_epoch_multitask_minibatch(model, criterion, train_loader, optimizer0, device, epoch, conf, 
                                                            batch_train = args.batch_train, 
                                                            accum_steps = args.batchsize, 
                                                            print_every = 100,
                                                            loss_method = args.loss_method, 
                                                            use_sep_criterion = args.use_sep_cri, 
                                                            criterion_da = criterion_da)
                        val_auc, val_acc, val_f1, val_loss = evaluate_multitask(model, criterion, val_loader, device, conf, 'Val', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                        test_auc1, test_acc1, test_f11, test_loss1 = evaluate_multitask(model, criterion, test_loader1, device, conf, 'Test', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                    #test_auc2, test_acc2, test_f12, test_loss2 = evaluate_multitask(model, criterion, test_loader2, device, conf, 'TCGA', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                    
                    save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,
                        save_path=os.path.join(outdir11 + 'checkpoint_' + 'epoch' + str(epoch) + '.pth'))
                    if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
                        best_state['epoch'] = epoch
                        best_state['val_auc'] = val_auc
                        best_state['val_acc'] = val_acc
                        best_state['val_f1'] = val_f1
                        best_state['test_auc'] = test_auc1
                        best_state['test_acc'] = test_acc1
                        best_state['test_f1'] = test_f11
                        save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,save_path=os.path.join(outdir11, 'checkpoint-best.pth'))
            
                print("Results on best epoch:")
                print(best_state)
                wandb.finish()
            
        else:
            # focal_alpha = args.f_alpha
            # focal_gamma = args.f_gamma
            
            #a_g_repairs = [(0.7,0),(0.7,7),(0.2,6), (0.8,9)]
            #a_g_repairs = [(0.2,6), (0.3,6) , (0.4,6), (0.7,0),(0.7,7), (0.8,9)]
            #a_g_repairs = [(0.9,6),(0.4,6), (1,0)]
    
            #for focal_alpha, focal_gamma in a_g_repairs:
        
            focal_alpha, focal_gamma = args.f_alpha ,args.f_gamma
            outdir11 = outdir1 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir22 = outdir2 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir33 = outdir3 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir44 = outdir4 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir55 = outdir5 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir_list = [outdir11,outdir22,outdir33,outdir44,outdir55]
            for out_path in outdir_list:
                create_dir_if_not_exists(out_path)
            
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
            
            ####################################
            #Ouput hyper para
            ####################################
            conf.focal_alpha = focal_alpha
            conf.focal_gamma = focal_gamma
            with open(outdir22 + 'final_config.yml', 'w') as file:
                yaml.dump(conf, file, sort_keys=False)
            
            if args.train_flag == True:
                ####################################################
                #            Train 
                ####################################################
                set_seed(0)
                # define optimizer, lr not important at this point
                optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=conf.lr, weight_decay=conf.wd)
                best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
                train_epoch = conf.train_epoch
                for epoch in range(train_epoch):
                    if args.train_with_samplingSTandNOST:
                        train_one_epoch_multitask_minibatch_randomSample(model, criterion, train_loader, optimizer0, device, epoch, conf, 
                                                            batch_train = args.batch_train, 
                                                            accum_steps = args.batchsize, 
                                                            print_every = 100,
                                                            loss_method = args.loss_method, 
                                                            use_sep_criterion = args.use_sep_cri, 
                                                            criterion_da = criterion_da)
                        val_auc, val_acc, val_f1, val_loss = evaluate_multitask_randomSample(model, criterion, val_loader, device, conf, 'Val', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                        test_auc1, test_acc1, test_f11, test_loss1 = evaluate_multitask_randomSample(model, criterion, test_loader1, device, conf, 'Test', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)

                    else:
                        #train_one_epoch_multitask(model, criterion, train_loader, optimizer0, device, epoch, conf, args.loss_method, use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                        train_one_epoch_multitask_minibatch(model, criterion, train_loader, optimizer0, device, epoch, conf, 
                                                            batch_train = args.batch_train, 
                                                            accum_steps = args.batchsize, 
                                                            print_every = 100,
                                                            loss_method = args.loss_method, 
                                                            use_sep_criterion = args.use_sep_cri, 
                                                            criterion_da = criterion_da)
                        val_auc, val_acc, val_f1, val_loss = evaluate_multitask(model, criterion, val_loader, device, conf, 'Val', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                        test_auc1, test_acc1, test_f11, test_loss1 = evaluate_multitask(model, criterion, test_loader1, device, conf, 'Test', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                    #test_auc2, test_acc2, test_f12, test_loss2 = evaluate_multitask(model, criterion, test_loader2, device, conf, 'TCGA', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                    
                    save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,
                        save_path=os.path.join(outdir11 + 'checkpoint_' + 'epoch' + str(epoch) + '.pth'))
                    if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
                        best_state['epoch'] = epoch
                        best_state['val_auc'] = val_auc
                        best_state['val_acc'] = val_acc
                        best_state['val_f1'] = val_f1
                        best_state['test_auc'] = test_auc1
                        best_state['test_acc'] = test_acc1
                        best_state['test_f1'] = test_f11
                        save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,save_path=os.path.join(outdir11, 'checkpoint-best.pth'))
            
                print("Results on best epoch:")
                print(best_state)
                wandb.finish()
                
    
            
            ###################################################
            #  TEST
            ###################################################
            #Load model
            if args.arch == 'ga':
                model2 = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop)
            elif args.arch == 'ga_mt':
                if args.GRL == False:
                    model2 = ACMIL_GA_MultiTask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
                else:
                    model2 = ACMIL_GA_MultiTask_DA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
            else:
                model2 = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
            model2.to(device)
            
            
            ###################################################
            #  TEST
            ###################################################        
            # Load the checkpoint
            #checkpoint = torch.load(ckpt_dir + 'checkpoint-best.pth')
            mode_idxes = conf.train_epoch-1
            checkpoint = torch.load(os.path.join(outdir11 ,'checkpoint_epoch'+ str(mode_idxes) + '.pth'))
            model2.load_state_dict(checkpoint['model'])
            
            
            out_path_pred = os.path.join(outdir44)
            out_path_pref = os.path.join(outdir55)
            
    
            
            # VAL
            output_pred_perf(model2, val_loader, val_ids, SELECTED_LABEL, conf, "VAL", out_path_pred, out_path_pref, criterion_da, device)
             
            
            # Test 1
            output_pred_perf(model2, test_loader1, test_ids1, SELECTED_LABEL, conf, train_cohort1, out_path_pred, out_path_pref, criterion_da, device)
            
            # Test 2
            output_pred_perf(model2, test_loader2, test_ids2, SELECTED_LABEL, conf, train_cohort2 , out_path_pred, out_path_pref, criterion_da, device)
            
            
            #Test Comb
            output_pred_perf(model2, test_loader, test_ids, SELECTED_LABEL, conf, "TEST_COMB", out_path_pred, out_path_pref, criterion_da, device)
    
            
            #External Validation 1 (z_nostnorm_nep)
            output_pred_perf(model2, ext_loader_st0, nep_ids0, SELECTED_LABEL, conf, "z_nostnorm_NEP", out_path_pred, out_path_pref, criterion_da, device)
    
            
            #External Validation 2 (normed nep)
            output_pred_perf(model2, ext_loader_st1, nep_ids1, SELECTED_LABEL, conf, "NEP", out_path_pred, out_path_pref, criterion_da, device)
            
            
            #External Validation 3 (union)
            output_pred_perf(model2, ext_loader_union, nep_id, SELECTED_LABEL, conf, "NEP_union" , out_path_pred, out_path_pref, criterion_da, device)

