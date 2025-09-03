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
import time
matplotlib.use('Agg')
import pandas as pd
import warnings
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists, set_seed
from Eval import output_pred_perf_with_logit_singletask
from train_utils import FocalLoss,FocalLoss_logitadj
from train_utils import str2bool, random_sample_tiles
from data_loader import get_final_model_data_v2
from data_loader import combine_cohort_data
from train_utils import get_model_ready_data
from ACMIL import ACMIL_GA_singletask, train_one_epoch_singletask,evaluate_singletask, get_slide_feature_singletask
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
import wandb

#source /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/ACMIL-main/acmil/bin/activate
#python3 -u 7_train_ACMIL_mixed_0727_singlemutation.py  --sample_training_n 1000 --out_folder pred_out_081225 --train_flag True  --mutation HR2 --train_cohort union_STNandNSTN_TCGA_NEP

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--s_fold', default=0, type=int, help='select fold')
parser.add_argument('--loss_method', default='', type=str, help='ATTLOSS or ''')
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--learning_method', default='acmil', type=str, help=': e.g., acmil, abmil')
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='AR', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_overlap', default=0, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--train_cohort', default= 'OPX', type=str, help='TCGA or OPX or OPX_TCGA or z_nostnorm_OPX_TCGA or union_STNandNSTN_OPX_TCGA or comb_STNandNSTN_OPX_TCGA')
parser.add_argument('--out_folder', default= 'pred_out_090325', type=str, help='out folder name')

############################################################################################################
#Training Para 
############################################################################################################
parser.add_argument('--train_flag', type=str2bool, default=True, help='train flag')
parser.add_argument('--sample_training_n', default= 0, type=int, help='random sample K tiles')
# parser.add_argument('--f_alpha', default= -1, type=float, help='focal alpha')
# parser.add_argument('--f_gamma', default= 0, type=float, help='focal gamma')
parser.add_argument('--f_alpha', default= -1, type=float, help='focal alpha')
parser.add_argument('--f_gamma', default= 0, type=float, help='focal gamma')

############################################################################################################
#     Model Para
############################################################################################################
#parser.add_argument('--DROPOUT', default=0, type=int, help='drop out rate')
parser.add_argument('--DIM_OUT', default=128, type=int, help='')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--arch', default='ga', type=str, help='e.g., ga_mt, or ga')
parser.add_argument('--train_epoch', default=100, type=int, help='')




            
            
if __name__ == '__main__':
    
    args = parser.parse_args()
    fold_list = [0,1,2,3,4]
    args.train_epoch = 100
    fold_list = [0]
    
    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    
    
    ###################################
    #Load
    ###################################
    start_time = time.time()
    opx_ol100 = torch.load(os.path.join(data_dir, "z_nostnorm_OPX" ,"IMSIZE250_OL100", "feature_uni2", "TFT" + str(args.tumor_frac) ,"z_nostnorm_OPX_data.pth"))
    opx_ol0 = torch.load(os.path.join(data_dir, "z_nostnorm_OPX" ,"IMSIZE250_OL0", "feature_uni2", "TFT" + str(args.tumor_frac) ,"z_nostnorm_OPX_data.pth"))
    elapsed_time = time.time() - start_time
    print(elapsed_time/60)
    

    for f in fold_list:
        
        args.s_fold = 0 
        
        ####################################
        #Load data
        ####################################    
        
        #get train test and valid
        opx_train, opx_train_sp_ids, opx_train_pt_ids = get_model_ready_data(opx_ol100, 'fold'+ str(0), 'TRAIN', selected_label = args.mutation)
        opx_test,  opx_test_sp_ids,  opx_test_pt_ids  = get_model_ready_data(opx_ol0, 'fold'+ str(0), 'TEST', selected_label = args.mutation)
        opx_val,  opx_val_sp_ids,  opx_val_pt_ids  = get_model_ready_data(opx_ol0, 'fold'+ str(0), 'VALID', selected_label = args.mutation)

        has_overlap = any(item in opx_test_sp_ids for item in opx_train_sp_ids)


        train_data = opx_train
        test_data = opx_test
        val_data = opx_val
        
        #Feature and Label N
        N_FEATURE =  train_data[0][0].shape[1]
        N_LABELS  =  train_data[0][1].shape[1]
        

        ####################################               
        #Model Config
        ####################################
        config_dir = "myconf.yml"
        with open(config_dir, "r") as ymlfile:
            c = yaml.load(ymlfile, Loader=yaml.FullLoader)
            conf = Struct(**c)
        conf.train_epoch = args.train_epoch
        conf.D_feat = N_FEATURE
        conf.D_inner = args.DIM_OUT
        conf.n_class = 1
        conf.lr = args.lr
        conf.n_task = N_LABELS
        conf.sample_training_n = args.sample_training_n
        conf.learning_method   = args.learning_method
        conf.arch = args.arch
       
        
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
            

            
        ######################
        #Create output-dir
        ######################
        folder_name1 = args.fe_method + '/TrainOL' + str(args.train_overlap) +  '_TestOL' + str(args.test_overlap) + '_TFT' + str(args.tumor_frac)  + "/" 
        outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                               'trainCohort_' + args.train_cohort + '_Samples' + str(args.sample_training_n),
                               args.learning_method,
                               folder_name1,
                               'FOLD' + str(args.s_fold),
                               args.mutation)
        outdir1 =  outdir0  + "/saved_model/"
        outdir2 =  outdir0  + "/model_para/"
        outdir3 =  outdir0  + "/logs/"
        outdir4 =  outdir0  + "/predictions/"
        outdir5 =  outdir0  + "/perf/"
        outdir6 =  outdir0  + "/trained_features/"
        outdir_list = [outdir0,outdir1,outdir2,outdir3,outdir4,outdir5,outdir6]
        
        for out_path in outdir_list:
            create_dir_if_not_exists(out_path)
    
        ####################################################
        #Select GPU
        ####################################################
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(device)


        ####################################################
        #Training with sampling
        ####################################################
        if conf.sample_training_n > 0:
            #Random Sample 1000 tiles or oriingal N tiles (if total number is < 1000) for training data
            random_sample_tiles(train_data, k = conf.sample_training_n, random_seed = 42)

        
        ####################################################
        #Dataloader for training
        ####################################################
        train_loader = DataLoader(dataset=train_data,batch_size=1, shuffle=False)
        test_loader  = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        val_loader  = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

        ####################################################
        # define network
        ####################################################
        if args.arch == 'ga':
            model = ACMIL_GA_singletask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop)
        model.to(device)
        criterion_da = None 

        ####################################
        #Set out folder
        ####################################
        
        focal_alpha, focal_gamma = args.f_alpha ,args.f_gamma
        outdir11 = outdir1 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
        outdir22 = outdir2 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
        outdir33 = outdir3 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
        outdir44 = outdir4 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
        outdir55 = outdir5 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
        outdir_list = [outdir11,outdir22,outdir33,outdir44,outdir55]
        for out_path in outdir_list:
            create_dir_if_not_exists(out_path)
        
        #criterion = FocalLoss_logitadj(alpha=focal_alpha, gamma=focal_gamma,prior_prob = 0.04,tau = 10, reduction='mean')
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
            best_state = {'epoch':-1, 'val_roc':0, 'test_roc':0}
            train_epoch = conf.train_epoch
            for epoch in range(train_epoch):
                
                train_one_epoch_singletask(model, criterion, train_loader, optimizer0, device, epoch, conf, 
                                           print_every = 100,
                                           loss_method = args.loss_method)
                val_loss, val_roc, val_pr = evaluate_singletask(model, criterion, val_loader, device, conf, 
                                                                         thres = 0.5,
                                                                         cohort_name = "VAL")
                test_loss, test_roc, test_pr = evaluate_singletask(model, criterion, test_loader, device, conf, 
                                                                         thres = 0.5,
                                                                         cohort_name = "TEST_OPX")

                save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,
                    save_path=os.path.join(outdir11 + 'checkpoint_' + 'epoch' + str(epoch) + '.pth'))
                if  val_roc > best_state['val_roc']:
                    best_state['epoch'] = epoch
                    best_state['val_roc'] = val_roc
                    best_state['test_roc'] = test_roc
                    save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,save_path=os.path.join(outdir11, 'checkpoint-best.pth'))
        
            print("Results on best epoch:")
            print(best_state)
            wandb.finish()
            

        
 
    
