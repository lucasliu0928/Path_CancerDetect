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
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists, set_seed
from Eval import output_pred_perf_with_logit_singletask
from train_utils import FocalLoss,FocalLoss_logitadj
from train_utils import str2bool, random_sample_tiles
from train_utils import get_final_model_data_v2
from train_utils import combine_cohort_data
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
parser.add_argument('--mutation', default='HR1', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_overlap', default=100, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--train_cohort', default= 'union_STNandNSTN_OPX_TCGA', type=str, help='TCGA or OPX or OPX_TCGA or z_nostnorm_OPX_TCGA or union_STNandNSTN_OPX_TCGA or comb_STNandNSTN_OPX_TCGA')
parser.add_argument('--out_folder', default= 'pred_out_082225', type=str, help='out folder name')

############################################################################################################
#Training Para 
############################################################################################################
parser.add_argument('--train_flag', type=str2bool, default=True, help='train flag')
parser.add_argument('--sample_training_n', default= 1000, type=int, help='random sample K tiles')
# parser.add_argument('--f_alpha', default= -1, type=float, help='focal alpha')
# parser.add_argument('--f_gamma', default= 0, type=float, help='focal gamma')
parser.add_argument('--f_alpha', default= 0.2, type=float, help='focal alpha')
parser.add_argument('--f_gamma', default= 6, type=float, help='focal gamma')

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
    #args.train_flag = True
    #args.out_folder = 'pred_out_081225'
    fold_list = [0,1,2,3,4]
    #args.train_epoch = 1
    
    #fold_list = [0,1,2,3,4]
    
    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    id_data_dir = os.path.join(proj_dir, 'intermediate_data', "3B_Train_TEST_IDS")
    data_out_dir = os.path.join(proj_dir, "intermediate_data", "5B_modelready_data")
    
    
    ####################################
    #Get model ready data cohort
    ####################################
    # start_time = time.time()
    # opx = combine_cohort_data(data_dir, id_data_dir, "OPX" , args.fe_method, args.tumor_frac)
    # torch.save(opx, os.path.join(data_out_dir,"opx.pth")) #['stnorm0_OL100', 'stnorm0_OL0', 'stnorm1_OL100', 'stnorm1_OL0', 'Union_OL100', 'Union_OL0']
    # elapsed_time = time.time() - start_time
    # print(elapsed_time/60)
    
    # start_time = time.time()
    # tcga = combine_cohort_data(data_dir, id_data_dir, "TCGA_PRAD" , args.fe_method, args.tumor_frac)
    # torch.save(tcga, os.path.join(data_out_dir,"TCGA_PRAD.pth")) #['stnorm0_OL100', 'stnorm0_OL0', 'stnorm1_OL100', 'stnorm1_OL0', 'Union_OL100', 'Union_OL0']
    # elapsed_time = time.time() - start_time
    # print(elapsed_time/60)
    
    
    # start_time = time.time()
    # nep = combine_cohort_data(data_dir, id_data_dir, "Neptune" , args.fe_method, args.tumor_frac)
    # torch.save(nep, os.path.join(data_out_dir,"Neptune.pth")) #['stnorm0_OL100', 'stnorm0_OL0', 'stnorm1_OL100', 'stnorm1_OL0', 'Union_OL100', 'Union_OL0']
    # elapsed_time = time.time() - start_time
    # print(elapsed_time/60)
    
    #Load data
    start_time = time.time()
    opx = torch.load(os.path.join(data_out_dir,"opx.pth")) #['stnorm0_OL100', 'stnorm0_OL0', 'stnorm1_OL100', 'stnorm1_OL0', 'Union_OL100', 'Union_OL0']
    tcga = torch.load(os.path.join(data_out_dir,"TCGA_PRAD.pth")) #['stnorm0_OL100', 'stnorm0_OL0', 'stnorm1_OL100', 'stnorm1_OL0', 'Union_OL100', 'Union_OL0']
    nep = torch.load(os.path.join(data_out_dir,"Neptune.pth")) #['stnorm0_OL100', 'stnorm0_OL0', 'stnorm1_OL100', 'stnorm1_OL0', 'Union_OL100', 'Union_OL0']
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    
    for f in fold_list:
        
        args.s_fold = f

        ####################################
        #Load data
        ####################################            
        loaded_data, selected_label = get_final_model_data_v2(opx,tcga, nep, id_data_dir, args.train_cohort, args.mutation, args.fe_method, args.tumor_frac, args.s_fold)
        train_data, train_ids, train_name = loaded_data['train']
        val_data, val_ids, val_name = loaded_data['val']
        test_data, test_ids, test_name = loaded_data['test']
        test_data1, test_ids1, test_name1 = loaded_data['test1']
        test_data2, test_ids2, test_name2 = loaded_data['test2']
        test_data3, test_ids3, test_name3 = loaded_data['test3']

        ext_data_st0, ext_ids0, ext_name1 = loaded_data['ext_data_st0']
        ext_data_st1, ext_ids1, ext_name2 = loaded_data['ext_data_st1']
        ext_data_union, ext_ids, ext_name = loaded_data['ext_data_union']

        
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
        conf.learning_method = args.learning_method
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
    
        ##################
        #Select GPU
        ##################
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
        test_loader1  = DataLoader(dataset=test_data1, batch_size=1, shuffle=False)
        test_loader2  = DataLoader(dataset=test_data2, batch_size=1, shuffle=False)
        test_loader3  = DataLoader(dataset=test_data3, batch_size=1, shuffle=False)
        test_loader  = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        val_loader   = DataLoader(dataset=val_data,  batch_size=1, shuffle=False)            
        ext_loader_st0   = DataLoader(dataset=ext_data_st0,  batch_size=1, shuffle=False)
        ext_loader_st1   = DataLoader(dataset=ext_data_st1,  batch_size=1, shuffle=False) 
        ext_loader_union   = DataLoader(dataset=ext_data_union,  batch_size=1, shuffle=False)
        

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
                test_loss, test_roc, test_pr = evaluate_singletask(model, criterion, test_loader1, device, conf, 
                                                                         thres = 0.5,
                                                                         cohort_name = "TEST_OPX")
                test_loss2, test_roc2, test_pr2 = evaluate_singletask(model, criterion, test_loader2, device, conf, 
                                                                         thres = 0.5,
                                                                         cohort_name = "TEST_TCGA")

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
            

        
        ###################################################
        #  TEST
        ###################################################
        #Load model
        if args.arch == 'ga':
            model2 = ACMIL_GA_singletask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop)

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
        
        
        
        
        
        # Get features
        #TODO
        #Add label to the feature h5, and plot umap
        def output_trained_feature_singletask(net, dataloader, ids, cohort_name, task, conf, device):
            fea, lab = get_slide_feature_singletask(net, dataloader, device)
            fea = pd.DataFrame(fea.cpu())
            fea.to_hdf(outdir6 + cohort_name + "_feature.h5", key='feature', mode='w')
            ids_df = pd.DataFrame(ids)
            ids_df.to_hdf(outdir6 + cohort_name + "_feature.h5", key='id', mode='a')
            label_df = pd.DataFrame(lab.detach().cpu())
            label_df.to_hdf(outdir6 + cohort_name + "_feature.h5", key='label', mode='a')
            
            #combine all
            comb_df = pd.concat([ids_df,label_df,fea], axis = 1)
            
            return comb_df
        
        # #fLod eature
        # feature_df = pd.read_hdf(os.path.join(outdir6 + "Train_feature.h5"), key='feature')
        # feature_df.columns = feature_df.columns.astype(str)
        # feature_df.reset_index(drop = True, inplace = True)
        
        comb_df_train = output_trained_feature_singletask(model2, train_loader, train_ids, train_name, 0, conf, device)
        comb_df_val = output_trained_feature_singletask(model2, val_loader, val_ids, val_name, 0, conf, device)
        comb_df_test1 = output_trained_feature_singletask(model2, test_loader1, test_ids1, "TEST_" + test_name1, 0, conf, device)
        comb_df_test2 = output_trained_feature_singletask(model2, test_loader2, test_ids2, "TEST_"+ test_name2, 0, conf, device)
        if  len(test_ids3) > 0:
            comb_df_test3 = output_trained_feature_singletask(model2, test_loader3, test_ids3, "TEST_"+ test_name3, 0, conf, device)
        comb_df_test = output_trained_feature_singletask(model2, test_loader, test_ids, "TEST_COMB", 0, conf, device)
        
        if len(ext_ids0) > 0:
            comb_df_ext_st0 = output_trained_feature_singletask(model2, ext_loader_st0, ext_ids0, "EXT_" + ext_name1 + "_st0", 0, conf, device)
            comb_df_ext_st1 = output_trained_feature_singletask(model2, ext_loader_st1, ext_ids1, "EXT_" + ext_name2 + "_st1", 0, conf, device)
            #comb_df_nep = output_trained_feature_singletask(model2, ext_loader_union, ext_ids, ext_name + "union", 0, conf, device)
            


        # VAL
        output_pred_perf_with_logit_singletask(model2, val_loader, val_ids, selected_label, conf, val_name, criterion, out_path_pred, out_path_pref, criterion_da, device)

        
        # Test 1
        output_pred_perf_with_logit_singletask(model2, test_loader1, test_ids1, selected_label, conf, "TEST_" + test_name1, criterion, out_path_pred, out_path_pref, criterion_da, device)
        
        # Test 2
        output_pred_perf_with_logit_singletask(model2, test_loader2, test_ids2, selected_label, conf, "TEST_"+ test_name2 , criterion, out_path_pred, out_path_pref, criterion_da, device)
        
        # Test 3
        if  len(test_ids3) > 0:
            output_pred_perf_with_logit_singletask(model2, test_loader3, test_ids3, selected_label, conf, "TEST_"+ test_name3 , criterion, out_path_pred, out_path_pref, criterion_da, device)


        #Test Comb
        output_pred_perf_with_logit_singletask(model2, test_loader, test_ids, selected_label, conf, "TEST_COMB", criterion, out_path_pred, out_path_pref, criterion_da, device)

        

        if len(ext_ids0) > 0:
            #External Validation 1 (z_nostnorm_nep)
            output_pred_perf_with_logit_singletask(model2, ext_loader_st0, ext_ids0, selected_label, conf, "EXT_" + ext_name1 + "_st0", criterion, out_path_pred, out_path_pref, criterion_da, device)
    
            
            #External Validation 2 (normed nep)
            output_pred_perf_with_logit_singletask(model2, ext_loader_st1, ext_ids1, selected_label, conf, "EXT_" + ext_name2 + "_st1", criterion, out_path_pred, out_path_pref, criterion_da, device)
            
            
        #External Validation 3 (union)
        #output_pred_perf_with_logit_singletask(model2, ext_loader_union, ext_ids, selected_label, conf, "NEP_union" , criterion, out_path_pred, out_path_pref, criterion_da, device)



