#!/usr/bin/env python
# coding: utf-8

#NOTE: use python env acmil in ACMIL folder
import sys
import os
import numpy as np
import openslide
%matplotlib inline
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import warnings
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from skimage import filters
sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists
from Utils import generate_deepzoom_tiles, extract_tile_start_end_coords, get_map_startend
from Utils import get_downsample_factor
from Utils import minmax_normalize, set_seed
from Utils import get_image_at_target_mag
from Utils import do_mask_original
from Eval import plot_LOSS, compute_performance_each_label, get_attention_and_tileinfo
from Eval import boxplot_predprob_by_mutationclass, get_performance, plot_roc_curve
from Eval import bootstrap_ci_from_df, calibrate_probs_isotonic
from train_utils import pull_tiles, FocalLoss, get_feature_idexes, get_partial_data, get_train_test_val_data
from ACMIL import ACMIL_GA_MultiTask, predict_v2, train_one_epoch_multitask, evaluate_multitask
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


#Run: python3 -u 7_train_dynamic_tiles_ACMIL_AddReg_working-MultiTasking_NewFeature_TCGA_ACMIL_UpdatedOPX.py --SELECTED_FOLD 0 --loss_method ''

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--SELECTED_FOLD', default=0, type=int, help='select fold')
parser.add_argument('--loss_method', default='', type=str, help='ATTLOSS or ''')
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--feature_extraction_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--learning_method', default='acmil', type=str, help=': e.g., acmil, abmil')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--out_folder', default= 'pred_out_040725', type=str, help='out folder name')




# ===============================================================
#     Model Para
# ===============================================================
parser.add_argument('--BATCH_SIZE', default=1, type=int, help='batch size')
#parser.add_argument('--DROPOUT', default=0, type=int, help='drop out rate')
parser.add_argument('--DIM_OUT', default=128, type=int, help='')
parser.add_argument('--SELECTED_MUTATION', default='MSI_POS', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--arch', default='ga_mt', type=str, help='e.g., ga_mt, or ga')
parser.add_argument('--use_sep_cri', action= 'store_true', default=False, help='use seperate focal parameters for each mutation')


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    ####################################
    ######      USERINPUT       ########
    ####################################
    ALL_LABEL = ["AR","HR","PTEN","RB1","TP53","TMB","MSI_POS"]
    SELECTED_FEATURE = get_feature_idexes(args.feature_extraction_method, include_tumor_fraction = False)
    N_FEATURE = len(SELECTED_FEATURE)
            
    # ===============================================================
    #     Get config
    # ===============================================================
    config_dir = "myconf.yml"
    with open(config_dir, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        conf = Struct(**c)
    conf.train_epoch = 100
    conf.D_feat = N_FEATURE
    conf.D_inner = args.DIM_OUT
    conf.n_class = 1
    conf.wandb_mode = 'disabled'
    if args.SELECTED_MUTATION == 'MT':
        SELECTED_LABEL = ALL_LABEL
        conf.n_task = 7
        select_label_index = 'ALL'
    else:
        conf.n_task = 1
        SELECTED_LABEL = [args.SELECTED_MUTATION]
        select_label_index = ALL_LABEL.index(SELECTED_LABEL[0])
    conf.lr = 0.001
    
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
    folder_name_overlap = "IMSIZE250_OL100"
    folder_name_nonoverlap = "IMSIZE250_OL0"
    feature_path_opx_train =  os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', "OPX", folder_name_overlap, 'feature_' + args.feature_extraction_method, 'TFT' + str(args.TUMOR_FRAC_THRES))
    feature_path_opx_test =  os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', "OPX", folder_name_nonoverlap, 'feature_' + args.feature_extraction_method, 'TFT' + str(args.TUMOR_FRAC_THRES))
    train_val_test_id_path =  os.path.join(proj_dir + 'intermediate_data/3B_Train_TEST_IDS', "OPX" ,'TFT' + str(args.TUMOR_FRAC_THRES))
    
    
    ######################
    #Create output-dir
    ######################
    folder_name1 = args.feature_extraction_method + '/TrainOL100_TestOL0_TFT' + str(args.TUMOR_FRAC_THRES)  + "/"
    outdir0 =  proj_dir + "intermediate_data/" + args.out_folder + "/" + folder_name1 + 'FOLD' + str(args.SELECTED_FOLD) + '/' + args.SELECTED_MUTATION + "/" 
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
    
    
    ################################################
    #     Model ready data 
    ################################################
    opx_data_ol100 = torch.load(feature_path_opx_train + '/OPX_data.pth')
    opx_data_ol0  = torch.load(feature_path_opx_test + '/OPX_data.pth')

    #Get Train, test, val data
    train_test_val_id_df = pd.read_csv(os.path.join(train_val_test_id_path, "train_test_split.csv"))
    train_test_val_id_df.rename(columns = {'TMB_HIGHorINTERMEDITATE': 'TMB'}, inplace = True)
    (train_data, train_ids), (val_data, val_ids), (test_data, test_ids) = get_train_test_val_data(opx_data_ol100, opx_data_ol0, train_test_val_id_df, args.SELECTED_FOLD, select_label_index)

    #Get postive freqency to choose alpha or gamma
    check = train_test_val_id_df.loc[train_test_val_id_df['FOLD' + str(2)] == 'TRAIN']
    
    # Define different values for alpha and gamma
    if args.use_sep_cri:
        alpha_values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6]  # Example alpha values
        gamma_values = [2,    2,    2,   2,  2,   2,  10]   # Example gamma values
    else:
        #Best before
        #For RB1: gamma = 2, focal_alpha = 0.1
        #For MSI: gamma = 10, focal_alpha = 0.6
        focal_gamma = 7    #harder eaxmaple
        focal_alpha = 0.75 #postive ratio
        
    # print(check['AR'].value_counts()) 
    # print(check['HR'].value_counts()) 
    # print(check['PTEN'].value_counts()) 
    # print(check['RB1'].value_counts()) 
    # print(check['TP53'].value_counts()) 
    # print(check['TMB'].value_counts()) 
    # print(check['MSI_POS'].value_counts()) 




    ####################################################
    #Dataloader for training
    ####################################################
    train_loader = DataLoader(dataset=train_data, batch_size=args.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(dataset=test_data, batch_size=args.BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(dataset=val_data, batch_size=args.BATCH_SIZE, shuffle=False)
    
    
    ####################################################
    # define network
    ####################################################
    if args.arch == 'ga':
        model = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop)
    elif args.arch == 'ga_mt':
        model = ACMIL_GA_MultiTask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
    else:
        model = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
    model.to(device)
    
    
    # Create a list of FocalLoss criteria with different alpha and gamma
    if args.use_sep_cri:
        criterion = [FocalLoss(alpha=a, gamma=g, reduction='mean') for a, g in zip(alpha_values, gamma_values)]
    else:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
    
    # define optimizer, lr not important at this point
    optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.wd)
    

    ####################################################
    #            Train 
    ####################################################
    set_seed(0)
    # define optimizer, lr not important at this point
    optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=conf.lr, weight_decay=conf.wd)
    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    train_epoch = conf.train_epoch
    for epoch in range(train_epoch):
        train_one_epoch_multitask(model, criterion, train_loader, optimizer0, device, epoch, conf, args.loss_method, use_sep_criterion = args.use_sep_cri)
        val_auc, val_acc, val_f1, val_loss = evaluate_multitask(model, criterion, val_loader, device, conf, 'Val', use_sep_criterion = args.use_sep_cri)
        test_auc, test_acc, test_f1, test_loss = evaluate_multitask(model, criterion, test_loader, device, conf, 'Test', use_sep_criterion = args.use_sep_cri)
    
        save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,
            save_path=os.path.join(outdir1 + 'checkpoint_' + 'epoch' + str(epoch) + '.pth'))
    print("Results on best epoch:")
    print(best_state)
    wandb.finish()
    
    
    
    ###################################################
    #           Test 
    ###################################################   
    # define network
    if args.arch == 'ga':
        model2 = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop)
    elif args.arch == 'ga_mt':
        model2 = ACMIL_GA_MultiTask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
    else:
        model2 = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
    model2.to(device)
    
    # Load the checkpoint
    #checkpoint = torch.load(ckpt_dir + 'checkpoint-best.pth')
    checkpoint = torch.load(outdir1 + 'checkpoint_epoch99.pth')
    
    # Load the state_dict into the model
    model2.load_state_dict(checkpoint['model'])

    y_pred_tasks_test, y_predprob_task_test, y_true_task_test = predict_v2(model2, test_loader, device, conf, 'Test')
    y_pred_tasks_val,  y_predprob_task_val, y_true_task_val = predict_v2(model2, val_loader, device, conf, 'Test')
    
    
    pred_df_list = []
    perf_df_list = []
    for i in range(conf.n_task):
        #Calibration
        prob_calibrated = calibrate_probs_isotonic(y_true_task_val[i], y_predprob_task_val[i], y_predprob_task_test[i])
        pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], test_ids, SELECTED_LABEL[i], THRES = np.quantile(y_predprob_task_test[i],0.5))
        pred_df_list.append(pred_df)
        perf_df_list.append(perf_df)
    
    all_perd_df = pd.concat(pred_df_list)
    all_perf_df = pd.concat(perf_df_list)
    print(all_perf_df)
    
    all_perd_df.to_csv(outdir4 + "/n_token" + str(conf.n_token) + "_TEST_pred_df.csv",index = False)
    all_perf_df.to_csv(outdir5 + "/n_token" + str(conf.n_token) + "_TEST_perf.csv",index = True)
    print(round(all_perf_df['AUC'].mean(),2))

    #bootstrap perforance
    ci_list = []
    for i in range(conf.n_task):
        print(i)
        cur_pred_df = all_perd_df.loc[all_perd_df['OUTCOME'] == SELECTED_LABEL[i]]
        cur_ci_df = bootstrap_ci_from_df(cur_pred_df, y_true_col='Y_True', y_pred_col='Pred_Class', y_prob_col='Pred_Prob', num_bootstrap=1000, ci=95, seed=42)
        cur_ci_df['OUTCOME'] = SELECTED_LABEL[i]
        ci_list.append(cur_ci_df)
    ci_final_df = pd.concat(ci_list)
    print(ci_final_df)
    ci_final_df.to_csv(outdir5 + "/n_token" + str(conf.n_token) + "_TEST_perf_bootstrap.csv",index = True)
    
    


    #plot boxplot of pred prob by mutation class
    boxplot_predprob_by_mutationclass(all_perd_df, outdir4)

    
    #plot roc curve by mutation class
    for i in range(conf.n_task):
        cur_pred_df = all_perd_df.loc[all_perd_df['OUTCOME'] == SELECTED_LABEL[i]]
        plot_roc_curve(list(cur_pred_df['Pred_Prob']),list(cur_pred_df['Y_True']), outdir4, SELECTED_LABEL[i])
