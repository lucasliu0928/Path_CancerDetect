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
from Eval import bootstrap_ci_from_df, calibrate_probs_isotonic
from train_utils import FocalLoss, get_feature_idexes, get_selected_labels,has_seven_csv_files, get_train_test_val_data, update_label, load_model_ready_data
from train_utils import str2bool
from ACMIL import ACMIL_GA_MultiTask,ACMIL_GA_MultiTask_DA, predict_v2, train_one_epoch_multitask, evaluate_multitask, get_emebddings
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

#source other_model_code/ACMIL-main/acmil/bin/activate
# cd code_s/cancer_detection_final/
#Run: python3 -u 7_train_ACMIL_mixed2.py --GRL False --mutation HR2_MSI --cuda_device 'cuda:1' --train_flag
############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--s_fold', default=2, type=int, help='select fold')
parser.add_argument('--loss_method', default='', type=str, help='ATTLOSS or ''')
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--learning_method', default='acmil', type=str, help=': e.g., acmil, abmil')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='HR_MSI', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_overlap', default=100, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--train_cohort', default= 'z_nostnorm_TCGA_OPX', type=str, help='TCGA_PRAD or OPX or TCGA_OPX')
parser.add_argument('--external_cohort', default= 'Neptune', type=str, help='TCGA_PRAD or OPX or Neptune')
parser.add_argument('--f_alpha', default= 0.8, type=float, help='focal alpha')
parser.add_argument('--f_gamma', default= 9, type=float, help='focal gamma')
parser.add_argument('--hr_type', default= "HR2", type=str, help='HR version 1 or 2 (2 only include 3 genes)')
parser.add_argument('--GRL', type=str2bool, default=False, help='Enable Gradient Reserse Layer for domain prediciton (yes/no, true/false)')
parser.add_argument('--train_flag', action= 'store_true', default=False, help='train flag')
parser.add_argument('--out_folder', default= 'pred_out_060325_z_nostnorm_TCGA_OPX_stnorm_Nep', type=str, help='out folder name')

############################################################################################################
#     Model Para
############################################################################################################
parser.add_argument('--BATCH_SIZE', default=1, type=int, help='batch size')
#parser.add_argument('--DROPOUT', default=0, type=int, help='drop out rate')
parser.add_argument('--DIM_OUT', default=128, type=int, help='')
parser.add_argument('--train_epoch', default=100, type=int, help='')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--arch', default='ga_mt', type=str, help='e.g., ga_mt, or ga')
parser.add_argument('--use_sep_cri', action= 'store_true', default=False, help='use seperate focal parameters for each mutation')


if __name__ == '__main__':
    
    args = parser.parse_args()
    #args.GRL = True
    #args.s_fold  = 2
    #args.train_flag = False
    
    selected_folds = [2,3,4,0,1]
    selected_folds = [2]
    for s_fold in selected_folds:
        args.s_fold = s_fold
        ####################################
        ######      USERINPUT       ########
        ####################################
        #Feature
        SELECTED_FEATURE = get_feature_idexes(args.fe_method, include_tumor_fraction = False)
        N_FEATURE = len(SELECTED_FEATURE)
        
        #mutations = [ "MSI_POS","HR1", "HR2", "AR", "PTEN","RB1","TP53","TMB"]
        #mutations = ['MT']
        #mutations = ['HR_TP53']
        
        #for mut in mutations:
        #args.mutation = mut
        #Label
        SELECTED_LABEL, selected_label_index = get_selected_labels(args.mutation, args.hr_type, args.train_cohort)
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
        train_val_test_id_path1 =  os.path.join(proj_dir + 'intermediate_data/3B_Train_TEST_IDS', 
                                                "z_nostnorm_OPX" ,
                                               'TFT' + str(args.tumor_frac))
        train_val_test_id_path2 =  os.path.join(proj_dir + 'intermediate_data/3B_Train_TEST_IDS', 
                                                "z_nostnorm_TCGA_PRAD" ,
                                               'TFT' + str(args.tumor_frac))
        ######################
        #Create output-dir
        ######################
        folder_name1 = args.fe_method + '/TrainOL' + str(args.train_overlap) +  '_TestOL' + str(args.test_overlap) + '_TFT' + str(args.tumor_frac)  + "/" 
        outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                               'trainCohort_' + args.train_cohort + 'GRL' + str(args.GRL),
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
        
        
        ################################################
        #     Model ready data 
        ################################################
        data_path = proj_dir + 'intermediate_data/5_model_ready_data'
        data_ol100_opx = load_model_ready_data(data_path, "z_nostnorm_OPX", args.train_overlap, args.fe_method, args.tumor_frac)
        data_ol0_opx   = load_model_ready_data(data_path, "z_nostnorm_OPX", args.test_overlap, args.fe_method, args.tumor_frac)
        data_ol100_tcga = load_model_ready_data(data_path, "z_nostnorm_TCGA_PRAD", args.train_overlap, args.fe_method, args.tumor_frac)
        data_ol0_tcga   = load_model_ready_data(data_path, "z_nostnorm_TCGA_PRAD", args.test_overlap, args.fe_method, args.tumor_frac)
        data_ol0_nep   = load_model_ready_data(data_path, "Neptune", args.test_overlap, args.fe_method, args.tumor_frac)
    
        
        #Clean (updated label and remove reduant info)
        data_ol100_opx, _ = update_label(data_ol100_opx, selected_label_index)
        data_ol0_opx, _ = update_label(data_ol0_opx, selected_label_index)
        data_ol100_tcga, _ = update_label(data_ol100_tcga, selected_label_index)
        data_ol0_tcga, _ = update_label(data_ol0_tcga, selected_label_index)
        data_ol0_nep, nep_ids = update_label(data_ol0_nep, selected_label_index)
    
        #Get Train, test, val data
        train_test_val_id_df1 = pd.read_csv(os.path.join(train_val_test_id_path1, "train_test_split.csv"))
        train_test_val_id_df1.rename(columns = {'TMB_HIGHorINTERMEDITATE': 'TMB'}, inplace = True)
        (train_data1, train_ids1), (val_data1, val_ids1), (test_data1, test_ids1) = get_train_test_val_data(data_ol100_opx, data_ol0_opx, 
                                                                                                      train_test_val_id_df1, 
                                                                                                      args.s_fold)
        #add domain label:
        train_data1 = [item[:3] + (torch.tensor(1.0),) + item[3:] for item in train_data1] #1 for OPX, biopsy
            
        train_test_val_id_df2 = pd.read_csv(os.path.join(train_val_test_id_path2, "train_test_split.csv"))
        train_test_val_id_df2.rename(columns = {'TMB_HIGHorINTERMEDITATE': 'TMB'}, inplace = True)
        (train_data2, train_ids2), (val_data2, val_ids2), (test_data2, test_ids2) = get_train_test_val_data(data_ol100_tcga, data_ol0_tcga, 
                                                                                                      train_test_val_id_df2, 
                                                                                                      args.s_fold)
        train_data2 = [item[:3] + (torch.tensor(0.0),) + item[3:] for item in train_data2]  #0 for TCGA, surgical
    
        train_data = train_data1 + train_data2
        val_data = val_data1 + val_data2
        train_ids = train_ids1 + train_ids2
        val_ids = val_ids1 + val_ids2
        test_data = test_data1 + test_data2 #put two test together
        test_ids = test_ids1 + test_ids2
        
        #Exclude tile info data, sample ID, patient ID, do not needed it for training
        train_data = [item[:-3] for item in train_data]
        test_data1 = [item[:-3] for item in test_data1]   
        test_data2 = [item[:-3] for item in test_data2]   
        test_data = [item[:-3] for item in test_data]   
        val_data = [item[:-3] for item in val_data]
        ext_data = [item[:-3] for item in data_ol0_nep]
        
        # ################################################################################
        # #up sampling HR
        # from collections import defaultdict

        # class_0 = []  
        # class_1 = []  
        
        # for sample in train_data:
        #     label = sample[1].squeeze().tolist()  # Convert from tensor([[a, b]]) to [a, b]
        
        #     if label[1] == 1:  # [1, 1] or [1, 0] → class 1
        #         class_1.append(sample)
        #     else:              # [0, 1] or [0, 0] → class 0
        #         class_0.append(sample)
                
                
        # if len(class_0) > len(class_1):
        #     minority = class_1
        #     majority = class_0
        # else:
        #     minority = class_0
        #     majority = class_1
        
        # diff = abs(len(class_0) - len(class_1))
        
        # import random

        # upsampled_minority = random.choices(minority, k=diff)
        
        # balanced_data = majority + minority + upsampled_minority
        # random.shuffle(balanced_data)
        # train_data = balanced_data
        # ################################################################################

        
        

        # Define different values for alpha and gamma
        if args.use_sep_cri:
            alpha_values = [0.6, 0.1, 0.1, 0.1, 0.1, 0.75, 0.75]  # Example alpha values
            gamma_values = [10,    2,    2,   2,  2,   15,  15]   # Example gamma values
        else:
            #Best before
            #For RB1: gamma = 2, focal_alpha = 0.1
            #For MSI: gamma = 10, focal_alpha = 0.6
            # focal_gamma = 2   #harder eaxmaple
            # focal_alpha = 0.8 #postive ratio
            a_g_repairs = [(0.7,0),(0.7,7),(0.2,6)]
            # gamma_list = [args.f_gamma] 
            # alpha_list = [args.f_alpha]
            #a_g_repairs = [(0.2, 6)]
            # gamma_list = [6]
            # alpha_list = [0.2]
            #a_g_repairs = [(0.2,6), (0.8,9)]
            a_g_repairs = [(0.2,6)]
    
        for focal_alpha, focal_gamma in a_g_repairs:
            outdir11 = outdir1 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir22 = outdir2 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir33 = outdir3 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir44 = outdir4 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir55 = outdir5 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir_list = [outdir11,outdir22,outdir33,outdir44,outdir55]
            for out_path in outdir_list:
                create_dir_if_not_exists(out_path)
        
            
            ####################################################
            #Dataloader for training
            ####################################################
            train_loader = DataLoader(dataset=train_data,batch_size=args.BATCH_SIZE, shuffle=False)
            test_loader1  = DataLoader(dataset=test_data1, batch_size=args.BATCH_SIZE, shuffle=False)
            test_loader2  = DataLoader(dataset=test_data2, batch_size=args.BATCH_SIZE, shuffle=False)
            test_loader  = DataLoader(dataset=test_data, batch_size=args.BATCH_SIZE, shuffle=False)
            val_loader   = DataLoader(dataset=val_data,  batch_size=args.BATCH_SIZE, shuffle=False)            
            ext_loader   = DataLoader(dataset=ext_data,  batch_size=args.BATCH_SIZE, shuffle=False)            
    
            
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
            
            
            # Create a list of FocalLoss criteria with different alpha and gamma
            if args.use_sep_cri:
                criterion = [FocalLoss(alpha=a, gamma=g, reduction='mean') for a, g in zip(alpha_values, gamma_values)]
            else:
                criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
            
            if args.GRL == False:
                criterion_da = None 
            else:
                criterion_da = nn.BCEWithLogitsLoss()
            
            
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
                    train_one_epoch_multitask(model, criterion, train_loader, optimizer0, device, epoch, conf, args.loss_method, use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                    val_auc, val_acc, val_f1, val_loss = evaluate_multitask(model, criterion, val_loader, device, conf, 'Val', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                    test_auc1, test_acc1, test_f11, test_loss1 = evaluate_multitask(model, criterion, test_loader1, device, conf, 'Test', use_sep_criterion = args.use_sep_cri, criterion_da = criterion_da)
                    #test_auc2, test_acc2, test_f12, test_loss2 = evaluate_multitask(model, criterion, test_loader2, device, conf, 'TCGA', use_sep_criterion = args.use_sep_cri)
        
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
            #           Test 1
            ###################################################   
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
            
            # Load the checkpoint
            #checkpoint = torch.load(ckpt_dir + 'checkpoint-best.pth')
            mode_idxes = args.train_epoch-1
            checkpoint = torch.load(outdir11 + 'checkpoint_epoch'+ str(mode_idxes) + '.pth')
            
            # Load the state_dict into the model
            model2.load_state_dict(checkpoint['model'])
        
            y_pred_tasks_test, y_predprob_task_test, y_true_task_test = predict_v2(model2, test_loader1, device, conf, 'Test', criterion_da= criterion_da)
            y_pred_tasks_val,  y_predprob_task_val, y_true_task_val = predict_v2(model2, val_loader, device, conf, 'Test', criterion_da = criterion_da)


            # #Get embeddings
            # embedding_opx = get_emebddings(model2, test_loader11, device, criterion_da = criterion_da)
            # embedding_tcga = get_emebddings(model2, test_loader22, device, criterion_da = criterion_da)
            # embedding_nep = get_emebddings(model2, ext_loader2, device, criterion_da = criterion_da)

            

            # from sklearn.linear_model import LogisticRegression
            # import numpy as np
        
            # # Step 1: Fit Platt scaling (logistic regression) on validation set
            # platt_model = LogisticRegression()
            # platt_model.fit(np.array(y_predprob_task_test[i]).reshape(-1, 1), np.array(y_true_task_test[i]))
            # calibrated_probs = platt_model.predict_proba(np.array(y_predprob_task_test[i]).reshape(-1, 1))[:, 1]
        
            
            pred_df_list = []
            perf_df_list = []
            for i in range(conf.n_task):
                #Calibration
                prob_calibrated = calibrate_probs_isotonic(y_true_task_val[i], y_predprob_task_val[i], y_predprob_task_test[i])
                pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], test_ids1, SELECTED_LABEL[i], THRES = np.quantile(y_predprob_task_test[i],0.5))
                pred_df_list.append(pred_df)
                perf_df_list.append(perf_df)
            
            all_perd_df = pd.concat(pred_df_list)
            all_perf_df = pd.concat(perf_df_list)
            print(all_perf_df)
            
            all_perd_df.to_csv(outdir44 + "/n_token" + str(conf.n_token) + "_OPX_pred_df.csv",index = False)
            all_perf_df.to_csv(outdir55 + "/n_token" + str(conf.n_token) + "_OPX_perf.csv",index = True)
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
            ci_final_df.to_csv(outdir55 + "/n_token" + str(conf.n_token) + "_OPX_perf_bootstrap.csv",index = True)
            
            
        
        
            #plot boxplot of pred prob by mutation class
            boxplot_predprob_by_mutationclass(all_perd_df, outdir44)
        
            
            #plot roc curve by mutation class
            for i in range(conf.n_task):
                cur_pred_df = all_perd_df.loc[all_perd_df['OUTCOME'] == SELECTED_LABEL[i]]
                plot_roc_curve(list(cur_pred_df['Pred_Prob']),list(cur_pred_df['Y_True']), outdir44, SELECTED_LABEL[i])
                
                
            ###################################################
            #           Test 2
            ###################################################       
            y_pred_tasks_test, y_predprob_task_test, y_true_task_test = predict_v2(model2, test_loader2, device, conf, 'Test', criterion_da= criterion_da)
            y_pred_tasks_val,  y_predprob_task_val, y_true_task_val = predict_v2(model2, val_loader, device, conf, 'Test', criterion_da= criterion_da)
            
            # from sklearn.linear_model import LogisticRegression
            # import numpy as np
        
            # # Step 1: Fit Platt scaling (logistic regression) on validation set
            # platt_model = LogisticRegression()
            # platt_model.fit(np.array(y_predprob_task_test[i]).reshape(-1, 1), np.array(y_true_task_test[i]))
            # calibrated_probs = platt_model.predict_proba(np.array(y_predprob_task_test[i]).reshape(-1, 1))[:, 1]
        
            
            pred_df_list = []
            perf_df_list = []
            for i in range(conf.n_task):
                #Calibration
                prob_calibrated = calibrate_probs_isotonic(y_true_task_val[i], y_predprob_task_val[i], y_predprob_task_test[i])
                pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], test_ids2, SELECTED_LABEL[i], THRES = np.quantile(y_predprob_task_test[i],0.5))
                pred_df_list.append(pred_df)
                perf_df_list.append(perf_df)
            
            all_perd_df = pd.concat(pred_df_list)
            all_perf_df = pd.concat(perf_df_list)
            print(all_perf_df)
            
            all_perd_df.to_csv(outdir44 + "/n_token" + str(conf.n_token) + "_TCGA_pred_df.csv",index = False)
            all_perf_df.to_csv(outdir55 + "/n_token" + str(conf.n_token) + "_TCGA_perf.csv",index = True)
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
            ci_final_df.to_csv(outdir55 + "/n_token" + str(conf.n_token) + "_TCGA_perf_bootstrap.csv",index = True)
            
            
        
        
            #plot boxplot of pred prob by mutation class
            boxplot_predprob_by_mutationclass(all_perd_df, outdir44)
        
            
            #plot roc curve by mutation class
            for i in range(conf.n_task):
                cur_pred_df = all_perd_df.loc[all_perd_df['OUTCOME'] == SELECTED_LABEL[i]]
                plot_roc_curve(list(cur_pred_df['Pred_Prob']),list(cur_pred_df['Y_True']), outdir44, SELECTED_LABEL[i])
            
            
            ###################################################
            #Test comb
            ###################################################
            y_pred_tasks_test, y_predprob_task_test, y_true_task_test = predict_v2(model2, test_loader, device, conf, 'Test_comb', criterion_da= criterion_da)
            y_pred_tasks_val,  y_predprob_task_val, y_true_task_val = predict_v2(model2, val_loader, device, conf, 'Test', criterion_da= criterion_da)

            # from sklearn.linear_model import LogisticRegression
            # import numpy as np
        
            # # Step 1: Fit Platt scaling (logistic regression) on validation set
            # platt_model = LogisticRegression()
            # platt_model.fit(np.array(y_predprob_task_test[i]).reshape(-1, 1), np.array(y_true_task_test[i]))
            # calibrated_probs = platt_model.predict_proba(np.array(y_predprob_task_test[i]).reshape(-1, 1))[:, 1]
        
            
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
            
            all_perd_df.to_csv(outdir44 + "/n_token" + str(conf.n_token) + "_TEST_COMB_pred_df.csv",index = False)
            all_perf_df.to_csv(outdir55 + "/n_token" + str(conf.n_token) + "_TEST_COMB_perf.csv",index = True)
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
            ci_final_df.to_csv(outdir55 + "/n_token" + str(conf.n_token) + "_TEST_COMB_perf_bootstrap.csv",index = True)
            
            
        
        
            #plot boxplot of pred prob by mutation class
            boxplot_predprob_by_mutationclass(all_perd_df, outdir44)
        
            
            #plot roc curve by mutation class
            for i in range(conf.n_task):
                cur_pred_df = all_perd_df.loc[all_perd_df['OUTCOME'] == SELECTED_LABEL[i]]
                plot_roc_curve(list(cur_pred_df['Pred_Prob']),list(cur_pred_df['Y_True']), outdir44, SELECTED_LABEL[i])
            
            
            
            
            
            ###################################################
            #           external validation 
            ###################################################   
            y_pred_tasks_test, y_predprob_task_test, y_true_task_test = predict_v2(model2, ext_loader, device, conf, 'EXT', criterion_da= criterion_da)
            
            # from sklearn.linear_model import LogisticRegression
            # import numpy as np
        
            # # Step 1: Fit Platt scaling (logistic regression) on validation set
            # platt_model = LogisticRegression()
            # platt_model.fit(np.array(y_predprob_task_test[i]).reshape(-1, 1), np.array(y_true_task_test[i]))
            # calibrated_probs = platt_model.predict_proba(np.array(y_predprob_task_test[i]).reshape(-1, 1))[:, 1]
        
            
            pred_df_list = []
            perf_df_list = []
            for i in range(conf.n_task):
                #Calibration
                prob_calibrated = calibrate_probs_isotonic(y_true_task_val[i], y_predprob_task_val[i], y_predprob_task_test[i])
                pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], nep_ids, SELECTED_LABEL[i], THRES = np.quantile(y_predprob_task_test[i],0.5))
                pred_df_list.append(pred_df)
                perf_df_list.append(perf_df)
            
            all_perd_df = pd.concat(pred_df_list)
            all_perf_df = pd.concat(perf_df_list)
            print(all_perf_df)
            
            all_perd_df.to_csv(outdir44 + "/n_token" + str(conf.n_token) + "_EXT_pred_df.csv",index = False)
            all_perf_df.to_csv(outdir55 + "/n_token" + str(conf.n_token) + "_EXT_perf.csv",index = True)
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
            ci_final_df.to_csv(outdir55 + "/n_token" + str(conf.n_token) + "_EXT_perf_bootstrap.csv",index = True)
            
        
        
            #plot boxplot of pred prob by mutation class
            boxplot_predprob_by_mutationclass(all_perd_df, outdir44)
        
            
            #plot roc curve by mutation class
            for i in range(conf.n_task):
                cur_pred_df = all_perd_df.loc[all_perd_df['OUTCOME'] == SELECTED_LABEL[i]]
                plot_roc_curve(list(cur_pred_df['Pred_Prob']),list(cur_pred_df['Y_True']), outdir44, SELECTED_LABEL[i])
            
