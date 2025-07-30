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
from Eval import predict_v2, predict_v2_sp_nost_andst, output_pred_perf_with_logit
from train_utils import FocalLoss,FocalLoss_logitadj, get_feature_idexes, get_selected_labels,has_seven_csv_files, get_train_test_val_data, update_label, load_model_ready_data
from train_utils import str2bool, random_sample_tiles
from train_utils import get_larger_tumor_fraction_tile, get_matching_tile_index, combine_data_from_stnorm_and_nostnorm
from train_utils import load_data, get_final_model_data,load_model_ready_data
from ACMIL import ACMIL_GA_MultiTask,ACMIL_GA_MultiTask_DA, train_one_epoch_multitask, train_one_epoch_multitask_minibatch, evaluate_multitask, get_emebddings
from ACMIL import decouple_classifier, train_one_epoch_multitask_minibatch_randomSample,evaluate_multitask_randomSample, get_slide_feature
warnings.filterwarnings("ignore")
from Preprocessing import extract_before_second_underscore,extract_before_third_dash
from torch.utils.data import Dataset

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
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
from Eval import eval_decouple
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)          # if using GPU
    torch.cuda.manual_seed_all(seed)      # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class decouple_classifier(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(decouple_classifier, self).__init__()
        self.fc1 = nn.Linear(n_channels, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

def get_comb_feature_label(feature_path, label_data, cohort_name, mutation):
    #Load feature
    feature_df = pd.read_hdf(os.path.join(feature_path, cohort_name + "_feature.h5"), key='feature')
    feature_df.columns = feature_df.columns.astype(str)
    feature_df.reset_index(drop = True, inplace = True)
    id_df = pd.read_hdf(os.path.join(feature_path, cohort_name + "_feature.h5"), key='id')
    feature_df["ID"] = list(id_df[0])
    
    #Load label
    label_list = []
    for i, row in feature_df.iterrows():
        
        cur_id = row['ID']
        if "OPX" in cur_id:
            cur_pt_id = extract_before_second_underscore(cur_id)
        elif "TCGA" in cur_id:
            cur_pt_id = extract_before_third_dash(cur_id)
        elif "NEP" in cur_id:
            cur_pt_id = cur_id[:7]
        cur_lab = label_data.loc[label_data['PATIENT_ID'] == cur_pt_id, mutation].item()
        label_list.append(cur_lab)
    
    #Add to df
    feature_df[mutation] = label_list
        
    return feature_df


#Model Ready Data    
class ModelReadyData(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    '''
    def __init__(self,
                 df,
                 x_col,
                 y_col
                ):
        
        #get X
        self.x = torch.FloatTensor(df[x_col].to_numpy())
                        
        # Get the Y labels
        self.y = torch.LongTensor(df[y_col].to_numpy())
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x = self.x[index]
        y = self.y[index]
        
        return x, y

import matplotlib.pyplot as plt
def plot_LOSS (train_loss, valid_loss, outdir):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(valid_loss,label="Validation")
    plt.plot(train_loss,label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(outdir + 'LOSS.png')
  
    
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss_twologits(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        '''
        Focal Loss for softmax-based binary classification (2-class).

        Args:
            alpha (float or list of float): class balancing factor. If float, applies to class 1.
            gamma (float): focusing parameter.
            reduction (str): 'mean', 'sum', or 'none'.
        '''
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = [1 - alpha, alpha]
        elif isinstance(alpha, list):
            assert len(alpha) == 2, "alpha list must be of length 2"
            self.alpha = alpha
        else:
            raise TypeError("alpha must be float or list of floats")
        
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        '''
        Args:
            logits: [batch_size, 2], raw logits for class 0 and 1
            target: [batch_size], ground truth class indices (0 or 1)
        '''
        log_probs = F.log_softmax(logits, dim=1)  # [B, 2]
        probs = torch.exp(log_probs)              # [B, 2]

        target_log_probs = log_probs[range(len(target)), target]
        target_probs = probs[range(len(target)), target]

        focal_term = (1 - target_probs) ** self.gamma
        alpha_factor = torch.tensor([self.alpha[t.item()] for t in target], device=logits.device)

        loss = -alpha_factor * focal_term * target_log_probs

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # [batch_size]


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
parser.add_argument('--mutation', default='MSI', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_overlap', default=100, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--train_cohort', default= 'union_STNandNSTN_OPX_TCGA', type=str, help='TCGA or OPX or OPX_TCGA or z_nostnorm_OPX_TCGA or union_STNandNSTN_OPX_TCGA or comb_STNandNSTN_OPX_TCGA')
parser.add_argument('--out_folder', default= 'pred_out_063025_new', type=str, help='out folder name')

############################################################################################################
#Training Para 
############################################################################################################
parser.add_argument('--train_flag', type=str2bool, default=False, help='train flag')
parser.add_argument('--sample_training_n', default= 1000, type=int, help='random sample K tiles')
parser.add_argument('--train_with_samplingSTandNOST', type=str2bool, default=False, help='train flag')
parser.add_argument('--f_alpha', default= -1, type=float, help='focal alpha')
parser.add_argument('--f_gamma', default= 0, type=float, help='focal gamma')
parser.add_argument('--GRL', type=str2bool, default=False, help='Enable Gradient Reserse Layer for domain prediciton (yes/no, true/false)')


############################################################################################################
#     Model Para
############################################################################################################
parser.add_argument('--batchsize', default=32, type=int, help='training batch size')
#parser.add_argument('--DROPOUT', default=0, type=int, help='drop out rate')
parser.add_argument('--DIM_OUT', default=128, type=int, help='')
parser.add_argument('--train_epoch', default=100, type=int, help='')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

            
if __name__ == '__main__':
    set_seed(42)
    args = parser.parse_args()
    args.out_folder = 'pred_out_072925v2_decoupled'
    fold_list = [0,1,2,3,4]
    args.train_epoch = 100

    
    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    id_data_dir = os.path.join(proj_dir, 'intermediate_data', "3B_Train_TEST_IDS")
    label_data_dir = os.path.join(proj_dir, 'intermediate_data', "3A_otherinfo")
        
    #Label data
    opx_label_df = pd.read_csv(os.path.join(label_data_dir, "OPX","IMSIZE250_OL0", "all_tile_info.csv"))
    opx_label_df = opx_label_df.drop_duplicates(subset=['PATIENT_ID'])
    opx_label_df = opx_label_df[['SAMPLE_ID', 'PATIENT_ID', 'AR', 'HR2', 'PTEN', 'RB1', 'TP53', 'MSI_POS']]

    tcga_label_df = pd.read_csv(os.path.join(label_data_dir, "TCGA_PRAD","IMSIZE250_OL100", "all_tile_info.csv"))
    tcga_label_df = tcga_label_df.drop_duplicates(subset=['PATIENT_ID'])
    tcga_label_df = tcga_label_df[['SAMPLE_ID', 'PATIENT_ID', 'AR', 'HR2', 'PTEN', 'RB1', 'TP53', 'MSI_POS']]
    
    nep_label_df = pd.read_csv(os.path.join(label_data_dir, "Neptune","IMSIZE250_OL0", "all_tile_info.csv"))
    nep_label_df = nep_label_df.drop_duplicates(subset=['PATIENT_ID'])
    nep_label_df = nep_label_df[['SAMPLE_ID', 'PATIENT_ID', 'AR', 'HR2', 'PTEN', 'RB1', 'TP53' , 'MSI_POS']]

    label_df = pd.concat([opx_label_df,tcga_label_df,nep_label_df])
    label_df.rename(columns = {'MSI_POS':'MSI'}, inplace = True)
            
    #fold_list = [0,1,2,3,4]
    all_pref_df_list = []
    for f in fold_list:
        
        args.s_fold = f
        

            
        ######################
        #Create output-dir
        ######################
        folder_name1 = args.fe_method + '/TrainOL' + str(args.train_overlap) +  '_TestOL' + str(args.test_overlap) + '_TFT' + str(args.tumor_frac)  + "/" 
        outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                               'trainCohort_' + args.train_cohort + '_Samples' + str(args.sample_training_n) + '_GRL' + str(args.GRL),
                               args.learning_method,
                               folder_name1,
                               'FOLD' + str(args.s_fold),
                               args.mutation, "decoupled")
        outdir1 =  outdir0  + "/saved_model/"
        outdir2 =  outdir0  + "/model_para/"
        outdir3 =  outdir0  + "/logs/"
        outdir4 =  outdir0  + "/predictions/"
        outdir5 =  os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                               'trainCohort_' + args.train_cohort + '_Samples' + str(args.sample_training_n) + '_GRL' + str(args.GRL),
                               args.learning_method,
                               folder_name1, "perf")
        outdir_list = [outdir0,outdir1,outdir2,outdir3,outdir4,outdir5]
        
        for out_path in outdir_list:
            create_dir_if_not_exists(out_path)

        #Load feature

        
        feature_path =  os.path.join(proj_dir + "intermediate_data/" + args.out_folder.replace("_decoupled",""),
                               'trainCohort_' + args.train_cohort + '_Samples' + str(args.sample_training_n) + '_GRL' + str(args.GRL),
                               args.learning_method,
                               folder_name1,
                               'FOLD' + str(args.s_fold),
                               args.mutation, "trained_features")
        
        


        #Load feature
        train_df = get_comb_feature_label(feature_path, label_df, "Train", args.mutation)
        val_df = get_comb_feature_label(feature_path, label_df, "VAL", args.mutation)
        test_df1 = get_comb_feature_label(feature_path, label_df, "TEST_OPX", args.mutation)
        test_df2 = get_comb_feature_label(feature_path, label_df, "TEST_TCGA", args.mutation)
        test_df = get_comb_feature_label(feature_path, label_df, "TEST_COMB", args.mutation)
        nep_df_st0 = get_comb_feature_label(feature_path, label_df, "z_nostnorm_NEP", args.mutation)
        nep_df_st1 = get_comb_feature_label(feature_path, label_df, "NEP", args.mutation)
        nep_df_union = get_comb_feature_label(feature_path, label_df, "NEP_union", args.mutation)


        f_indexes = [str(x) for x in range(0,128)]
        train_data = ModelReadyData(train_df, [str(x) for x in range(0,128)],args.mutation)
        val_data = ModelReadyData(val_df, [str(x) for x in range(0,128)],args.mutation)
        test_data1 = ModelReadyData(test_df1, [str(x) for x in range(0,128)],args.mutation)
        test_data2 = ModelReadyData(test_df2, [str(x) for x in range(0,128)],args.mutation)
        test_data = ModelReadyData(test_df, [str(x) for x in range(0,128)],args.mutation)
        nep_data_st0 = ModelReadyData(nep_df_st0, [str(x) for x in range(0,128)],args.mutation)
        nep_data_st1 = ModelReadyData(nep_df_st1, [str(x) for x in range(0,128)],args.mutation)
        nep_data_union = ModelReadyData(nep_df_union, [str(x) for x in range(0,128)],args.mutation)

        ##################
        #Select GPU
        ##################
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(device)

      

        ####################################################
        #            Train 
        ####################################################
        LEARNING_RATE = 0.0001
        BATCH_SIZE  = 16
        EPOCHS = args.train_epoch #16
                               
                
        ####################################################
        #Dataloader for training
        ####################################################
        #Upsamplling
        labels = list(train_df[args.mutation])
        # Calculate class frequencies
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)


        #train_loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE, shuffle=False)
        test_loader1  = DataLoader(dataset=test_data1, batch_size=1, shuffle=False)
        test_loader2  = DataLoader(dataset=test_data2, batch_size=1, shuffle=False)
        test_loader  = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        val_loader   = DataLoader(dataset=val_data,  batch_size=1, shuffle=False)            
        ext_loader_st0   = DataLoader(dataset=nep_data_st0,  batch_size=1, shuffle=False)
        ext_loader_st1   = DataLoader(dataset=nep_data_st1,  batch_size=1, shuffle=False) 
        ext_loader_union   = DataLoader(dataset=nep_data_union,  batch_size=1, shuffle=False)
        

        #Construct model
        model = decouple_classifier(n_channels = 128, n_classes = 2, droprate = 0.8)

        #Optimizer
        import torch.optim as optim
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        #Loss
        focal_alpha = 0.2
        focal_gamma = 6
        loss_func = FocalLoss_twologits(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        #loss_func = nn.CrossEntropyLoss()
        
        # --- Compute Empirical Class Priors ---
        class_counts = np.bincount(train_data.y.numpy())
        class_priors = class_counts / class_counts.sum()  # π_y
        class_priors = torch.tensor(class_priors, dtype=torch.float32)
        tau = 0  # or tune via validation
        log_class_priors = tau * torch.log(class_priors)
        

        #train:
        train_loss = []
        valid_loss = []
        prev_loss = float("Inf")
        for epoch in range(EPOCHS):
            for x,y in train_loader:
                #zero the gradients
                optimizer.zero_grad() 
                #Forward 
                logits = model(x)
                
                adjusted_logits = logits - log_class_priors
                
                loss = loss_func(adjusted_logits,y) 
                #Backward
                loss.backward()           
                #Optimize
                optimizer.step()
        
            #Compute loss per epoch 
            #Training
            train_pred = model(train_data.x)
            cur_loss = loss_func(train_pred,train_data.y).detach().numpy().item()
            train_loss.append(cur_loss)

            
            #Validation
            with torch.no_grad():
                val_pred = model(val_data.x)
                valid_loss.append(loss_func(val_pred,val_data.y).detach().numpy())
            
            print("Epoch"+ str(epoch) + ":",
                  "Train-LOSS:" + "{:.5f}".format(train_loss[epoch]) + ", " +
                  "Valid-LOSS:" +  "{:.5f}".format(valid_loss[epoch]))
            
            #Save model parameters
            torch.save(model.state_dict(), outdir1 + "model" + str(epoch) + '.pth')
        
        plot_LOSS(train_loss, valid_loss, outdir1)
        
        
        #TEST
        

        # --- Post-hoc Logit Adjustment ---
        use_adjusted  = True  
        tau = 1
        pred_df1, perf_df1 = eval_decouple(model, train_data, list(train_df['ID']), train_data.y, args.mutation, use_adjusted  ,class_priors, tau = tau, THRES = 0.5)
        perf_df1['Cohort'] = 'Train_OPX_TCGA'
        print(perf_df1)
        
        #pred_df, perf_df = eval_decouple(model, val_data, list(val_df['ID']), val_data.y, args.mutation, use_adjusted  ,class_priors, tau = 1.0, THRES = 0.5)
        #print(perf_df)
        
        pred_df2, perf_df2 = eval_decouple(model, test_data, list(test_df['ID']), test_data.y, args.mutation, use_adjusted  ,class_priors, tau = tau, THRES = 0.5)
        perf_df2['Cohort'] = 'Test_OPX_TCGA'
        print(perf_df2)
        pred_df2.to_csv(os.path.join(outdir4, "Test_OPX_TCGA_pred_df.csv"),index = False)

        
        #pred_df, perf_df = eval_decouple(model, test_data1, list(test_df1['ID']), test_data1.y, args.mutation, use_adjusted  ,class_priors, tau = tau, THRES = 0.5)
        #print(perf_df)
        
        #pred_df, perf_df = eval_decouple(model, test_data2, list(test_df2['ID']), test_data2.y, args.mutation, use_adjusted  ,class_priors, tau = tau, THRES = 0.5)
        #print(perf_df)
        
        pred_df3, perf_df3 = eval_decouple(model, nep_data_st0, list(nep_df_st0['ID']), nep_data_st0.y, args.mutation, use_adjusted  ,class_priors, tau = tau, THRES = 0.5)
        perf_df3['Cohort'] = 'NEP_ST0'
        print(perf_df3)
        perf_df3.to_csv(os.path.join(outdir4,  "NEP_ST0_pred_df.csv"),index = False)

        
        pred_df4, perf_df4 = eval_decouple(model, nep_data_st1, list(nep_df_st1['ID']), nep_data_st1.y, args.mutation, use_adjusted  ,class_priors, tau = tau, THRES = 0.5)
        perf_df4['Cohort'] = 'NEP_ST1'
        print(perf_df4)
        pred_df4.to_csv(os.path.join(outdir4,  "NEP_ST1_pred_df.csv"),index = False)

        
        
        all_pref_df = pd.concat([perf_df1,perf_df2,perf_df3,perf_df4])
        all_pref_df['FOLD'] = f
        all_pref_df_list.append(all_pref_df)
    
    final_all_pref_df = pd.concat(all_pref_df_list)
    # Metrics to summarize
    metrics = ['AUC', 'Recall', 'Specificity', 'ACC', 'Precision', 'PR_AUC', 'F1', 'F2', 'F3']
    
    # Compute mean and std
    agg_df = final_all_pref_df.groupby("Cohort")[metrics].agg(['mean', 'std'])
    
    # Format mean ± std
    formatted_df = pd.DataFrame()
    for metric in metrics:
        formatted_df[metric] = agg_df[metric].apply(lambda x: f"{x['mean']:.2f} ± {x['std']:.2f}", axis=1)

    formatted_df.to_csv(os.path.join(outdir5, "/all_perf.csv"),index = True)


