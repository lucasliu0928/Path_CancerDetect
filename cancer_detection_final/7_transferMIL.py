#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 23:26:41 2025

@author: jliu6
"""

from src.builder import create_model
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import copy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,fbeta_score,average_precision_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


sys.path.insert(0, '../Utils/')
from data_loader import merge_data_lists, load_dataset_splits
from data_loader import combine_all, just_test, downsample, uniform_sample_all_samples
from Loss import FocalLoss, compute_logit_adjustment
from TransMIL import TransMIL
from misc_utils import str2bool
from misc_utils import create_dir_if_not_exists, set_seed
from plot_utils import plot_loss
 
#FOR MIL-Lab
sys.path.insert(0, os.path.normpath(os.path.join(os.getcwd(), '..', '..', 'other_model_code','MIL-Lab',"src")))
from models.abmil import ABMILModel
from models.dsmil import DSMILModel
from models.transmil import TransMILModel

# source ~/.bashrc
# conda activate mil






def compute_performance(y_true,y_pred_prob,y_pred_class, cohort_name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel() #CM

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob, pos_label=1)
    
    # Find best threshold using Youden's J
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]

    
    # Average precision score = PR-AUC
    PR_AUC = average_precision_score(y_true, y_pred_prob)

    AUC = round(metrics.auc(fpr, tpr),2)
    ACC = round(accuracy_score(y_true, y_pred_class),2)
    F1 = round(f1_score(y_true, y_pred_class),2)
    F2 = round(fbeta_score(y_true, y_pred_class,beta = 2),2)
    F3 = round(fbeta_score(y_true, y_pred_class,beta = 3),2)
    Recall = round(recall_score(y_true, y_pred_class),2)
    Precision = round(precision_score(y_true, y_pred_class),2)
    Specificity = round(tn / (tn + fp),2)
    perf_tb = pd.DataFrame({"best_thresh": best_thresh,
                            "AUC": AUC,
                            "PR_AUC":PR_AUC,
                            "Recall": Recall,
                            "Specificity":Specificity,
                            "ACC": ACC,
                            "Precision":Precision,
                            "F1": F1,
                            "F2": F2,
                            "F3": F3},index = [cohort_name])
    
    return perf_tb

        


def train(train_loader, 
          model, 
          criterion, 
          optimizer, 
          model_name = 'Transfer_MIL' ,
          l2_coef = 1e-4, 
          logit_adjustments = None,
          logit_adj_train = True):
    """ Run one train epoch """
    
    total_loss = 0
    model.train()
    for data_it, data in enumerate(train_loader):
        
        #Get data
        x = data[0].to(device, dtype=torch.float32) #[1, N_Patches, N_FEATURE]
        y = data[1][0].long().view(-1).to(device)
        tf = data[2].to(device, dtype=torch.float32)            #(1, N_Patches]
                        
        #Run model     
        if model_name == 'TransMIL':
            results = model(data = x)
        elif model_name == "Transfer_MIL":
            results, _ = model(x)
        
        #Get output   
        output = results['logits']
        
        #Logit adjustment 
        if logit_adj_train == True:
            output = output + logit_adjustments.to(device)
        
        #Get loss
        loss = criterion(output, y)
        
        #L2 reg
        loss_r = 0
        for parameter in model.parameters():
            loss_r += torch.sum(parameter ** 2)
        
        loss = loss +  l2_coef * loss_r

        total_loss += loss    #Store total loss 
        
        optimizer.zero_grad() #zero grad, avoid grad acuum
        loss.backward()       #compute new gradients
        optimizer.step()      #update paraters
        
    
    avg_loss = total_loss/(len(train_loader)) 
    
    return avg_loss



def validate(test_data, sample_ids, model, criterion, logit_adjustments, 
             model_name = 'Transfer_MIL', logit_adj_infer = True, logit_adj_train = False,
             l2_coef = 0, pred_thres = 0.5):
       
    model.eval()
    with torch.no_grad():
        total_loss = 0
        logits_list = []
        labels_list = []
        logits_list_adj = []
        for i in range(len(test_data)):
            x, y, tf, sl = test_data[i]        
            x = x.unsqueeze(0).to(device)      # add batch dim
            y = y.long().view(-1).to(device)
                            
            #Run model     
            if model_name == 'TransMIL':
                results = model(data = x)
            elif model_name == "Transfer_MIL":
                results, _ = model(x)
            
            #Get output   
            output = results['logits']
            logits_list.append(output.squeeze(0).detach().cpu()) #Get logit before posthoc adj
            loss = criterion(output, y) #get loss before post-hoc logit adust)
            
            #get adjust output post hoc
            if logit_adj_infer == True:
                adj_output = output - logit_adjustments.to(device) #get logit after posthoc adj
                logits_list_adj.append(adj_output.squeeze(0).detach().cpu()) #Get logit before posthoc adj
            else:
                logits_list_adj = None
                        
            #if logit adjustment used for training
            if logit_adj_train == True:
                loss = criterion(output + logit_adjustments.to(device), y)
                
            #L2 reg
            loss_r = 0
            for parameter in model.parameters():
                loss_r += torch.sum(parameter ** 2)
            loss = loss +  l2_coef * loss_r
                
            total_loss += loss #Store total loss 
            
            #get label
            labels_list.append(int(y.detach().cpu()))
        
        avg_loss = total_loss/(len(test_data)) 
        
    
    df = get_predict_df(logits_list, labels_list, sample_ids, logits_list_adj, thres = pred_thres)
    
    return avg_loss, df


def get_predict_df(logits_list, labels_list, sample_ids, logits_list_adj = None, thres = 0.5):
    
    
    #Get logits and prob df
    logits = torch.stack(logits_list, dim=0)              # (N, C)
    probs  = torch.softmax(logits, dim=1)             # (N, C)
    
    num_classes = logits.size(1)
    logit_cols = [f"logit_{i}" for i in range(num_classes)]
    prob_cols  = [f"prob_{i}"  for i in range(num_classes)]
    
    df_logits = pd.DataFrame(logits.numpy(), columns=logit_cols)
    df_probs  = pd.DataFrame(probs.numpy(),  columns=prob_cols)
    df = pd.concat([df_logits, df_probs], axis=1)
    df["Pred_Class"] = (df["prob_1"] > thres).astype(int)
    
    
    #if post hoc logit adj
    if logits_list_adj is not None:
        logits_adj = torch.stack(logits_list_adj, dim=0)               # (N, C)
        probs_adj  = torch.softmax(logits_adj, dim=1)           # (N, C)
        
        num_classes = logits_adj.size(1)
        logit_cols = [f"adj_logit_{i}" for i in range(num_classes)]
        prob_cols  = [f"adj_prob_{i}"  for i in range(num_classes)]
        
        df_logits_adj = pd.DataFrame(logits_adj.numpy(), columns=logit_cols)
        df_probs_adj  = pd.DataFrame(probs_adj.numpy(),  columns=prob_cols)
        df_adj = pd.concat([df_logits_adj, df_probs_adj], axis=1)
        df_adj["Pred_Class_adj"] = (df_adj["adj_prob_1"] > thres).astype(int)
        
        #combine ther before logit adj
        df = pd.concat([df, df_adj], axis = 1)
            
    #add label and ids
    df["True_y"] = labels_list
    df['SAMPLE_ID'] = sample_ids


    if logits_list_adj is not None:
        #reorder
        df = df[['SAMPLE_ID','True_y',
                 'logit_0', 'logit_1','prob_0', 'prob_1','Pred_Class',
                 'adj_logit_0', 'adj_logit_1','adj_prob_0', 'adj_prob_1','Pred_Class_adj']]
    else:
        #reorder
        df = df[['SAMPLE_ID','True_y',
                 'logit_0', 'logit_1','prob_0', 'prob_1','Pred_Class']]
    
    return df


def run_eval(data, sp_ids, cohort, criterion, logit_adj_infer, logit_adj_train, l2_coef=0, pred_thres = 0.5):
    avg_loss, pred_df = validate(
        data, sp_ids, model, criterion,
        logit_adjustments,
        model_name='Transfer_MIL',
        logit_adj_infer=logit_adj_infer,
        logit_adj_train=logit_adj_train,
        l2_coef = l2_coef,
        pred_thres = pred_thres
    )

    if logit_adj_infer:
        y_true, prob, pred = pred_df['True_y'], pred_df['adj_prob_1'], pred_df['Pred_Class_adj']
    else:
        y_true, prob, pred = pred_df['True_y'], pred_df['prob_1'], pred_df['Pred_Class']

    perf_tb = compute_performance(y_true, prob, pred, cohort)
    print(perf_tb)
    return avg_loss, pred_df, perf_tb




class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.num_bad_epochs = 0
        self.best_state = None

    def step(self, metric, model):
        """
        Check if validation metric improved.
        Returns True if training should stop.
        """
        improved = (self.best - metric) > self.min_delta
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience

    def __repr__(self):
        return (f"EarlyStopper(patience={self.patience}, min_delta={self.min_delta}, "
                f"best={self.best}, num_bad_epochs={self.num_bad_epochs})")

    
    

def build_model(model_name: str, device, num_classes=2, n_feature=None):
    """
    Create and return a model based on the model_name.

    Args:
        model_name (str): Name of the model type ('Transfer_MIL' or 'TransMIL').
        device (torch.device): Device to move the model to.
        num_classes (int): Number of output classes.
        n_feature (int): Required only for TransMIL.

    Returns:
        model (nn.Module): The constructed model on the specified device.
    """
    if model_name == "Transfer_MIL":
        model = create_model('abmil.base.uni_v2.pc108-24k', num_classes=num_classes)
        model.to(device)

        # in_dim = model.model.classifier.in_features
        # model.model.classifier = nn.Sequential(
        #     nn.Dropout(p=0.3),
        #     nn.Linear(in_dim, 64),
        #     nn.Linear(64, 32),
        #     nn.Linear(32, num_classes),  # last layer matches num_classes
        # )
        # model.to(device)

    elif model_name == "TransMIL":
        if n_feature is None:
            raise ValueError("n_feature must be provided for TransMIL")
        model = TransMIL(n_classes=num_classes, in_dim=n_feature)
        model.to(device)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--tumor_frac', default= 0.0, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath, virchow2')
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='MSI', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--logit_adj_train', default=True, type=str2bool, help='Train with logit adjustment')
parser.add_argument('--logit_adj_infer', default=True, type=str2bool, help='Train with logit adjustment')

parser.add_argument('--out_folder', default= 'pred_out_092925', type=str, help='out folder name')

############################################################################################################
#     Model Para
############################################################################################################
parser.add_argument('--DIM_OUT', default=512, type=int, help='')
parser.add_argument('--droprate', default=0.01, type=float, help='drop out rate')
parser.add_argument('--lr', default = 3e-4, type=float, help='learning rate') #0.01 works for DA with union , OPX + TCGA
parser.add_argument('--train_epoch', default=10, type=int, help='')

            
if __name__ == '__main__':
    
    args = parser.parse_args()
    #fold_list = [0,1,2,3,4]
    fold_list = [0]

    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    
    ######################
    #Create output-dir
    ######################
    outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                           args.mutation,
                           'FOLD' + str(fold_list[0]))
    outdir1 =  outdir0  + "/saved_model/"
    outdir2 =  outdir0  + "/model_para/"
    outdir3 =  outdir0  + "/logs/"
    outdir4 =  outdir0  + "/predictions/"
    outdir5 =  outdir0  + "/perf/"
    outdir_list = [outdir0,outdir1,outdir2,outdir3,outdir4,outdir5]
    
    for out_path in outdir_list:
        create_dir_if_not_exists(out_path)
            
    
    
    ###################################
    #Load
    ###################################
    cohorts = [
        "z_nostnorm_OPX",
        "z_nostnorm_TCGA_PRAD",
        "z_nostnorm_Neptune",
        "OPX",
        "TCGA_PRAD",
        "Neptune"
    ]
    
    data = {}
    
    for cohort_name in cohorts:
        start_time = time.time()
        base_path = os.path.join(data_dir, 
                                 cohort_name, 
                                 "IMSIZE250_{}", 
                                 f'feature_{args.fe_method}', 
                                 f"TFT{str(args.tumor_frac)}", 
                                 f'{cohort_name}_data.pth')
    
        data[f'{cohort_name}_ol100'] = torch.load(base_path.format("OL100"), weights_only = False)
        data[f'{cohort_name}_ol0'] = torch.load(base_path.format("OL0"),weights_only = False)
        
        elapsed_time = time.time() - start_time
        print(f"Time taken for {cohort_name}: {elapsed_time/60:.2f} minutes")
    
    opx_ol100_nst = data['z_nostnorm_OPX_ol100']
    opx_ol0_nst = data['z_nostnorm_OPX_ol0']
    tcga_ol100_nst = data['z_nostnorm_TCGA_PRAD_ol100']
    tcga_ol0_nst = data['z_nostnorm_TCGA_PRAD_ol0']
    nep_ol100_nst = data['z_nostnorm_Neptune_ol100']
    nep_ol0_nst = data['z_nostnorm_Neptune_ol0']
    
    opx_ol100 = data['OPX_ol100']
    opx_ol0 = data['OPX_ol0']
    tcga_ol100 = data['TCGA_PRAD_ol100']
    tcga_ol0 = data['TCGA_PRAD_ol0']
    nep_ol100 = data['Neptune_ol100']
    nep_ol0= data['Neptune_ol0']
    
        
    ##########################################################################################
    #Count tiles
    ##########################################################################################
    # outdir_0 = os.path.join(proj_dir, "intermediate_data","8_discrip_stats","tile_counts")
    # #Count AVG , MAX, MIN number of tiles
    # nep_count_df, nep_sample_count_df = count_num_tiles(nep_ol0,'Nep')
    # tcga_count_df, tcga_sample_count_df = count_num_tiles(tcga_ol0,'TCGA')
    # opx_count_df, opx_sample_count_df = count_num_tiles(opx_ol0,'OPX')
    # all_count_df = pd.concat([opx_count_df, tcga_count_df, nep_count_df])
    # all_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0.csv"))
    # nep_sample_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0_nep.csv"))
    # tcga_sample_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0_tcga.csv"))
    # opx_sample_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0_opx.csv"))
    
    # # Plot bar charts
    # plot_n_tiles_by_labels(opx_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL0_OPX.png"))
    # plot_n_tiles_by_labels(nep_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL0_NEP.png"))
    # plot_n_tiles_by_labels(tcga_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL0_TCGA.png"))

    
    # #Count AVG , MAX, MIN number of tiles
    # nep_count_df, nep_sample_count_df = count_num_tiles(nep_ol100,'Nep')
    # tcga_count_df, tcga_sample_count_df = count_num_tiles(tcga_ol100,'TCGA')
    # opx_count_df, opx_sample_count_df = count_num_tiles(opx_ol100,'OPX')
    # all_count_df = pd.concat([opx_count_df, tcga_count_df, nep_count_df])
    # all_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100.csv"))
    # nep_sample_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100_nep.csv"))
    # tcga_sample_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100_tcga.csv"))
    # opx_sample_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100_opx.csv"))
    # plot_n_tiles_by_labels(opx_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL100_OPX.png"))
    # plot_n_tiles_by_labels(nep_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL100_NEP.png"))
    # plot_n_tiles_by_labels(tcga_sample_count_df, 
    #                        label_cols=None, 
    #                        value_col="N_TILES",
    #                        agg="mean", 
    #                        save_path=os.path.join(outdir_0, "tile_bar_OL100_TCGA.png"))
    
    
    
    ##########################################################################################
    #Merge st norm and no st-norm
    ##########################################################################################
    opx_union_ol100  = merge_data_lists(opx_ol100_nst, opx_ol100, merge_type = 'union')
    opx_union_ol0    = merge_data_lists(opx_ol0_nst, opx_ol0, merge_type = 'union')
    tcga_union_ol100 = merge_data_lists(tcga_ol100_nst, tcga_ol100, merge_type = 'union')
    tcga_union_ol0   = merge_data_lists(tcga_ol0_nst, tcga_ol0, merge_type = 'union')
    nep_union_ol100  = merge_data_lists(nep_ol100_nst, nep_ol100, merge_type = 'union')
    nep_union_ol0    = merge_data_lists(nep_ol0_nst, nep_ol0, merge_type = 'union')
    
    #Combine
    comb_ol100 = opx_union_ol100 + tcga_union_ol100 
    comb_ol0   = opx_union_ol0 + tcga_union_ol0 

    for f in fold_list:
        f = 0
        ####################################
        #Load data
        ####################################    
        #get train test and valid
        opx_split    =  load_dataset_splits(opx_union_ol0, opx_union_ol0, f, args.mutation, concat_tf = False)
        tcga_split   =  load_dataset_splits(tcga_union_ol0, tcga_union_ol0, f, args.mutation, concat_tf = False)
        nep_split    =  load_dataset_splits(nep_ol0, nep_ol0, f, args.mutation, concat_tf = False)
        comb_splits  =  load_dataset_splits(comb_ol0, comb_ol0, f, args.mutation, concat_tf = False)
 

        train_data, train_sp_ids, train_pt_ids, train_cohorts, train_coords  = comb_splits['train']
        test_data, test_sp_ids, test_pt_ids, test_cohorts, _ = comb_splits['test']
        val_data, val_sp_ids, val_pt_ids, val_cohorts,_ = comb_splits['val']
        

        
        # OPX all
        test_data1, test_sp_ids1, test_pt_ids1, test_cohorts1 = combine_all(opx_split)

        # NEP all
        test_data2, test_sp_ids2, test_pt_ids2, test_cohorts2 = combine_all(nep_split)

        # TCGA all
        test_data3, test_sp_ids3, test_pt_ids3, test_cohorts3 = combine_all(tcga_split)

        # OPX / TCGA / NEP test only
        test_data4, test_sp_ids4, test_pt_ids4, test_cohorts4 = just_test(opx_split)
        test_data5, test_sp_ids5, test_pt_ids5, test_cohorts5 = just_test(tcga_split)
        test_data6, test_sp_ids6, test_pt_ids6, test_cohorts6 = just_test(nep_split)
        


        #samplling, sample could has less than 400, if original tile is <400
        train_data = uniform_sample_all_samples(train_data, train_coords, max_bag = 2000, 
                                                grid = 32, sample_by_tf = True, plot = False,
                                                tf_threshold = 0.9) 
        
        train_data, excluded_idx = (
            [x for x in train_data if len(x[0]) != 0],
            [i for i, x in enumerate(train_data) if len(x[0]) == 0]
        )  #exclude non-feature data after tf_threshold: #0.9: n = 421
        train_sp_ids = [x for i, x in enumerate(train_sp_ids) if i not in excluded_idx]

        #UMAP
        # all_feature_train, all_labels_train, site_list_train = get_feature_label_site(train_data)
        # all_feature_test, all_labels_test, site_list_test = get_feature_label_site(test_data)
        # plot_umap(all_feature_train, all_labels_train, site_list_train, train_cohorts)

       
        # train_x, train_y = get_slide_embedding(train_data)
        # test_x, test_y   = get_slide_embedding(test_data)
        # test_x2, test_y2   = get_slide_embedding(test_data2)

        # scaler = MinMaxScaler(feature_range=(0,1))
        # train_x = scaler.fit_transform(train_x)
        # test_x = scaler.fit_transform(test_x)
        # test_x2 = scaler.fit_transform(test_x2)
        
        # knn = KNeighborsClassifier(n_neighbors = 1)
        # knn.fit(train_x, train_y)
        
        # y_pred = knn.predict(train_x)
        # y_pred_prob = knn.predict_proba(train_x)[:,1]
        # compute_performance(train_y,y_pred_prob,y_pred, "OPX_TRAIN")
        
        
        # y_pred = knn.predict(test_x)
        # y_pred_prob = knn.predict_proba(test_x)[:,1]
        # compute_performance(test_y,y_pred_prob,y_pred, "OPX")
        
        # y_pred = knn.predict(test_x2)
        # y_pred_prob = knn.predict_proba(test_x2)[:,1]
        # compute_performance(test_y2,y_pred_prob,y_pred, "NEP")
        
        # plot_umap(train_x, train_y, [], [])
        

        ####################################################
        #Select GPU
        ####################################################
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(device)
                
        
        #Feature and Label N
        N_FEATURE =  train_data[0][0].shape[1]
        N_LABELS  =  train_data[0][1].shape[1]
        
        
        #ABMILModel(in_dim=1024, num_classes=2, config = None)
        
        args.lr = 1e-4
        args.logit_adj_train = False
        args.l2_coef = 5e-4
        model_name = "Transfer_MIL"
        
        # construct the model from src and load the state dict from HuggingFace 
        model = build_model(model_name = model_name, 
                    device = device, 
                    num_classes=2, 
                    n_feature = N_FEATURE)
            
        #loss_fn = FocalLoss(alpha=-1, gamma=0, reduction='mean')
        loss_fn = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
 

        
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, #3e-4,
        )
        
        
        # Scheduler 
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
        
        #train loader
        train_loader = DataLoader(dataset=train_data,batch_size=1, shuffle=False)


        # Set the network to training mode
        logit_adjustments, label_freq = compute_logit_adjustment(train_loader, tau = 0.5) #[-0.2093, -4.6176] The rarer class (1) gets a much more negative adjustment, which means during training its logits will be shifted down harder unless the model compensates.

        early_stopper = EarlyStopper(patience=10, min_delta=1e-4)  
        
                
        # avg_loss, pred_df = run_eval(train_data, train_sp_ids, "TRAIN",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data4, test_sp_ids4, "OPX_test",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data5, test_sp_ids5, "TCGA_test",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data6, test_sp_ids6, "NEP_test",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data0, test_sp_ids0, "OPX_All",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data3, test_sp_ids3, "TCGA_All",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train)
        # avg_loss, pred_df = run_eval(test_data2, test_sp_ids2, "NEP_ALL",    loss_fn,  logit_adj_infer = False, logit_adj_train = args.logit_adj_train)


        train_loss = []
        val_loss =[]
        for epoch in range(100):
            
            avg_loss  =  train(train_loader, model, loss_fn, optimizer, 
                                     model_name = 'Transfer_MIL' ,l2_coef = args.l2_coef, 
                                     logit_adjustments = logit_adjustments,
                                     logit_adj_train = args.logit_adj_train)
            lr_scheduler.step()
            
            avg_loss_val, pred_df_val = validate(val_data, val_sp_ids, model, loss_fn, 
                                         logit_adjustments, 
                                         model_name = 'Transfer_MIL',
                                         logit_adj_infer = args.logit_adj_infer,
                                         logit_adj_train = args.logit_adj_train,
                                         l2_coef = args.l2_coef)
            
            
            
            # Manual logging
            train_loss.append(avg_loss.item())
            val_loss.append(avg_loss_val.item())
            
            log_items = [
                f"EPOCH: {epoch}",
                f"lr: {optimizer.param_groups[0]['lr']:.8f}",
                f"train loss: {avg_loss.item():.4f}",
                f"val loss: {avg_loss_val.item():.4f}",
                f"best val: {early_stopper.best:.4f}",
            ]
            print(" | ".join(log_items))
            
            #Save checkpoint
            torch.save(model.state_dict(), outdir1 + "checkpoint" + str(epoch) + '.pth')
        
            # Early stop check
            if early_stopper.step(avg_loss_val.item(), model):
                print(f"Early stopping at epoch {epoch} (best val loss {early_stopper.best:.4f})")
                break
        
        plot_loss(train_loss, val_loss, outdir5)
        
        

        avg_loss, pred_df, pref_tb  = run_eval(train_data, train_sp_ids, "TRAIN",    loss_fn, logit_adj_infer = True, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        best_th = pref_tb['best_thresh'].item()
        avg_loss, pred_df, pref_tb = run_eval(test_data4, test_sp_ids4, "OPX_test",     loss_fn,logit_adj_infer = True, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef, pred_thres = best_th)
        avg_loss, pred_df, pref_tb = run_eval(test_data5, test_sp_ids5, "TCGA_test",    loss_fn, logit_adj_infer = True, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef, pred_thres = best_th )
        avg_loss, pred_df, pref_tb = run_eval(test_data6, test_sp_ids6, "NEP_test",     loss_fn,logit_adj_infer = True, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef, pred_thres = best_th )
        avg_loss, pred_df, pref_tb = run_eval(test_data1, test_sp_ids1, "OPX_All",     loss_fn,logit_adj_infer = True, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef, pred_thres = best_th )
        avg_loss, pred_df, pref_tb = run_eval(test_data3, test_sp_ids3, "TCGA_All",    loss_fn, logit_adj_infer = True, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef , pred_thres = best_th)
        avg_loss, pred_df, pref_tb = run_eval(test_data2, test_sp_ids2, "NEP_ALL",    loss_fn,  logit_adj_infer = True, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef, pred_thres = best_th )

        #get attention:
        for i in range(len(test_data4)):
            x, y, tf, sl = test_data4[i]        
            x = x.unsqueeze(0).to(device)      # add batch dim
            y = y.long().view(-1).to(device)
                            
            #Run model     
            results, log_dict = model(x,return_attention=True,
                               return_slide_feats=True)
            attention = log_dict['attention']
            s_feature = log_dict['slide_feats']
                
        # ##############################################################################
        # # DEcoupled training stage 2
        # ##############################################################################
        # mode_idxes = 43        
        # checkpoint = torch.load(os.path.join(outdir1,'checkpoint'+ str(mode_idxes) + '.pth'))
        
        # model = build_model(model_name = model_name, 
        #             device = device, 
        #             num_classes=2, 
        #             n_feature = N_FEATURE)
        
        # model.load_state_dict(checkpoint)
        
        # args.logit_adj_train = False
        
        # ###Freeze the feature extraction part
        # freeze_feature = False
        # if freeze_feature:
        #     if model_name == "Transfer_MIL":
        #         for p in model.parameters():
        #             p.requires_grad = False
        #         for p in model.model.classifier.parameters():
        #             p.requires_grad = True
        #     elif model_name == "TransMIL":
        #         for p in model.parameters():
        #             p.requires_grad = False
        #         for p in model._fc2.parameters():
        #             p.requires_grad = True
            
        # #double check
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)
        
        # args.lr = 1e-5
        # optimizer = torch.optim.AdamW(
        #     filter(lambda p: p.requires_grad, model.parameters()),
        #     lr=args.lr, #3e-4,
        # )
        
        # #loss_fn = FocalLoss(alpha=0.8, gamma=10, reduction='mean')
        # loss_fn = FocalLoss(alpha=-1, gamma=0, reduction='mean')

        
        # # Scheduler 
        # warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
        # cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-6)
        # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
        
        

        # #down sampling
        # pos_count = len([x for x in train_data if x[1] == 1])
        # print(pos_count)
        # samples = downsample(train_data, n_times=10, n_samples=pos_count, seed=42)
        
        # #train loader
        # for samp in samples:
            
        #     #Train loader
        #     train_loader = DataLoader(dataset=samp,batch_size=1, shuffle=False)
            
        #     # Scheduler 
        #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        
            
        #     # Set the network to training mode
        #     logit_adjustments, label_freq = compute_logit_adjustment(train_loader, tau = 2) #[-0.2093, -4.6176] The rarer class (1) gets a much more negative adjustment, which means during training its logits will be shifted down harder unless the model compensates.
    
    
        #     early_stopper = EarlyStopper(patience=5, min_delta=1e-4)  
            
        #     train_loss = []
        #     val_loss =[]
        #     for epoch in range(100):
                
        #         avg_loss  =  train(train_loader, model, loss_fn, optimizer, 
        #                                  model_name = 'Transfer_MIL' ,l2_coef = args.l2_coef, 
        #                                  logit_adjustments = logit_adjustments,
        #                                  logit_adj_train = args.logit_adj_train)
        #         lr_scheduler.step()
                
        #         avg_loss_val, pred_df_val = validate(val_data, val_sp_ids, model, loss_fn, 
        #                                      logit_adjustments, 
        #                                      model_name = 'Transfer_MIL',
        #                                      logit_adj_infer = args.logit_adj_infer,
        #                                      logit_adj_train = args.logit_adj_train,
        #                                      l2_coef = args.l2_coef)
                
                
                
        #         # Manual logging
        #         train_loss.append(avg_loss.item())
        #         val_loss.append(avg_loss_val.item())
                
        #         log_items = [
        #             f"EPOCH: {epoch}",
        #             f"lr: {optimizer.param_groups[0]['lr']:.8f}",
        #             f"train loss: {avg_loss.item():.4f}",
        #             f"val loss: {avg_loss_val.item():.4f}",
        #             f"best val: {early_stopper.best:.4f}",
        #         ]
        #         print(" | ".join(log_items))
                
        #         #Save checkpoint
        #         torch.save(model.state_dict(), outdir1 + "checkpoint_stage2_" + str(epoch) + '.pth')
            
        #         # Early stop check
        #         if early_stopper.step(avg_loss_val.item(), model):
        #             print(f"Early stopping at epoch {epoch} (best val loss {early_stopper.best:.4f})")
        #             break
        
        # plot_loss(train_loss, val_loss, outdir5)
        
        # avg_loss, pred_df = run_eval(train_data, train_sp_ids, "TRAIN",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        # avg_loss, pred_df = run_eval(test_data4, test_sp_ids4, "OPX_test",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        # avg_loss, pred_df = run_eval(test_data5, test_sp_ids5, "TCGA_test",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        # avg_loss, pred_df = run_eval(test_data6, test_sp_ids6, "NEP_test",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        # avg_loss, pred_df = run_eval(test_data0, test_sp_ids0, "OPX_All",     loss_fn,logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        # avg_loss, pred_df = run_eval(test_data3, test_sp_ids3, "TCGA_All",    loss_fn, logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )
        # avg_loss, pred_df = run_eval(test_data2, test_sp_ids2, "NEP_ALL",    loss_fn,  logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )

        
        
       
        
        # comb_data = test_data4 + test_data5 
        # comb_ids  = test_sp_ids4 + test_sp_ids5 
        # avg_loss, pred_df = run_eval(comb_data, comb_ids, "TCGA_OPX_NEP",    loss_fn,  logit_adj_infer = False, logit_adj_train = args.logit_adj_train, l2_coef = args.l2_coef )


        


    

# import torch
# from typing import List, Tuple

# import torch
# import pandas as pd
# import numpy as np

# def _norm(v: torch.Tensor) -> torch.Tensor:
#     return torch.nn.functional.normalize(v.float().reshape(-1), dim=0)

# def _pair_cos(a_feat: torch.Tensor, b_feat: torch.Tensor, reduce: str = "flatten") -> float:
#     """
#     Cosine similarity between two feature tensors using your reduction rules:
#     - try the requested `reduce`
#     - if flatten sizes differ, fallback to 'mean'
#     """
#     va = _to_vec(a_feat, reduce=reduce)
#     vb = _to_vec(b_feat, reduce=reduce)

#     if reduce == "flatten" and va.numel() != vb.numel():
#         # fallback to mean for both
#         va = _to_vec(a_feat, reduce="mean")
#         vb = _to_vec(b_feat, reduce="mean")

#     # If sizes *still* differ (very rare unless D differs across items), truncate to min
#     if va.numel() != vb.numel():
#         L = min(va.numel(), vb.numel())
#         va = va[:L]
#         vb = vb[:L]

#     va = _norm(va)
#     vb = _norm(vb)
#     return torch.dot(va, vb).item()

# def compute_similarity_matrix(
#     dataset,                    # List[Tuple[Tensor, Tensor, Tensor, Tensor]]
#     ids,                        # List[str] aligned with dataset
#     reduce: str = "flatten"     # or "mean"
# ):
#     n = len(dataset)
#     S = torch.empty((n, n), dtype=torch.float32)
#     for i in range(n):
#         ai = dataset[i][0]
#         # diagonal is always 1.0 (after normalization), but compute for consistency
#         for j in range(n):
#             bj = dataset[j][0]
#             S[i, j] = _pair_cos(ai, bj, reduce=reduce)
#     # DataFrame for nice labeling/CSV
#     df = pd.DataFrame(S.numpy(), index=ids, columns=ids)
#     return S, df

# # ---- Run it on your data ----
# # Assumes `test_sp_ids4` is a list of IDs and `test_data4` is aligned (same order).
# S_torch, S_df = compute_similarity_matrix(test_data4, test_sp_ids4, reduce="flatten")

# # Save as CSV and NumPy (optional)
# csv_path = "similarity_matrix.csv"
# npy_path = "similarity_matrix.npy"
# S_df.to_csv(csv_path)
# np.save(npy_path, S_torch.numpy())

# labels = [x[1].item() for x in test_data4]
# label_df = pd.DataFrame({'ids': test_sp_ids4, args.mutation: labels})


