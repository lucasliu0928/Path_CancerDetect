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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,fbeta_score,average_precision_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.insert(0, '../Utils/')
from data_loader import merge_data_lists, load_dataset_splits
from plot_utils import plot_umap
 
#FOR MIL-Lab
sys.path.insert(0, os.path.normpath(os.path.join(os.getcwd(), '..', '..', 'other_model_code','MIL-Lab',"src")))
from models.abmil import ABMILModel
from models.dsmil import DSMILModel
from models.transmil import TransMILModel

# source ~/.bashrc
# conda activate mil

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def plot_embeddings(all_feats, all_labels, method="pca", **kwargs):
    method = method.lower()
    
    if method == "pca":
        reducer = PCA(n_components=2, **kwargs)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, **kwargs)
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP is not installed. Install it with `pip install umap-learn`.")
        reducer = umap.UMAP(n_components=2, random_state=42, **kwargs)
    else:
        raise ValueError("Invalid method. Choose from 'pca', 'tsne', or 'umap'.")

    X_proj = reducer.fit_transform(all_feats)

    # Plot with fixed red/blue colors
    plt.figure(figsize=(6, 6))
    label_color_map = {0: "blue", 1: "red"}

    for label in np.unique(all_labels):
        plt.scatter(
            X_proj[all_labels == label, 0],
            X_proj[all_labels == label, 1],
            c=label_color_map.get(label, "gray"),  # gray if unexpected label
            label=str(label),
            s=10
        )

    plt.legend()
    plt.title(f"{method.upper()} projection of embeddings")
    plt.show()
    plt.show()

    return X_proj



def compute_performance(y_true,y_pred_prob,y_pred_class, cohort_name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel() #CM

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob, pos_label=1)
    
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
    perf_tb = pd.DataFrame({"AUC": AUC,
                            "PR_AUC":PR_AUC,
                            "Recall": Recall,
                            "Specificity":Specificity,
                            "ACC": ACC,
                            "Precision":Precision,
                            "F1": F1,
                            "F2": F2,
                            "F3": F3},index = [cohort_name])
    
    return perf_tb

def evaluate(test_data, model):
    """
    Runs inference over a (indexable) dataset of (features, label) pairs.
    Returns a DataFrame with logits, probabilities, true labels, and a predicted class.
    Assumes binary classification when creating Pred_Class (threshold=0.5 on class 1).
    """
    model.eval()
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for i in range(len(test_data)):
            x, y, tf, sl = test_data[i]        
            x = x.unsqueeze(0).to(device)      # add batch dim
            y = y.long().view(-1).to(device)

            results, _ = model(
                x,
                loss_fn=loss_fn,
                label=y,
                return_attention=True,
                return_slide_feats=True,
            )

            # results["logits"] expected shape: (1, num_classes)
            logits = results["logits"].squeeze(0).detach().cpu()  # -> (num_classes,)
            logits_list.append(logits)
            labels_list.append(int(y.detach().cpu()))

    # Stack logits once; derive probabilities from them
    logits_tensor = torch.stack(logits_list, dim=0)               # (N, C)
    probs_tensor  = torch.softmax(logits_tensor, dim=1)           # (N, C)

    # Build DataFrames with clear column names
    num_classes = logits_tensor.size(1)
    logit_cols = [f"logit_{i}" for i in range(num_classes)]
    prob_cols  = [f"prob_{i}"  for i in range(num_classes)]

    df_logits = pd.DataFrame(logits_tensor.numpy(), columns=logit_cols)
    df_probs  = pd.DataFrame(probs_tensor.numpy(),  columns=prob_cols)

    df = pd.concat([df_logits, df_probs], axis=1)
    df["True_y"] = labels_list

    # Binary head: predict class-1 using 0.5 threshold
    if num_classes >= 2:
        df["Pred_Class"] = (df["prob_1"] > 0.5).astype(int)
    else:
        # Fallback: argmax over available classes
        df["Pred_Class"] = probs_tensor.argmax(dim=1).numpy()

    return df


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, reduction='mean'):
        r'''
        if alpha = -1, gamma = 0, then it is = CE loss
        '''
        super(FocalLoss, self).__init__()
        self.alpha = alpha #alpha is for the class = 1
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits, target):
        """
        pred_logits: [N, C] raw logits (e.g., [N, 2])
        target: [N] or [N,1] with values in {0,...,C-1}
        """
        
        if not (0 <= self.alpha <= 1) and self.alpha != -1:
            raise ValueError(f"Invalid alpha value: {self.alpha}. alpha must be in the range [0,1] or -1 for ignore.")

        # if target.dim() > 1:
        #     target = target.squeeze(1)  # [N,1] -> [N]
        if target.dtype != torch.long:
            target = target.long()              # floats -> int class
    
        ce_loss = F.cross_entropy(pred_logits, target, reduction="none")
        pt = torch.exp(-ce_loss)  # pt = softmax prob of the true class
        loss =  ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha != -1:
            alpha_t = target*self.alpha + (1.0 - target)*(1.0 - self.alpha)
            loss = alpha_t*loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

        
def get_slide_embedding(indata, net):
    net.eval()
    rep_list = []
    label_list = []
    with torch.no_grad():
        for data_it, data in enumerate(indata):
            
            #Get data
            x = data[0].to(device, dtype=torch.float32).unsqueeze(0) #[1, N_Patches, N_FEATURE]
            y = data[1][0].long().view(-1).to(device)
            
            #Run model            
            results, results2 = net(
                x,
                return_slide_feats = True)
            
            rep_list.append(results2['slide_feats'].detach().cpu())
            label_list.append(y.detach().cpu())
            
    slides_rep = torch.concat(rep_list).numpy()
    labels = torch.concat(label_list).numpy()
        
    return slides_rep, labels



def extract_features(model, dataloader, device, layer_idx=2):
    """
    Extract features from a specified classifier layer using forward hooks.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model.
    dataloader : DataLoader
        DataLoader providing (xb, yb, ..., ...) batches.
    device : torch.device
        Device to run the model on.
    layer_idx : int, optional (default=2)
        Index of the layer in `model.model.classifier` to hook.

    Returns
    -------
    all_feats : torch.Tensor
        Extracted features concatenated across all batches. Shape: [N, D]
    all_labels : torch.Tensor
        Labels concatenated across all batches. Shape: [N]
    """

    all_feats = []
    all_labels = []
    features = []

    # Hook function
    def grab_layer(_, __, output):
        features.append(output.detach().cpu())

    # Register hook
    handle = model.model.classifier[layer_idx].register_forward_hook(grab_layer)

    model.eval()
    with torch.no_grad():
        for xb, yb, *_ in dataloader:  # handles datasets with extra items
            features.clear()  # reset buffer per batch
            _ = model(xb.to(device).unsqueeze(0))
            all_feats.append(features[0])   # shape [B, D]
            all_labels.append(yb.cpu())

    # Remove hook
    handle.remove()

    # Concatenate results
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0).squeeze()

    return all_feats, all_labels

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath, virchow2')
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='MSI', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--out_folder', default= 'pred_out_091325', type=str, help='out folder name')


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
    

    opx_union_ol100  = merge_data_lists(opx_ol100_nst, opx_ol100, merge_type = 'union')
    opx_union_ol0    = merge_data_lists(opx_ol0_nst, opx_ol0, merge_type = 'union')
    tcga_union_ol100 = merge_data_lists(tcga_ol100_nst, tcga_ol100, merge_type = 'union')
    tcga_union_ol0   = merge_data_lists(tcga_ol0_nst, tcga_ol0, merge_type = 'union')
    nep_union_ol100  = merge_data_lists(nep_ol100_nst, nep_ol100, merge_type = 'union')
    nep_union_ol0    = merge_data_lists(nep_ol0_nst, nep_ol0, merge_type = 'union')

    #opx_union_ol100 = ListDataset(opx_union_ol100)
    


    #Combine
    comb_ol100 = opx_union_ol100 + tcga_union_ol100 
    comb_ol0   = opx_union_ol0 + tcga_union_ol0 

    for f in fold_list:
        f = 0
        args.mutation = 'HR2'
        ####################################
        #Load data
        ####################################    
        #get train test and valid
        opx_split    =  load_dataset_splits(opx_union_ol100, opx_union_ol0, f, args.mutation)
        tcga_split   =  load_dataset_splits(tcga_union_ol100, tcga_union_ol0, f, args.mutation)
        nep_split    =  load_dataset_splits(nep_union_ol100, nep_union_ol0, f, args.mutation)
        comb_splits  =  load_dataset_splits(comb_ol100, comb_ol0, f, args.mutation)
 

        train_data, train_sp_ids, train_pt_ids, train_cohorts  = opx_split['train']
        test_data, test_sp_ids, test_pt_ids, test_cohorts = opx_split['test']
        val_data, val_sp_ids, val_pt_ids, val_cohorts = opx_split['val']
        

        test_data3, test_sp_ids3, test_pt_ids3, test_cohorts3 = tcga_split['test']
        test_data4, test_sp_ids4, test_pt_ids4, test_cohorts4 = opx_split['test']
        
        #Nep test
        test_data_nep1, test_sp_ids_nep1, test_pt_ids_nep1, test_cohorts_nep1 = nep_split['test']
        test_data_nep2, test_sp_ids_nep2, test_pt_ids_nep2, test_cohorts_nep2 = nep_split['train']
        test_data_nep3, test_sp_ids_nep3, test_pt_ids_nep3, test_cohorts_nep3 = nep_split['val']
        
        test_data2 = test_data_nep1 + test_data_nep2 + test_data_nep3
        test_sp_ids2 = test_sp_ids_nep1 +  test_sp_ids_nep2 + test_sp_ids_nep3
        test_pt_ids2 = test_pt_ids_nep1 + test_pt_ids_nep2 + test_pt_ids_nep3
        test_cohorts2 = test_cohorts_nep1 + test_cohorts_nep2 + test_cohorts_nep3



        
        
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

        # construct the model from src and load the state dict from HuggingFace
        model = create_model('abmil.base.uni_v2.pc108-24k', num_classes=2)
        model.to(device)
        in_dim = model.model.classifier.in_features
        model.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_dim, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 2),
            nn.Linear(2, 2),
        )
        model.to(device)
        
        # for p in model.parameters():
        #     p.requires_grad = False
        # for p in model.model.classifier.parameters():
        #     p.requires_grad = True
        
        

        

 
        #loss_fn = nn.CrossEntropyLoss()
        loss_fn = FocalLoss(alpha=0.8, gamma=3, reduction='mean')

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr #3e-4,
            #weight_decay=0.01
        )
        
        


        pos_data = [x for x in train_data if x[1] == 1]
        neg_data = [x for x in train_data if x[1] == 0]
        
        selected1 = pos_data[0:10] 
        selected0 = neg_data[0:10] 
        selected_all =  selected1 + selected0
        
        
        train_loader = DataLoader(dataset=train_data,batch_size=1, shuffle=False)


        # Set the network to training mode
        
        model.train()
        for epoch in range(20):
            total_loss = 0
            for data_it, data in enumerate(train_loader):
                
                #Get data
                x = data[0].to(device, dtype=torch.float32) #[1, N_Patches, N_FEATURE]
                y = data[1][0].long().view(-1).to(device)
                tf = data[2].to(device, dtype=torch.float32)            #(1, N_Patches]
                
                #Run model            
                results, _ = model(
                    x,
                    loss_fn=loss_fn,
                    label=y)
                
                loss_fn(results['logits'],y)

                #Compute loss
                loss = results['loss']
                total_loss += loss
                avg_loss = total_loss/(data_it+1)     
                
                #Backpropagate error and update parameters 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
               
            
            #Manual Logging
            log_items = [
                f"EPOCH: {epoch}",
                f"lr: {optimizer.param_groups[0]['lr']:.6f}",
                f"total_loss: {avg_loss.item():.4f}"
            ]
            print(" | ".join(log_items))
            
        
        all_final = evaluate(train_data, model)
        compute_performance(all_final['True_y'],all_final['prob_1'],all_final['Pred_Class'], "OPX_TRAIN")
        
        all_final = evaluate(test_data4, model)
        compute_performance(all_final['True_y'],all_final['prob_1'],all_final['Pred_Class'], "OPX")
        
        


        #train_x, train_y = get_slide_embedding(train_data, model)
        #test_x, test_y = get_slide_embedding(test_data, model)
        
        #get classifer features and plot
        all_feats_train, all_labels_train = extract_features(model, test_data, device, layer_idx=3)
        proj = plot_embeddings(all_feats_train, all_labels_train, method="PCA")
        
        # Scatter plot
        plt.figure(figsize=(6,6))
        scatter = plt.scatter(proj[:,0], proj[:,1], c=all_labels_train, cmap="bwr", s=20, alpha=0.8)
        plt.colorbar(scatter, label="Label")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.grid(True)
        plt.show()

        
        all_final = evaluate(test_data, model)
        compute_performance(all_final['True_y'],all_final['prob_1'],all_final['Pred_Class'], "Comb")
        
        all_final = evaluate(test_data2, model)
        compute_performance(all_final['True_y'],all_final['prob_1'],all_final['Pred_Class'], "NEP")
        
        all_final = evaluate(test_data3, model)
        compute_performance(all_final['True_y'],all_final['prob_1'],all_final['Pred_Class'], "TCGA")
        
        all_final = evaluate(test_data4, model)
        compute_performance(all_final['True_y'],all_final['prob_1'],all_final['Pred_Class'], "OPX")







