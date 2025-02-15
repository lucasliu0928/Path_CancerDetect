#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NOTE: use python env acmil in ACMIL folder
import sys
import os
import numpy as np
import openslide
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import warnings
import torch
import torch.nn as nn

from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.optim as optim
from pathlib import Path
import PIL
from skimage import filters
import random

    
sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists
from Utils import generate_deepzoom_tiles, extract_tile_start_end_coords, get_map_startend
from Utils import get_downsample_factor
from Utils import minmax_normalize, set_seed
from Utils import log_message
from Eval import compute_performance, plot_LOSS, compute_performance_each_label, get_attention_and_tileinfo
from train_utils import pull_tiles
from train_utils import ModelReadyData_diffdim, convert_to_dict, prediction_sepatt, BCE_Weighted_Reg, BCE_Weighted_Reg_focal, compute_loss_for_all_labels_sepatt
from Model import Mutation_MIL_MT_sepAtt #, Mutation_MIL_MT
warnings.filterwarnings("ignore")


#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
from utils.utils import save_model, Struct, set_seed
import yaml
import sys
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import save_model, Struct, set_seed
from datasets.datasets import build_HDF5_feat_dataset
from architecture.transformer import ACMIL_GA #ACMIL_GA
from architecture.transformer import ACMIL_MHA
import torch.nn.functional as F

from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from timm.utils import accuracy
import torchmetrics
import wandb



def predict(net, criterion, data_loader, device, conf, header):

    metric_logger = MetricLogger(delimiter="  ")
    
    y_pred = []
    y_true = []
    y_pred_prob = []
    # Set the network to evaluation mode
    net.eval()
    for data in metric_logger.log_every(data_loader, 100, header):
        image_patches = data[0].to(device, dtype=torch.float32)
        label_lists = data[1][0]
        sub_preds_list, slide_preds_list, attn_list = net(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
        
        #Compute loss for each task, then sum
        pred_list = []
        pred_prob_list = []
        for k in range(conf.n_task):
            sub_preds = sub_preds_list[k]
            slide_preds = slide_preds_list[k]
            attn = attn_list[k]
            labels = label_lists[:,k].to(device, dtype = torch.int64).to(device)
            pred = torch.softmax(slide_preds, dim=-1)
            pred_list.append(pred)
            pred_prob = torch.softmax(slide_preds, dim=-1)[:,1]
            pred_prob_list.append(pred_prob)
    
        y_pred.append(pred_list)
        y_true.append(label_lists)
        y_pred_prob.append(pred_prob_list)

    #Get prediction for each task
    y_predprob_task = []
    y_pred_tasks = []
    y_true_tasks = []
    for k in range(conf.n_task):
        y_pred_tasks.append([p[k] for p in y_pred])
        y_predprob_task.append([p[k].item() for p in y_pred_prob])
        y_true_tasks.append([t[:,k].to(device, dtype = torch.int64).item() for t in y_true])
    
    return y_pred_tasks, y_predprob_task, y_true_tasks

def get_performance(y_predprob, y_true, cohort_ids, outcome):

    #Prediction df
    pred_df = pd.DataFrame({"SAMPLE_IDs":  cohort_ids, 
                            "Y_True": y_true, 
                            "Pred_Prob" :  y_predprob,
                            "OUTCOME": outcome})
        
    THRES = round(pred_df['Pred_Prob'].quantile(0.8),2)
    pred_df['Pred_Class'] = 0
    pred_df.loc[pred_df['Pred_Prob'] > THRES,'Pred_Class'] = 1


    perf_df = compute_performance_each_label([outcome], pred_df, "SAMPLE_LEVEL")

    return pred_df, perf_df


from architecture.network import Classifier_1fc, DimReduction, DimReduction1

class Classifier_multitask(nn.Module):
    def __init__(self, n_channels, n_classes, n_tasks, droprate=0.0):
        super(Classifier_multitask, self).__init__()
        self.fc =  nn.ModuleList([nn.Linear(n_channels, n_classes) for _ in range(n_tasks)])  
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)

        out = []
        for i in range(len(self.fc)):
            out.append(self.fc[i](x))
            
        return out
        
class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN


        return A  ### K x N


class ACMIL_GA_MultiTask(nn.Module):
    def __init__(self, conf, D=128, droprate=0, n_token=1, n_masked_patch=0, mask_drop=0, n_task = 7):
        super(ACMIL_GA_MultiTask, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention_multitask = nn.ModuleList()
        for i in range(n_task):
            self.attention_multitask.append(Attention_Gated(conf.D_inner, D, n_token))
        
        self.classifier_multitask = nn.ModuleList()
        for i in range(n_task):
            classifier = nn.ModuleList()
            for j in range(n_token):
                classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
            self.classifier_multitask.append(classifier)
        
        self.n_masked_patch = n_masked_patch
        self.n_token = conf.n_token

        self.Slide_classifier_multitask = nn.ModuleList()
        for i in range(n_task):
            Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
            self.Slide_classifier_multitask.append(Slide_classifier)
            
        self.mask_drop = mask_drop


    def forward(self, x): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)

        #Each task has its own attention
        A_out_list = []
        outputs_list = []
        bag_feat_list = []
        for i, head in enumerate(self.attention_multitask): 
            A = head(x)  ## K x N
            
            if self.n_masked_patch > 0 and self.training:
                # Get the indices of the top-k largest values
                k, n = A.shape
                n_masked_patch = min(self.n_masked_patch, n)
                _, indices = torch.topk(A, n_masked_patch, dim=-1)
                rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
                masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
                random_mask = torch.ones(k, n).to(A.device)
                random_mask.scatter_(-1, masked_indices, 0)
                A = A.masked_fill(random_mask == 0, -1e9)
            
            A_out = A
            A_out_list.append(A_out.unsqueeze(0))
            
            A = F.softmax(A, dim=1)  # softmax over N
            afeat = torch.mm(A, x) ## K x L
            outputs = []
            for j, head2 in enumerate(self.classifier_multitask[i]):
                outputs.append(head2(afeat[j]))
            outputs_list.append(torch.stack(outputs, dim=0))
            
            bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
            bag_feat = torch.mm(bag_A, x)
            bag_feat_list.append(self.Slide_classifier_multitask[i](bag_feat))
            
        return outputs_list, bag_feat_list, A_out_list #torch.stack(outputs, dim=0), self.Slide_classifier(bag_feat), A_out.unsqueeze(0) #torch.stack(outputs, dim=0)

    def forward_feature(self, x, use_attention_mask=False): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N


        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        return bag_feat


class FocalLoss_withATT(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss_withATT, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.att_reg_flag = True
        self.att_reg_loss = nn.MSELoss()

    def forward(self, inputs, targets, tumor_fractions, attention_scores):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            F_loss =  F_loss.mean()
        elif self.reduction == 'sum':
            F_loss =  F_loss.sum()

        if self.att_reg_flag == True:
            attention_scores_mean = torch.softmax(attention_scores, dim=-1).mean(dim = 1) #Take the mean across all braches
            F_loss = F_loss + self.att_reg_loss(tumor_fractions, attention_scores)

        return F_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
            

def train_one_epoch_multitask(model, criterion, data_loader, optimizer0, device, epoch, conf):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100


    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        # Move input batch onto GPU if eager execution is enabled (default), else leave it on CPU
        # Data is a dict with keys `input` (patches) and `{task_name}` (labels for given task)
        image_patches = data[0].to(device, dtype=torch.float32)
        label_lists = data[1][0]
        tf = data[2].to(device, dtype=torch.float32)

        # # Calculate and set new learning rate
        adjust_learning_rate(optimizer0, epoch + data_it/len(data_loader), conf)

        # Compute loss
        sub_preds_list, slide_preds_list, attn_list = model(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
        
        #Compute loss for each task, then sum
        loss = 0
        for k in range(conf.n_task):
            sub_preds = sub_preds_list[k]
            slide_preds = slide_preds_list[k]
            attn = attn_list[k]
            labels = label_lists[:,k].to(device, dtype = torch.int64).to(device)
        
        
            
            if conf.n_token > 1:
                loss0 = criterion(sub_preds, labels.repeat_interleave(conf.n_token))
            else:
                loss0 = torch.tensor(0.)
            loss1 = criterion(slide_preds, labels)
            
            
            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            attn = torch.softmax(attn, dim=-1)
            
            for i in range(conf.n_token):
                for j in range(i + 1, conf.n_token):
                    diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                                conf.n_token * (conf.n_token - 1) / 2)

            if loss_method == 'ATTLOSS': #ATTLOSS
                pass
                #ATT loss
                # avg_attn = attn.mean(dim = 1) #Across tokens
                # loss2 = F.mse_loss(avg_attn, tf)
                #for each token
                # loss2 = 0
                # for i in range(conf.n_token):
                #     loss2 += F.mse_loss(attn[:,i,:], tf)
                # loss += diff_loss + loss0 + loss1 + loss2
            else:
                loss += diff_loss + loss0 + loss1 

        optimizer0.zero_grad()
        # Backpropagate error and update parameters
        loss.backward()
        optimizer0.step()


        metric_logger.update(lr=optimizer0.param_groups[0]['lr'])
        metric_logger.update(sub_loss=loss0.item())
        metric_logger.update(diff_loss=diff_loss.item())
        metric_logger.update(slide_loss=loss1.item())

        if conf.wandb_mode != 'disabled':
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            wandb.log({'sub_loss': loss0}, commit=False)
            wandb.log({'diff_loss': diff_loss}, commit=False)
            wandb.log({'slide_loss': loss1})
            #wandb.log({'att_loss': loss2})


# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate_multitask(net, criterion, data_loader, device, conf, header):

    # Set the network to evaluation mode
    net.eval()

    y_pred = []
    y_true = []

    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):
        image_patches = data[0].to(device, dtype=torch.float32)
        label_lists = data[1][0]
        tf = data[2].to(device, dtype=torch.float32)


        sub_preds_list, slide_preds_list, attn_list = net(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
        
        #Compute loss for each task, then sum
        loss = 0
        div_loss = 0
        pred_list = []
        acc1_list = []
        for k in range(conf.n_task):
            sub_preds = sub_preds_list[k]
            slide_preds = slide_preds_list[k]
            attn = attn_list[k]
            labels = label_lists[:,k].to(device, dtype = torch.int64).to(device)
            
            div_loss += torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / attn.shape[1]
            loss += criterion(slide_preds, labels)
            pred = torch.softmax(slide_preds, dim=-1)
            acc1 = accuracy(pred, labels, topk=(1,))[0]

            pred_list.append(pred)
            acc1_list.append(acc1)
            
        avg_acc = sum(acc1_list)/conf.n_task

        metric_logger.update(loss=loss.item())
        metric_logger.update(div_loss=div_loss.item())
        metric_logger.meters['acc1'].update(avg_acc.item(), n=labels.shape[0])

        y_pred.append(pred_list)
        y_true.append(label_lists)

    #Get prediction for each task
    y_pred_tasks = []
    y_true_tasks = []
    for k in range(conf.n_task):
        y_pred_tasks.append([p[k] for p in y_pred])
        y_true_tasks.append([t[:,k].to(device, dtype = torch.int64) for t in y_true])
    
    #get performance for each calss
    auroc_each = 0
    f1_score_each = 0
    for k in range(conf.n_task):
        y_pred_each = torch.cat(y_pred_tasks[k], dim=0)
        y_true_each = torch.cat(y_true_tasks[k], dim=0)
    
        AUROC_metric = torchmetrics.AUROC(num_classes = conf.n_class, task='multiclass').to(device)
        AUROC_metric(y_pred_each, y_true_each)
        auroc_each += AUROC_metric.compute().item()
    
        F1_metric = torchmetrics.F1Score(num_classes = conf.n_class, task='multiclass').to(device)
        F1_metric(y_pred_each, y_true_each)
        f1_score_each += F1_metric.compute().item()
        print("AUROC",str(k),":",AUROC_metric.compute().item())
    auroc = auroc_each/conf.n_task
    f1_score = f1_score_each/conf.n_task

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))

    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg




#Run: python3 -u 7_train_dynamic_tiles_ACMIL_AddReg_working-MultiTasking_NewFeature.py --feature_extraction_method uni1 --n_token 4

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Mutation Prediction")
parser.add_argument('--feature_extraction_method', default='uni1', type=str, help='feature extraction model: retccl, uni1, uni2')
parser.add_argument('--n_token', default=3,type=int)
parser.add_argument('--fold', default=0,type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    ####################################
    ######      USERINPUT       ########
    ####################################
    ALL_LABELS = ["AR","MMR (MSH2, MSH6, PMS2, MLH1, MSH3, MLH3, EPCAM)2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]
    TUMOR_FRAC_THRES = 0.9 
    feature_extraction_method = args.feature_extraction_method #retccl, uni1
    learning_method = "acmil"
    focal_gamma = 2
    focal_alpha = 0.1
    loss_method = 'REGLOSS' #ATTLOSS

    ################################
    #model Para
    BATCH_SIZE  = 1
    DROPOUT = 0
    DIM_OUT = 128
    SELECTED_MUTATION = "MT"
    SELECTED_FOLD = args.fold

    if feature_extraction_method == 'retccl':
        SELECTED_FEATURE = [str(i) for i in range(0,2048)] + ['TUMOR_PIXEL_PERC'] #If retccl 2048, if uni 1024
        N_FEATURE = 2048
    elif feature_extraction_method == 'uni1': 
        SELECTED_FEATURE = [str(i) for i in range(0,1024)] + ['TUMOR_PIXEL_PERC'] #If retccl 2048, if uni 1024
        N_FEATURE = 1024
    elif feature_extraction_method == 'uni2':
        SELECTED_FEATURE = [str(i) for i in range(0,1536)] + ['TUMOR_PIXEL_PERC'] #If retccl 2048, if uni 1024
        N_FEATURE = 1536
        
    ################################
    # get config
    config_dir = "myconf.yml"
    with open(config_dir, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        #c.update(vars(args))
        conf = Struct(**c)

    conf.train_epoch = 100
    conf.D_feat = N_FEATURE
    conf.D_inner = DIM_OUT
    conf.n_token = args.n_token
    conf.n_class = 2
    conf.wandb_mode = 'disabled'
    conf.mask_drop = 0.6
    conf.n_masked_patch = 0
    conf.n_task = 7
    #conf.lr = 0.000001 #change this for HR only

    # Print all key-value pairs in the conf object
    for key, value in conf.__dict__.items():
        print(f"{key}: {value}")
        
    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    folder_name_overlap = "IMSIZE250_OL100"
    folder_name_nonoverlap = "IMSIZE250_OL0"
    feature_path_opx_train =  os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', "OPX", folder_name_overlap, 'feature_' + feature_extraction_method, 'TFT' + str(TUMOR_FRAC_THRES))
    feature_path_opx_test =  os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', "OPX", folder_name_nonoverlap, 'feature_' + feature_extraction_method, 'TFT' + str(TUMOR_FRAC_THRES))
    feature_path_tma = os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', "TAN_TMA_Cores",folder_name_nonoverlap, 'feature_' + feature_extraction_method, 'TFT' + str(TUMOR_FRAC_THRES))
    folder_name_ids = 'uni1/TrainOL100_TestOL0_TFT' + str(TUMOR_FRAC_THRES)  + "/"
    train_val_test_id_path =  os.path.join(proj_dir + 'intermediate_data/6_Train_TEST_IDS', folder_name_ids)


    ######################
    #Create output-dir
    ################################################
    folder_name1 = feature_extraction_method + '/TrainOL100_TestOL0_TFT' + str(TUMOR_FRAC_THRES)  + "/"
    outdir0 =  proj_dir + "intermediate_data/pred_out02122025" + "/" + folder_name1 + 'FOLD' + str(SELECTED_FOLD) + '/' + SELECTED_MUTATION + "/" 
    outdir1 =  outdir0  + "/saved_model/"
    outdir2 =  outdir0  + "/model_para/"
    outdir3 =  outdir0  + "/logs/"
    outdir4 =  outdir0  + "/predictions/"
    outdir5 =  outdir0  + "/perf/"


    create_dir_if_not_exists(outdir0)
    create_dir_if_not_exists(outdir1)
    create_dir_if_not_exists(outdir2)
    create_dir_if_not_exists(outdir3)
    create_dir_if_not_exists(outdir4)
    create_dir_if_not_exists(outdir5)

    ##################
    #Select GPU
    ##################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    
    ################################################
    #     Model ready data 
    ################################################
    opx_data_ol100 = torch.load(feature_path_opx_train + '/OPX_data.pth')
    opx_ids_ol100 = torch.load(feature_path_opx_train + '/OPX_ids.pth')
    opx_info_ol100  = torch.load(feature_path_opx_train + '/OPX_info.pth')

    opx_data_ol0 = torch.load(feature_path_opx_test + '/OPX_data.pth')
    opx_ids_ol0 = torch.load(feature_path_opx_test + '/OPX_ids.pth')
    opx_info_ol0  = torch.load(feature_path_opx_test + '/OPX_info.pth')

    tma_data = torch.load(feature_path_tma + '/tma_data.pth')
    tma_ids = torch.load(feature_path_tma + '/tma_ids.pth')
    tma_info  = torch.load(feature_path_tma + '/tma_info.pth')


    ########################################################
    #Update tma
    ########################################################
    haslabel_indexes = []
    for i in range(len(tma_data)):
        if torch.isnan(tma_data[i][1]).all() == False:
            #print(f"Item {i} has the second element all NaNs.")
            haslabel_indexes.append(i)


    tma_data = Subset(tma_data, haslabel_indexes)
    tma_ids = list(Subset(tma_ids, haslabel_indexes))
    tma_info = list(Subset(tma_info, haslabel_indexes))
    len(tma_info) #355 if TF0.9, a lot of cores does not have enough cancer tiles > 0.9


    ################################################
    #Get train, test IDs
    #NOTE: this was in the old train: ['OPX_207','OPX_209','OPX_213','OPX_214','OPX_215']
    ################################################
    train_test_val_id_df = pd.read_csv(train_val_test_id_path + "train_test_split.csv")
    train_ids_all = list(train_test_val_id_df.loc[train_test_val_id_df['FOLD' + str(SELECTED_FOLD)] == 'TRAIN', 'SAMPLE_ID'])
    test_ids_all = list(train_test_val_id_df.loc[train_test_val_id_df['FOLD' + str(SELECTED_FOLD)] == 'TEST', 'SAMPLE_ID'])
    val_ids_all = list(train_test_val_id_df.loc[train_test_val_id_df['FOLD' + str(SELECTED_FOLD)] == 'VALID', 'SAMPLE_ID'])



    ################################################
    #Get Train, test, val data
    ################################################
    #Train:
    inc_idx = [opx_ids_ol100.index(x) for x in train_ids_all]
    train_data = Subset(opx_data_ol100, inc_idx)
    train_ids =  list(Subset(opx_ids_ol100, inc_idx))
    train_info = list(Subset(opx_info_ol100, inc_idx))

    #Val:
    inc_idx = [opx_ids_ol100.index(x) for x in val_ids_all]
    val_data = Subset(opx_data_ol100, inc_idx)
    val_ids =  list(Subset(opx_ids_ol100, inc_idx))
    val_info = list(Subset(opx_info_ol100, inc_idx))

    #Test:
    inc_idx = [opx_ids_ol0.index(x) for x in test_ids_all]
    test_data = Subset(opx_data_ol0, inc_idx)
    test_ids =  list(Subset(opx_ids_ol0, inc_idx))
    test_info = list(Subset(opx_info_ol0, inc_idx))



    #count labels in train
    train_label_counts = [dt[1] for dt in train_data]
    train_label_counts = torch.concat(train_label_counts)
    count_ones = (train_label_counts == 1).sum(dim=0)
    print(count_ones)
    perc_ones = count_ones/train_label_counts.shape[0] * 100
    formatted_numbers = [f"{x.item():.1f}" for x in perc_ones]
    print(formatted_numbers)

    #count labels in test
    test_label_counts = [dt[1] for dt in test_data]
    test_label_counts = torch.concat(test_label_counts)
    count_ones = (test_label_counts == 1).sum(dim=0)
    print(count_ones)
    perc_ones = count_ones/test_label_counts.shape[0] * 100
    formatted_numbers = [f"{x.item():.1f}" for x in perc_ones]
    print(formatted_numbers)

    #count labels in tma
    tma_label_counts = [dt[1] for dt in tma_data] 
    tma_label_counts = torch.concat(tma_label_counts)
    count_ones = (tma_label_counts == 1).sum(dim=0)
    print(count_ones)
    perc_ones = count_ones/tma_label_counts.shape[0] * 100
    formatted_numbers = [f"{x.item():.1f}" for x in perc_ones]
    print(formatted_numbers) #["AR","PTEN","RB1","TP53"


    print(len(train_data))
    print(len(val_data))
    print(len(test_data))



    ####################################################
    #            Train 
    ####################################################
    set_seed(0)
    #Dataloader for training
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)
    tma_loader = DataLoader(dataset=tma_data, batch_size=BATCH_SIZE, shuffle=False)




    arch = 'ga_mt'
    # define network
    if arch == 'ga':
        model = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop)
    elif arch == 'ga_mt':
        model = ACMIL_GA_MultiTask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
    else:
        model = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
    model.to(device)



    # Example usage:
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
    #criterion = nn.CrossEntropyLoss()

    # define optimizer, lr not important at this point
    optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.wd)


    ckpt_dir = outdir1 + SELECTED_MUTATION + "/"
    create_dir_if_not_exists(ckpt_dir)

    # define optimizer, lr not important at this point
    optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.wd)


    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    train_epoch = conf.train_epoch
    for epoch in range(train_epoch):
        train_one_epoch_multitask(model, criterion, train_loader, optimizer0, device, epoch, conf)

        val_auc, val_acc, val_f1, val_loss = evaluate_multitask(model, criterion, val_loader, device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss = evaluate_multitask(model, criterion, test_loader, device, conf, 'Test')
        #tma_auc, tma_acc, tma_f1, tma_loss = evaluate_multitask(model, criterion, tma_loader, device, conf, 'TMA')

        if conf.wandb_mode != 'disabled':
            wandb.log({'perf/val_acc1': val_acc}, commit=False)
            wandb.log({'perf/val_auc': val_auc}, commit=False)
            wandb.log({'perf/val_f1': val_f1}, commit=False)
            wandb.log({'perf/val_loss': val_loss}, commit=False)
            wandb.log({'perf/test_acc1': test_acc}, commit=False)
            wandb.log({'perf/test_auc': test_auc}, commit=False)
            wandb.log({'perf/test_f1': test_f1}, commit=False)
            wandb.log({'perf/test_loss': test_loss}, commit=False)


        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
            # best_state['tma_auc'] = tma_auc
            # best_state['tma_acc'] = tma_acc
            # best_state['tma_f1'] = tma_f1
            save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,
                save_path=os.path.join(ckpt_dir, 'checkpoint-best.pth'))
        print('\n')


        save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,
            save_path=os.path.join(ckpt_dir + 'checkpoint_' + 'epoch' + str(epoch) + '.pth'))
    print("Results on best epoch:")
    print(best_state)
    wandb.finish()


    # define network
    if arch == 'ga':
        model2 = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop)
    elif arch == 'ga_mt':
        model2 = ACMIL_GA_MultiTask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
    else:
        model2 = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
    model2.to(device)

    # Load the checkpoint
    #checkpoint = torch.load(ckpt_dir + 'checkpoint-best.pth')
    checkpoint = torch.load(ckpt_dir + 'checkpoint_epoch99.pth')

    # Load the state_dict into the model
    model2.load_state_dict(checkpoint['model'])


    y_pred_tasks_test, y_predprob_task_test, y_true_task_test = predict(model2, criterion, test_loader, device, conf, 'Test')
    pred_df_list = []
    perf_df_list = []
    for i in range(conf.n_task):
        pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], test_ids, ALL_LABELS[i])
        pred_df_list.append(pred_df)
        perf_df_list.append(perf_df)

    all_perd_df = pd.concat(pred_df_list)
    all_perf_df = pd.concat(perf_df_list)
    print(all_perf_df)

    all_perd_df.to_csv(outdir4 + "/n_token" + str(conf.n_token) + "_TEST_pred_df.csv",index = False)
    all_perf_df.to_csv(outdir5 + "/n_token" + str(conf.n_token) + "_TEST_perf.csv",index = True)



    print(round(all_perf_df['AUC'].mean(),2))



    y_pred_tasks_test, y_predprob_task_test, y_true_task_test = predict(model, criterion, tma_loader, device, conf, 'TMA')
    pred_df_list = []
    perf_df_list = []
    for i in range(conf.n_task):
        if i not in [1,5,6]:
            pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], tma_ids, ALL_LABELS[i])
            pred_df_list.append(pred_df)
            perf_df_list.append(perf_df)

    all_perd_df = pd.concat(pred_df_list)
    all_perf_df = pd.concat(perf_df_list)
    print(all_perf_df)
    all_perd_df.to_csv(outdir4 + "/n_token" + str(conf.n_token) + "_TMA_pred_df.csv",index = False)
    all_perf_df.to_csv(outdir5 + "/n_token" + str(conf.n_token) + "_TMA_perf.csv",index = True)



    print(round(all_perf_df['AUC'].mean(),2))

