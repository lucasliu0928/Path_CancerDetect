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
#get_ipython().run_line_magic('matplotlib', 'inline')


#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
from architecture.transformer import ACMIL_GA
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


# In[2]:


####################################
######      USERINPUT       ########
####################################
ALL_LABELS = ["AR","MMR (MSH2, MSH6, PMS2, MLH1, MSH3, MLH3, EPCAM)2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]

#for lab in ALL_LABELS:
SELECTED_LABEL = ["RB1"] #["AR"]
print(SELECTED_LABEL)
selected_label_index = ALL_LABELS.index(SELECTED_LABEL[0])
TRAIN_SAMPLE_SIZE = "ALLTUMORTILES"
TRAIN_OVERLAP = 100
TEST_OVERLAP = 0
SELECTED_FOLD = 0
TUMOR_FRAC_THRES = 0.9
TUMOR_FRAC_THRES_TEST = 0.9
TUMOR_FRAC_THRES_TMA = 0.9
feature_extraction_method = 'retccl'
learning_method = "acmil"
INCLUDE_TF = False
INCLUDE_CLUSTER = False
N_CLUSTERS = 4
focal_gamma = 2
focal_alpha = 0.1
SAVE_IMAGE_SIZE = 250
TMA_OVERLAP = 0


####
################################
#model Para
BATCH_SIZE  = 1
DROPOUT = 0
DIM_OUT = 128
SELECTED_MUTATION = SELECTED_LABEL[0]

if INCLUDE_TF == False and INCLUDE_CLUSTER == False:
    N_FEATURE = 2048
    feature_type = "emb_only"
elif INCLUDE_TF == True and INCLUDE_CLUSTER == False:
    N_FEATURE = 2049
    feature_type = "emb_and_tf"
elif INCLUDE_TF == False and INCLUDE_CLUSTER == True:
    N_FEATURE = 2049
    feature_type = "emb_and_cluster" + str(N_CLUSTERS)
elif INCLUDE_TF == True and INCLUDE_CLUSTER == True:
    N_FEATURE = 2050
    feature_type = "emb_and_tf_and_cluster" + str(N_CLUSTERS) 

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
#conf.n_token = 5
conf.n_class = 2
conf.wandb_mode = 'disabled'
conf.mask_drop = 0.6
conf.n_masked_patch = 0
#conf.lr = 0.000001 #change this for HR only

# Print all key-value pairs in the conf object
for key, value in conf.__dict__.items():
    print(f"{key}: {value}")


for nt in range(10):

    conf.n_token = nt 

    if nt not in [0,1]:

        ##################
        ###### DIR  ######
        ##################
        proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
        wsi_path = proj_dir + '/data/OPX/'
        folder_name = feature_extraction_method + '/MAXSS'+ str(TRAIN_SAMPLE_SIZE)  + '_TrainOL' + str(TRAIN_OVERLAP) +  '_TestOL' + str(TEST_OVERLAP) + '_TFT' + str(TUMOR_FRAC_THRES) + "/split_fold" + str(SELECTED_FOLD) + "/" 
        folder_name_test = feature_extraction_method + '/MAXSS'+ str(TRAIN_SAMPLE_SIZE)  + '_TrainOL' + str(TRAIN_OVERLAP) +  '_TestOL' + str(TEST_OVERLAP) + '_TFT' + str(TUMOR_FRAC_THRES_TEST) + "/split_fold" + str(SELECTED_FOLD) + "/" 

        in_data_path = proj_dir + 'intermediate_data/model_ready_data/feature_' + folder_name + "model_input/"
        in_data_path_test = proj_dir + 'intermediate_data/model_ready_data/feature_' + folder_name_test + "model_input/"

        in_data_path_tma = os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', 
                               "TAN_TMA_Cores/" + "IMSIZE" + str(SAVE_IMAGE_SIZE) + "_OL" + str(TMA_OVERLAP) + "/", 
                               'feature_' + feature_extraction_method, 
                               'TFT' + str(TUMOR_FRAC_THRES_TMA) + '/')

        model_data_path =  in_data_path + feature_type + "/"
        model_data_path_test =  in_data_path_test + feature_type + "/"
           
        ################################################
        #Create output-dir
        ################################################
        folder_name1 = feature_extraction_method + '/MAXSS'+ str(TRAIN_SAMPLE_SIZE)  + '_TrainOL' + str(TRAIN_OVERLAP) +  '_TestOL' + str(TEST_OVERLAP) + '_TRAINTEST_TFT' + str(TUMOR_FRAC_THRES) + '_TMA_TFT' + str(TUMOR_FRAC_THRES_TMA) + "/split_fold" + str(SELECTED_FOLD) + "/" 
        outdir0 =  proj_dir + "intermediate_data/pred_out02062025" + "/" + folder_name1 + "/DL_" + feature_type + "/" + SELECTED_MUTATION + "/" 
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
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print(device)





        ################################################
        #     Model ready data 
        ################################################
        train_data_old = torch.load(model_data_path + 'train_data.pth')
        test_data_old = torch.load(model_data_path_test + 'test_data.pth')
        val_data = torch.load(model_data_path + 'val_data.pth')

        train_ids_old = torch.load(model_data_path + 'train_ids.pth')
        test_ids_old = torch.load(model_data_path_test + 'test_ids.pth')

        train_info_old  = torch.load(model_data_path + 'train_info.pth')
        test_info_old  = torch.load(model_data_path_test + 'test_info.pth')

        new_data = torch.load(model_data_path_test + 'newMSI_test_data.pth')
        new_ids = torch.load(model_data_path_test + 'newMSI_test_ids.pth')
        new_info  = torch.load(model_data_path_test + 'newMSI_test_info.pth')



        tma_data = torch.load(in_data_path_tma + 'tma_data.pth')
        tma_ids = torch.load(in_data_path_tma + 'tma_ids.pth')
        tma_info  = torch.load(in_data_path_tma + 'tma_info.pth')


        # In[4]:


        ################################################
        # #Update tma test , exclude no label tmas
        ################################################
        haslabel_indexes = []
        for i in range(len(tma_data)):
            if torch.isnan(tma_data[i][1]).all() == False:
                #print(f"Item {i} has the second element all NaNs.")
                haslabel_indexes.append(i)


        tma_data = Subset(tma_data, haslabel_indexes)
        tma_ids = list(Subset(tma_ids, haslabel_indexes))
        tma_info = list(Subset(tma_info, haslabel_indexes))
        len(tma_info) #355 if TF0.9, a lot of cores does not have enough cancer tiles > 0.9


        # In[5]:


        ################################################
        #Exclude OPX_085, Prostate cancer find in colorectal site, patterns are for CRC, not for prostate
        ################################################
        exc_idx = test_ids_old.index('OPX_085')
        inc_idx = [i for i in range(len(test_data_old)) if i not in [exc_idx]]

        #Update old testset
        test_data_old = Subset(test_data_old, inc_idx)
        removed_id =   test_ids_old.pop(exc_idx)  
        removed_info = test_info_old.pop(exc_idx)  


        # In[6]:


        train_add_ids = ['OPX_207','OPX_209','OPX_213','OPX_214','OPX_215']
        test_add_ids =  [x for x in new_ids if x not in train_add_ids]
        print(train_add_ids)
        print(test_add_ids)


        # In[7]:


        ################################################
        #Add Ids in train 
        ################################################
        inc_idx = [new_ids.index(x) for x in train_add_ids]
        new_data_train = Subset(new_data, inc_idx)
        new_id_train =  list(Subset(new_ids, inc_idx))
        new_info_train = list(Subset(new_info, inc_idx))

        #Combine old and new train data
        train_data  = ConcatDataset([train_data_old, new_data_train])
        train_ids = train_ids_old +  new_id_train
        train_info = train_info_old +  new_info_train


        # In[8]:


        ################################################
        #Add Ids in test 
        ################################################
        inc_idx = [new_ids.index(x) for x in test_add_ids]
        new_data_test = Subset(new_data, inc_idx)
        new_id_test =  list(Subset(new_ids, inc_idx))
        new_info_test = list(Subset(new_info, inc_idx))

        #Combine old and new train data
        test_data  = ConcatDataset([test_data_old, new_data_test])
        test_ids = test_ids_old +  new_id_test
        test_info = test_info_old +  new_info_test


        # In[9]:


        #count labels in train
        train_label_counts = [dt[1] for dt in train_data]
        train_label_counts = torch.concat(train_label_counts)
        count_ones = (train_label_counts == 1).sum(dim=0)
        print(count_ones)
        perc_ones = count_ones/train_label_counts.shape[0] * 100
        formatted_numbers = [f"{x.item():.1f}" for x in perc_ones]
        print(formatted_numbers)


        # In[10]:


        #count labels in test
        test_label_counts = [dt[1] for dt in test_data]
        test_label_counts = torch.concat(test_label_counts)
        count_ones = (test_label_counts == 1).sum(dim=0)
        print(count_ones)
        perc_ones = count_ones/test_label_counts.shape[0] * 100
        formatted_numbers = [f"{x.item():.1f}" for x in perc_ones]
        print(formatted_numbers)


        # In[11]:


        #count labels in tma
        tma_label_counts = [dt[1] for dt in tma_data] 
        tma_label_counts = torch.concat(tma_label_counts)
        count_ones = (tma_label_counts == 1).sum(dim=0)
        print(count_ones)
        perc_ones = count_ones/tma_label_counts.shape[0] * 100
        formatted_numbers = [f"{x.item():.1f}" for x in perc_ones]
        print(formatted_numbers) #["AR","PTEN","RB1","TP53"


        # In[12]:


        print(len(train_data))
        print(len(val_data))
        len(test_data)


        # In[13]:


        ####################################################
        #            Train 
        ####################################################
        set_seed(0)
        #Dataloader for training
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)
        tma_loader = DataLoader(dataset=tma_data, batch_size=BATCH_SIZE, shuffle=False)


        # In[14]:


        arch = 'ga'
        # define network
        if arch == 'ga':
            model = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=0, mask_drop=0.6)
        else:
            model = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
        model.to(device)


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
                    
        # Example usage:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        #criterion = nn.CrossEntropyLoss()

        # define optimizer, lr not important at this point
        optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.wd)


        # In[15]:


        def train_one_epoch(model, criterion, data_loader, optimizer0, device, epoch, conf, selected_label_index):
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
                labels = data[1][0,:,selected_label_index].to(device, dtype = torch.int64).to(device)
                tf = data[2].to(device, dtype=torch.float32)
                
                # # Calculate and set new learning rate
                adjust_learning_rate(optimizer0, epoch + data_it/len(data_loader), conf)

                # Compute loss
                sub_preds, slide_preds, attn = model(image_patches)
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

                #ATT loss
                # avg_attn = attn.mean(dim = 1) #Across tokens
                # loss2 = F.mse_loss(avg_attn, tf)
                #for each token
                # loss2 = 0
                # for i in range(conf.n_token):
                #     loss2 += F.mse_loss(attn[:,i,:], tf)
                    
               
                #loss = diff_loss + loss0 + loss1 + loss2
                loss = diff_loss + loss0 + loss1 
                

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


        # In[16]:


        # Disable gradient calculation during evaluation
        @torch.no_grad()
        def evaluate(net, criterion, data_loader, device, conf, header, selected_label_index):

            # Set the network to evaluation mode
            net.eval()

            y_pred = []
            y_true = []

            metric_logger = MetricLogger(delimiter="  ")

            for data in metric_logger.log_every(data_loader, 100, header):
                image_patches = data[0].to(device, dtype=torch.float32)
                labels = data[1][0,:,selected_label_index].to(device, dtype = torch.int64).to(device)
                tf = data[2].to(device, dtype=torch.float32)

                sub_preds, slide_preds, attn = net(image_patches)
                div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / attn.shape[1]
                loss = criterion(slide_preds, labels)
                pred = torch.softmax(slide_preds, dim=-1)


                acc1 = accuracy(pred, labels, topk=(1,))[0]

                metric_logger.update(loss=loss.item())
                metric_logger.update(div_loss=div_loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=labels.shape[0])

                y_pred.append(pred)
                y_true.append(labels)

            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)

            AUROC_metric = torchmetrics.AUROC(num_classes = conf.n_class, task='multiclass').to(device)
            AUROC_metric(y_pred, y_true)
            auroc = AUROC_metric.compute().item()
            F1_metric = torchmetrics.F1Score(num_classes = conf.n_class, task='multiclass').to(device)
            F1_metric(y_pred, y_true)
            f1_score = F1_metric.compute().item()

            print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
                  .format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))

            return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg


        # In[17]:


        ckpt_dir = outdir1 + SELECTED_LABEL[0] + "/"
        create_dir_if_not_exists(ckpt_dir)

        # define optimizer, lr not important at this point
        optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.wd)


        best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
        train_epoch = conf.train_epoch
        for epoch in range(train_epoch):
            train_one_epoch(model, criterion, train_loader, optimizer0, device, epoch, conf, selected_label_index)


            val_auc, val_acc, val_f1, val_loss = evaluate(model, criterion, val_loader, device, conf, 'Val', selected_label_index)
            test_auc, test_acc, test_f1, test_loss = evaluate(model, criterion, test_loader, device, conf, 'Test', selected_label_index)

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
                save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,
                    save_path=os.path.join(ckpt_dir, 'checkpoint-best.pth'))
            print('\n')


        save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,
            save_path=os.path.join(ckpt_dir, 'checkpoint-last.pth'))
        print("Results on best epoch:")
        print(best_state)
        wandb.finish()


        # In[22]:


        # Set the network to evaluation mode
        model.eval()

        y_predprob = []
        y_pred = []
        y_true = []

        metric_logger = MetricLogger(delimiter="  ")

        for data in metric_logger.log_every(test_loader, 100, None):
            image_patches = data[0].to(device, dtype=torch.float32)
            labels = data[1][0,:,selected_label_index].to(device, dtype = torch.int64).to(device)

            sub_preds, slide_preds, attn = model(image_patches)
            pred = torch.softmax(slide_preds, dim=-1)
            pred_prob = torch.softmax(slide_preds, dim=-1)[:,1]

            y_predprob.append(pred_prob)
            y_pred.append(pred)
            y_true.append(labels)
            
        y_predprob = torch.cat(y_predprob, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        AUROC_metric = torchmetrics.AUROC(num_classes = conf.n_class, task='multiclass').to(device)
        AUROC_metric(y_pred, y_true)
        auroc = AUROC_metric.compute().item()
        F1_metric = torchmetrics.F1Score(num_classes = conf.n_class, task='multiclass').to(device)
        F1_metric(y_pred, y_true)
        f1_score = F1_metric.compute().item()
        print(auroc)
        print(f1_score)


        # In[23]:


        ####################################################################################
        #Predict
        ####################################################################################

        #predicts
        test_pred_prob  = y_predprob
        test_true_label = y_true

        #Prediction df
        pred_df = pd.DataFrame({"SAMPLE_IDs":  test_ids, 
                                "Y_True": y_true.cpu().detach().numpy(), 
                                "Pred_Prob" :  test_pred_prob.cpu().detach().numpy(),
                                "OUTCOME": SELECTED_LABEL[0]})

        #Add Predict class
        save_location = outdir4 + SELECTED_LABEL[0] + "/"
        create_dir_if_not_exists(save_location)

        THRES = round(pred_df['Pred_Prob'].quantile(0.8),2)
        pred_df['Pred_Class'] = 0
        pred_df.loc[pred_df['Pred_Prob'] > THRES,'Pred_Class'] = 1
        pred_df.to_csv(save_location + "/n_token" + str(conf.n_token) + "_TEST_pred_df.csv",index = False)


        # #Compute performance
        save_location = outdir5 + SELECTED_LABEL[0] + "/"
        create_dir_if_not_exists(save_location)

        perf_df = compute_performance_each_label(SELECTED_LABEL, pred_df, "SAMPLE_LEVEL")
        print(perf_df)
        perf_df.to_csv(save_location + "/n_token" + str(conf.n_token) + "_TEST_perf.csv",index = True)


        # In[24]:


        #TMA
        # Set the network to evaluation mode
        model.eval()

        y_predprob = []
        y_pred = []
        y_true = []

        metric_logger = MetricLogger(delimiter="  ")

        for data in metric_logger.log_every(tma_loader, 100, None):
            image_patches = data[0].to(device, dtype=torch.float32)
            labels = data[1][0,:,selected_label_index].to(device, dtype = torch.int64).to(device)

            sub_preds, slide_preds, attn = model(image_patches)
            pred = torch.softmax(slide_preds, dim=-1)
            pred_prob = torch.softmax(slide_preds, dim=-1)[:,1]

            y_predprob.append(pred_prob)
            y_pred.append(pred)
            y_true.append(labels)
            
        y_predprob = torch.cat(y_predprob, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        AUROC_metric = torchmetrics.AUROC(num_classes = conf.n_class, task='multiclass').to(device)
        AUROC_metric(y_pred, y_true)
        auroc = AUROC_metric.compute().item()
        F1_metric = torchmetrics.F1Score(num_classes = conf.n_class, task='multiclass').to(device)
        F1_metric(y_pred, y_true)
        f1_score = F1_metric.compute().item()
        print(auroc)
        print(f1_score)


        # In[25]:


        #TMA predict
        test_pred_prob  = y_predprob
        test_true_label = y_true

        #Prediction df
        pred_df = pd.DataFrame({"SAMPLE_IDs":  tma_ids, 
                                "Y_True": test_true_label.cpu().detach().numpy(), 
                                "Pred_Prob" :  test_pred_prob.cpu().detach().numpy(),
                                "OUTCOME": SELECTED_LABEL[0]})

        #Add Predict class
        save_location = outdir4 + SELECTED_LABEL[0] + "/"
        create_dir_if_not_exists(save_location)

        THRES = round(pred_df['Pred_Prob'].quantile(0.8),2)
        pred_df['Pred_Class'] = 0
        pred_df.loc[pred_df['Pred_Prob'] > THRES,'Pred_Class'] = 1
        pred_df.to_csv(save_location + "/n_token" + str(conf.n_token) + "_TMA_pred_df.csv",index = False)


        # #Compute performance
        save_location = outdir5 + SELECTED_LABEL[0] + "/"
        create_dir_if_not_exists(save_location)

        perf_df = compute_performance_each_label(SELECTED_LABEL, pred_df, "SAMPLE_LEVEL")
        print(perf_df)
        perf_df.to_csv(save_location + "/n_token" + str(conf.n_token) + "_TMA_perf.csv",index = True)

