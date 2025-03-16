#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:27:44 2024

@author: jliu6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
from architecture.network import Classifier_1fc, DimReduction, DimReduction1
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from timm.utils import accuracy
import torchmetrics
import wandb

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
            
        return outputs_list, bag_feat_list, A_out_list 

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


def predict(net, data_loader, device, conf, header):

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
            labels = label_lists[:,k].to(device, dtype = torch.float32).to(device)
            pred = torch.sigmoid(slide_preds)
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


def predict_v2(net, data_loader, device, conf, header):    
    y_pred = []
    y_true = []
    y_pred_prob = []
    # Set the network to evaluation mode
    net.eval()
    for data in data_loader:
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
            labels = label_lists[:,k].to(device, dtype = torch.float32).to(device)
            pred_prob = torch.sigmoid(slide_preds)
            pred = pred_prob[0][0].round()
            pred_list.append(pred)
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

def train_one_epoch_multitask(model, criterion, data_loader, optimizer0, device, epoch, conf, loss_method = 'none'):
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
            labels = label_lists[:,k].to(device, dtype = torch.float32).to(device)
                    
            if conf.n_token > 1:
                loss0 = criterion(sub_preds, labels.repeat_interleave(conf.n_token).unsqueeze(1))
            else:
                loss0 = torch.tensor(0.)
            loss1 = criterion(slide_preds, labels.unsqueeze(1))

            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            attn = torch.softmax(attn, dim=-1)
            
            for i in range(conf.n_token):
                for j in range(i + 1, conf.n_token):
                    diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                                conf.n_token * (conf.n_token - 1) / 2)

            if loss_method == 'ATTLOSS': #ATTLOSS
                #ATT loss
                # avg_attn = attn.mean(dim = 1) #Across tokens
                # loss2 = F.mse_loss(avg_attn, tf)
                #for each token
                loss2 = 0
                for i in range(conf.n_token):
                    loss2 += F.mse_loss(attn[:,i,:], tf)
                loss += diff_loss + loss0 + loss1 + loss2
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
            labels = label_lists[:,k].to(device, dtype = torch.float32).to(device)
            
            div_loss += torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / attn.shape[1]
            loss += criterion(slide_preds, labels.unsqueeze(1))
            pred = torch.sigmoid(slide_preds)
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
    
        AUROC_metric = torchmetrics.AUROC(num_classes = conf.n_class, task='binary').to(device)
        AUROC_metric(y_pred_each, y_true_each)
        auroc_each += AUROC_metric.compute().item()
    
        F1_metric = torchmetrics.F1Score(num_classes = conf.n_class, task='binary').to(device)
        F1_metric(y_pred_each, y_true_each.unsqueeze(1))
        f1_score_each += F1_metric.compute().item()
        print("AUROC",str(k),":",AUROC_metric.compute().item())
    auroc = auroc_each/conf.n_task
    f1_score = f1_score_each/conf.n_task

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))

    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg