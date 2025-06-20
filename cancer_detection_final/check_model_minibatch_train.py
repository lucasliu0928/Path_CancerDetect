#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 22:32:32 2025

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

#Gradiant reverse layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

def train_one_epoch_multitask(model, criterion, data_loader, optimizer0, device, epoch, conf, loss_method = 'none', use_sep_criterion = False, criterion_da = None):
    """
    Trains the given network for one epoch according to given criterions (loss functions)

    use_sep_criterion: use differnt focal paratermeters for each mutation outcome
    """

    data_loader = train_loader 
    loss_method = 'none'
    use_sep_criterion = False
    print_freq = 100
    
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
        
        if criterion_da is not None:
            dlabel = data[3].to(device, dtype=torch.float32)

        # # Calculate and set new learning rate
        adjust_learning_rate(optimizer0, epoch + data_it/len(data_loader), conf)

        # Compute loss
        if criterion_da is not None:
            sub_preds_list, slide_preds_list, attn_list, d_pred_list, bag_feat_list = model(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
        else:
            sub_preds_list, slide_preds_list, attn_list = model(image_patches)
            
        #Compute loss for each task, then sum
        loss = 0
        loss_d = 0
        for k in range(conf.n_task):
            sub_preds = sub_preds_list[k]
            slide_preds = slide_preds_list[k]
            attn = attn_list[k]
            labels = label_lists[:,k].to(device, dtype = torch.float32).to(device)
            
            #Compute domain loss
            if criterion_da is not None:
                d_pred = d_pred_list[k]
                dloss = criterion_da(d_pred, dlabel.unsqueeze(1))
                loss_d += grad_reverse(dloss)
                
            #Ohter loss
            if use_sep_criterion == False:      
                if conf.n_token > 1:
                    loss0 = criterion(sub_preds, labels.repeat_interleave(conf.n_token).unsqueeze(1))
                else:
                    loss0 = torch.tensor(0.)
                loss1 = criterion(slide_preds, labels.unsqueeze(1))
            else:
                if conf.n_token > 1:
                    loss0 = criterion[k](sub_preds, labels.repeat_interleave(conf.n_token).unsqueeze(1))
                else:
                    loss0 = torch.tensor(0.)
                loss1 = criterion[k](slide_preds, labels.unsqueeze(1))

            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            attn = torch.softmax(attn, dim=-1)
            
            for i in range(conf.n_token):
                for j in range(i + 1, conf.n_token):
                    diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                                conf.n_token * (conf.n_token - 1) / 2)


    
            if loss_method == 'ATTLOSS': #ATTLOSS
                #ATT loss
                #Take the AVG of each branch attention
                avg_attn = attn.mean(dim = 1) #Across branches
                att_loss = F.mse_loss(avg_attn, tf) #different in avg att to the tumor fraction
                # #Sum of att loss for each branch
                # att_loss = torch.tensor(0).to(device, dtype=torch.float)
                # for i in range(conf.n_token):
                #     att_loss += F.mse_loss(attn[:,i,:], tf)
                loss += diff_loss + loss0 + loss1 + att_loss
            else:
                loss += diff_loss + loss0 + loss1 
                

           
        optimizer0.zero_grad()
        # Backpropagate error and update parameters
        if criterion_da is not None:
            loss.backward(retain_graph=True)
            loss_d.backward()
        else:
            loss.backward()
            
        optimizer0.step()


        metric_logger.update(lr=optimizer0.param_groups[0]['lr'])
        metric_logger.update(sub_loss=loss0.item())
        metric_logger.update(diff_loss=diff_loss.item())
        metric_logger.update(slide_loss=loss1.item())
        if loss_method == 'ATTLOSS': 
            metric_logger.update(att_loss=att_loss.item())
        metric_logger.update(total_loss=loss.item())
        if criterion_da is not None:
            metric_logger.update(domain_loss=loss_d.item())

        if conf.wandb_mode != 'disabled':
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            wandb.log({'sub_loss': loss0}, commit=False)
            wandb.log({'diff_loss': diff_loss}, commit=False)
            wandb.log({'slide_loss': loss1})
            
            
            
            


print_every = print_freq  # assuming you have this defined
accum_steps = 32
batch_train = True

for data_it, data in enumerate(data_loader):
    
    image_patches = data[0].to(device, dtype=torch.float32) #[1, N_Patches, N_FEATURE]
    label_lists = data[1][0] #[1, N_FEATURE]
    tf = data[2].to(device, dtype=torch.float32)            #(1, N_Patches]

    if criterion_da is not None:
        dlabel = data[3].to(device, dtype=torch.float32)

    adjust_learning_rate(optimizer0, epoch + data_it / len(data_loader), conf)

    if criterion_da is not None:
        sub_preds_list, slide_preds_list, attn_list, d_pred_list, bag_feat_list = model(image_patches)
    else:
        sub_preds_list, slide_preds_list, attn_list, bag_feat_list = model(image_patches)

    loss = 0
    loss_d = 0

    for k in range(conf.n_task):
        
        #For each task:
        
        #predict from N_token branches
        sub_preds = sub_preds_list[k] #[N_Token, 1]
        
        #Prediction from slide
        slide_preds = slide_preds_list[k] #[1, 1]
        
        #attention from branches
        attn = attn_list[k]  #torch.Size([1, N_tokens, N_instances])
        
        #target labels
        labels = label_lists[:, k].to(device, dtype=torch.float32) #[1]

        if criterion_da is not None:
            d_pred = d_pred_list[k]
            dloss = criterion_da(d_pred, dlabel.unsqueeze(1))
            loss_d += grad_reverse(dloss)

        if not use_sep_criterion:
            loss0 = criterion(sub_preds, labels.repeat_interleave(conf.n_token).unsqueeze(1)) if conf.n_token > 1 else torch.tensor(0.).to(device)
            loss1 = criterion(slide_preds, labels.unsqueeze(1))
        else:
            loss0 = criterion[k](sub_preds, labels.repeat_interleave(conf.n_token).unsqueeze(1)) if conf.n_token > 1 else torch.tensor(0.).to(device)
            loss1 = criterion[k](slide_preds, labels.unsqueeze(1))

        diff_loss = torch.tensor(0.).to(device, dtype=torch.float)
        attn = torch.softmax(attn, dim=-1)
        for i in range(conf.n_token):
            for j in range(i + 1, conf.n_token):
                diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                    conf.n_token * (conf.n_token - 1) / 2)

        if loss_method == 'ATTLOSS':
            avg_attn = attn.mean(dim=1)
            att_loss = F.mse_loss(avg_attn, tf)
            loss += diff_loss + loss0 + loss1 + att_loss
        else:
            att_loss = None
            loss += diff_loss + loss0 + loss1
            

    
    if not batch_train:
        #Backpropagate error and update parameters 
        optimizer0.zero_grad()
        if criterion_da is not None:
            loss.backward(retain_graph=True)
            loss_d.backward()
        else:
            loss.backward()
        optimizer0.step()
        
    else:
        # Gradient accumulation: scale loss and backward (do not do total loss before of tensor freeed after each forward)
        loss = loss / accum_steps
        if criterion_da is not None:
            loss.backward(retain_graph=True)
            (loss_d / accum_steps).backward()
        else:
            loss.backward()

        # Step optimizer every accum_steps
        if (data_it + 1) % accum_steps == 0 or (data_it + 1) == len(data_loader):
            optimizer0.step()
            optimizer0.zero_grad()
            

    # === Manual Logging ===
    if data_it % print_every == 0 or data_it == len(data_loader) - 1:
        log_items = [
            f"[{data_it}/{len(data_loader)}]",
            f"lr: {optimizer0.param_groups[0]['lr']:.6f}",
            f"sub_loss: {loss0.item():.4f}",
            f"slide_loss: {loss1.item():.4f}",
            f"diff_loss: {diff_loss.item():.4f}",
            f"total_loss: {loss.item():.4f}"
        ]
        if att_loss is not None:
            log_items.append(f"att_loss: {att_loss.item():.4f}")
        if criterion_da is not None:
            log_items.append(f"domain_loss: {loss_d.item():.4f}")
        print(" | ".join(log_items))





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
        attnetion_list = []
        branch_pred_list = []
        bag_pred_list = []
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
            attnetion_list.append(A_out.unsqueeze(0)) # torch.Size([1, N_Tokens, N])
            
            # softmax over N 
            A = F.softmax(A, dim=1)  # torch.Size([N_Tokens, N])
            
            #features for each branch            
            afeat = torch.mm(A, x)   #torch.Size([N_Tokens, 128])
            
            outputs = []
            for j, head2 in enumerate(self.classifier_multitask[i]): #for each task,there are N_token classifers
                outputs.append(head2(afeat[j]))
            branch_pred_list.append(torch.stack(outputs, dim=0))
            
            #Bag Attention
            bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
            #Bag Feature
            bag_feat = torch.mm(bag_A, x)
            bag_feat_list.append(bag_feat)
            
            #Bag Prediction
            bag_pred_list.append(self.Slide_classifier_multitask[i](bag_feat))
            
        return branch_pred_list, bag_pred_list, attnetion_list, bag_feat_list


#Model Check
model = ACMIL_GA_MultiTask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
model.to(device)

x = data[0].to(device, dtype=torch.float32)

branch_pred_list, bag_pred_list, A_out_list, bag_feat_list  = model(x)


