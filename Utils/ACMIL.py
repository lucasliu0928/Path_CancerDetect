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
from torcheval.metrics import BinaryAUROC, BinaryAUPRC


class decouple_classifier(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(decouple_classifier, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x
    
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


class ACMIL_GA_singletask(nn.Module):
    """
    Single-task Attention-Gated MIL head.
    Modified from original implementation: https://github.com/dazhangyu123/ACMIL/blob/main/architecture/transformer.py#L291
    
    Args:
        conf: An object with attributes:
              - D_feat (int): input feature dim
              - D_inner (int): hidden/inner feature dim
              - n_class (int): number of classes
        D (int): attention hidden dimension.
        droprate (float): dropout prob for classifiers.
        n_token (int): number of attention tokens/heads (K).
        n_masked_patch (int): how many top-attended patches to consider for masking.
        mask_drop (float): fraction (0..1) of those top patches to drop at random.
        
    """

    def __init__(self, conf, D=128, droprate=0, n_token=1, n_masked_patch=0, mask_drop=0):
        super(ACMIL_GA_singletask, self).__init__()
        
        #Linear layer for dim reduction
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        
        #Attention layer
        self.attention = Attention_Gated(conf.D_inner, D, n_token)
        
        #Classifer
        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = n_masked_patch
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = mask_drop


    def forward(self, x): #x: N_Ttile x N_Feature
        x = x[0]
        x = self.dimreduction(x)  #torch.Size([N, 128])
        
        #Attention
        A = self.attention(x)  ## K x N
        
        #Masked patch
        if self.n_masked_patch > 0 and self.training:
            A = self._apply_attention_mask(A)
            
        A_out = A
        
        # softmax over N 
        A = F.softmax(A, dim=1)  # torch.Size([N_Tokens, N])
        
        #features for each branch            
        afeat = torch.mm(A, x)   #torch.Size([N_Tokens, 128])
        
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))
        branch_pred = torch.stack(outputs, dim=0)
        
        #Bag Attention
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        #Bag Feature
        bag_feat = torch.mm(bag_A, x) #torch.Size([1, 128])
        
        #Bag Prediction
        bag_pred =  self.Slide_classifier(bag_feat)
            
        return branch_pred, bag_pred, A_out.unsqueeze(0), bag_A, bag_feat #Branch prediction, bag prediction, brach_attention_raw, bag_attention_softmaxed, bag_feature
    
    def forward_feature(self, x, use_attention_mask=False): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N


        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            A = self._apply_attention_mask(A)

        A_out = A
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        
        return bag_feat    
    
    def _apply_attention_mask(self, A):
        # Get the indices of the top-k largest values
        k, n = A.shape
        n_masked_patch = min(self.n_masked_patch, n)
        _, indices = torch.topk(A, n_masked_patch, dim=-1) # Indices of top-k (per row/token)
        rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)] # Randomly choose a fraction of those to actually drop
        masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
        random_mask = torch.ones(k, n).to(A.device)
        random_mask.scatter_(-1, masked_indices, 0)
        A = A.masked_fill(random_mask == 0, -1e9)
        
        return A
    
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
        x = self.dimreduction(x)  #torch.Size([N, 128])

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
            bag_feat = torch.mm(bag_A, x) #torch.Size([1, 128])
            bag_feat_list.append(bag_feat)
            
            #Bag Prediction
            bag_pred_list.append(self.Slide_classifier_multitask[i](bag_feat))
            
        return branch_pred_list, bag_pred_list, attnetion_list, bag_feat_list



class ACMIL_GA_MultiTask_DA(nn.Module):
    def __init__(self, conf, D=128, droprate=0, n_token=1, n_masked_patch=0, mask_drop=0, n_task = 7):
        super(ACMIL_GA_MultiTask_DA, self).__init__()
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
        
        #Domain predction layer
        self.domain_layer  =  nn.Linear(conf.D_inner, 1)


    def forward(self, x): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)

        #Each task has its own attention
        A_out_list = []
        outputs_list = []
        bag_feat_list = []
        bag_pred_list = []
        domain_pred_list = []
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
            bag_feat_list.append(bag_feat)
            bag_pred_list.append(self.Slide_classifier_multitask[i](bag_feat))
            
            #Predict domain
            d_y = self.domain_layer(bag_feat)
            domain_pred_list.append(d_y)
            

        return outputs_list, bag_pred_list, A_out_list, domain_pred_list, bag_feat_list

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
        sub_preds_list, slide_preds_list, attn_list, bag_feat_list = net(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
        
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


def get_emebddings(net, data_loader, device, criterion_da = None):    

    bag_feature_list = []
    # Set the network to evaluation mode
    net.eval()
    with torch.no_grad():
        for data in data_loader:
            image_patches = data[0].to(device, dtype=torch.float32)
            
            if criterion_da is not None:
                sub_preds_list, slide_preds_list, attn_list, d_pred_list, bag_feat_list = net(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
            else:
                sub_preds_list, slide_preds_list, attn_list,bag_feat_list = net(image_patches)
            
            bag_feature_list.append(bag_feat_list)

    return bag_feature_list

                   
def train_one_epoch_singletask(model, criterion, data_loader, optimizer0, device, epoch, conf, 
                                        print_every = 100,
                                        loss_method = 'none'):

    # Set the network to training mode
    model.train()
    total_loss = 0
    for data_it, data in enumerate(data_loader):
        
        #Get data
        image_patches = data[0].to(device, dtype=torch.float32) #[1, N_Patches, N_FEATURE]
        labels = data[1][0].to(device)
        tf = data[2].to(device, dtype=torch.float32)            #(1, N_Patches]
    
        #Adjust LR
        adjust_learning_rate(optimizer0, epoch + data_it / len(data_loader), conf)
        
        #Run model
        branch_preds, slide_preds, branch_att_raw, bag_att , bag_feat = model(image_patches)
        
        
        #Compute loss
        loss, diff_loss, loss0,  loss1 = compute_loss_singletask(branch_preds, slide_preds, labels, branch_att_raw, criterion, device, conf)
        total_loss += loss
        avg_loss = total_loss/(data_it+1)     
        
        #Backpropagate error and update parameters 
        optimizer0.zero_grad()
        loss.backward()
        optimizer0.step()

    
        # === Manual Logging ===
        if print_every > 0 :
            if data_it % print_every == 0 or data_it == len(data_loader) - 1:
                log_items = [
                    f"EPOCH: {epoch}",
                    f"[{data_it}/{len(data_loader)}]",
                    f"lr: {optimizer0.param_groups[0]['lr']:.6f}",
                    f"branch_loss: {loss0.item():.4f}",
                    f"slide_loss: {loss1.item():.4f}",
                    f"diff_loss: {diff_loss.item():.4f}",
                    f"total_loss: {avg_loss.item():.4f}"
                ]
                print(" | ".join(log_items))


def compute_loss_singletask(branch_preds, slide_preds, labels, branch_att_raw, criterion, device, conf):
    # Compute loss
    if conf.n_token > 1:
        loss0 = criterion(branch_preds, labels.repeat_interleave(conf.n_token).unsqueeze(1))
    else:
        loss0 = torch.tensor(0.)
    loss1 = criterion(slide_preds, labels)
    
    
    diff_loss = torch.tensor(0.).to(device, dtype=torch.float32)
    attn = torch.softmax(branch_att_raw, dim=-1)
    for i in range(conf.n_token):
        for j in range(i + 1, conf.n_token):
            diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                        conf.n_token * (conf.n_token - 1) / 2)
    
    loss = diff_loss + loss0 + loss1
    
    return loss, diff_loss, loss0,  loss1
        




def train_one_epoch_multitask(model, criterion, data_loader, optimizer0, device, epoch, conf, loss_method = 'none', use_sep_criterion = False, criterion_da = None):
    """
    Trains the given network for one epoch according to given criterions (loss functions)

    use_sep_criterion: use differnt focal paratermeters for each mutation outcome
    """

    # loss_method = 'none' 
    # use_sep_criterion = False
    # epoch = 0
    # data_loader = train_loader
    # criterion_da = criterion_da

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
            sub_preds_list, slide_preds_list, attn_list, bag_feat_list = model(image_patches)
            
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


def train_one_epoch_multitask_minibatch(model, criterion, data_loader, optimizer0, device, epoch, conf, 
                                        batch_train = True,
                                        accum_steps = 32,
                                        print_every = 100,
                                        loss_method = 'none', 
                                        use_sep_criterion = False, 
                                        criterion_da = None):
    """
    Trains the given network for one epoch according to given criterions (loss functions)

    use_sep_criterion: use differnt focal paratermeters for each mutation outcome
    """

    # Set the network to training mode
    model.train()
    
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
                f"EPOCH: {epoch}",
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



def train_one_epoch_multitask_minibatch_randomSample(model, criterion, data_loader, optimizer0, device, epoch, conf, 
                                        batch_train = True,
                                        accum_steps = 32,
                                        print_every = 100,
                                        loss_method = 'none', 
                                        use_sep_criterion = False, 
                                        criterion_da = None):
    """
    Trains the given network for one epoch according to given criterions (loss functions)

    use_sep_criterion: use differnt focal paratermeters for each mutation outcome
    """
    
    rs_rate_stnorm = 0.8
    rs_rate_nostnorm = 0.2

    # Set the network to training mode
    model.train()
    
    for data_it, data in enumerate(data_loader):
        torch.manual_seed(epoch)
        image_patches1 = data[0].to(device, dtype=torch.float32) #[1, N_Patches, N_FEATURE]
        indices = torch.randperm(image_patches1.size(1))[:round(image_patches1.size(1)*rs_rate_stnorm)] #sample
        image_patches1 = image_patches1[:, indices, :]

        image_patches2 = data[4].to(device, dtype=torch.float32)
        indices = torch.randperm(image_patches2.size(1))[:round(image_patches2.size(1)*rs_rate_nostnorm)] #sample
        image_patches2 = image_patches2[:, indices, :]
        
        image_patches = torch.concat([image_patches1,image_patches2], dim = 1)
        
        
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
                f"EPOCH: {epoch}",
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
            

@torch.no_grad()
def get_slide_feature(net, data_loader, conf, device):
    net.eval()
    
    features_pertask = {f"task{i}": [] for i in range(conf.n_task)}

    for data in data_loader:
        
        image_patches = data[0].to(device, dtype=torch.float32)      
        
        _, _, _, bag_feat_list = net(image_patches)

        # Append each feature to its corresponding list
        for i, feat in enumerate(bag_feat_list):
            features_pertask[f"task{i}"].append(feat)
    
    #Per task, each key in dict store all samples's feature [n_sample, n_features]
    for i in range(conf.n_task):
        features_pertask[f"task{i}"] = torch.concat(features_pertask[f"task{i}"])
        
    return features_pertask


@torch.no_grad()
def get_slide_feature_singletask(net, data_loader, device):
    net.eval()
    
    features_list = [] 
    label_list = []
    for data in data_loader:
        
        image_patches = data[0].to(device, dtype=torch.float32)   
        label = data[1][0].to(device)   
        label_list.append(label)        
        bag_feat = net.forward_feature(image_patches)
        features_list.append(bag_feat)

    features = torch.concat(features_list, dim = 0)
    labels = torch.concat(label_list, dim = 0)
        
    return features, labels


# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate_singletask(net, criterion, data_loader, device, conf, thres, cohort_name ):
            
    # Set the network to evaluation mode
    net.eval()

    y_pred_prob = []
    y_pred_logit = []
    y_true = []
    total_loss = 0
    for data_it, data in enumerate(data_loader):
        image_patches = data[0].to(device, dtype=torch.float32)
        labels = data[1][0].to(device)
        tf = data[2].to(device, dtype=torch.float32)
    
    
        #Run model
        branch_preds, slide_preds, branch_att_raw, bag_att , bag_feat = net(image_patches)
        
        
        #Get predictions
        pred_prob = torch.sigmoid(slide_preds)
        y_pred_prob.append(pred_prob)
        y_pred_logit.append(slide_preds)
        y_true.append(labels)
        
        #Compute loss
        loss, diff_loss, loss0,  loss1 = compute_loss_singletask(branch_preds, slide_preds, labels, branch_att_raw, criterion, device, conf)
        total_loss += loss
        avg_loss = total_loss/(data_it+1)
        

    #Concatenate all pred and labels
    all_pred_prob  = torch.cat(y_pred_prob, dim=0)
    all_pred_logit  = torch.cat(y_pred_logit, dim=0)
    all_labels = torch.cat(y_true, dim=0)    
    all_pred_class = (all_pred_prob > thres).int()
    
    
    #Get performance
    roauc_metric = BinaryAUROC().to(device)
    roauc_metric.update(all_pred_prob.squeeze(), all_labels.squeeze())
    roc_auc = roauc_metric.compute().item()
    
    
    prauc_metric = BinaryAUPRC().to(device)
    prauc_metric.update(all_pred_prob.squeeze(), all_labels.squeeze())
    pr_auc = prauc_metric.compute().item()

    
    # === Manual Logging ===
    log_items = [
        f"cohort_name: {cohort_name}",
        f"total_loss: {avg_loss.item():.4f}",
        f"roauc: {roc_auc:.4f}",
        f"pr_auc: {pr_auc:.4f}",
    ]
    print(" | ".join(log_items))

    return total_loss, roc_auc , pr_auc


# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate_multitask(net, criterion, data_loader, device, conf, header, use_sep_criterion = False, criterion_da = None):

    # Set the network to evaluation mode
    net.eval()

    y_pred = []
    y_true = []

    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):
        image_patches = data[0].to(device, dtype=torch.float32)
        label_lists = data[1][0]
        tf = data[2].to(device, dtype=torch.float32)
        
        # if criterion_da is not None:
        #     dlabel = data[3].to(device, dtype=torch.float32)
        
        if criterion_da is not None:
            sub_preds_list, slide_preds_list, attn_list, d_pred_list, bag_feat_list = net(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
        else:
            sub_preds_list, slide_preds_list, attn_list, bag_feat_list = net(image_patches)
            
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

            if use_sep_criterion == False: 
                loss += criterion(slide_preds, labels.unsqueeze(1))
            else:
                loss += criterion[k](slide_preds, labels.unsqueeze(1))

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



# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate_multitask_randomSample(net, criterion, data_loader, device, conf, header, use_sep_criterion = False, criterion_da = None):
    
    rs_rate_stnorm = 1.0
    rs_rate_nostnorm = 1.0
    
    # Set the network to evaluation mode
    net.eval()

    y_pred = []
    y_true = []

    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):
 
        torch.manual_seed(0)
        image_patches1 = data[0].to(device, dtype=torch.float32) #[1, N_Patches, N_FEATURE]
        indices = torch.randperm(image_patches1.size(1))[:round(image_patches1.size(1)*rs_rate_stnorm)] #sample
        image_patches1 = image_patches1[:, indices, :]

        image_patches2 = data[3].to(device, dtype=torch.float32)
        indices = torch.randperm(image_patches2.size(1))[:round(image_patches2.size(1)*rs_rate_nostnorm)] #sample
        image_patches2 = image_patches2[:, indices, :]
        
        image_patches = torch.concat([image_patches1,image_patches2], dim = 1)
            
        label_lists = data[1][0]
        tf = data[2].to(device, dtype=torch.float32)
        
        # if criterion_da is not None:
        #     dlabel = data[3].to(device, dtype=torch.float32)
        
        if criterion_da is not None:
            sub_preds_list, slide_preds_list, attn_list, d_pred_list, bag_feat_list = net(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
        else:
            sub_preds_list, slide_preds_list, attn_list, bag_feat_list = net(image_patches)
            
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

            if use_sep_criterion == False: 
                loss += criterion(slide_preds, labels.unsqueeze(1))
            else:
                loss += criterion[k](slide_preds, labels.unsqueeze(1))

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




