#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:27:44 2024

@author: jliu6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import os
import sys

#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from timm.utils import accuracy
import torchmetrics
import wandb
from torcheval.metrics import BinaryAUROC, BinaryAUPRC

#FROM ACMIL implementation architecture.network
class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=1):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x
    
class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x
    
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
    
    
    


# -------- Gradient Reversal Layer (for DANN) --------
class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # multiply by -lambda on the way back
        return grad_output.neg() * ctx.lambd, None

class GRL(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x):
        return _GradReverse.apply(x, self.lambd)


class ACMIL_GA_singletask_DA(nn.Module):
    """
    Single-task Attention-Gated MIL head + optional domain adaptation head.

    Domain head predicts slide domain: 0 = distant, 1 = local (2-class softmax).
    Uses Gradient Reversal to encourage domain-invariant bag features.

    Args (added):
        enable_domain (bool): turn domain branch on/off
        lambda_grl (float): GRL strength (0..1 typical)
        domain_droprate (float): dropout in domain head
    """

    def __init__(self, conf, D=128, droprate=0, n_token=1, n_masked_patch=0, mask_drop=0,
                 enable_domain=True, lambda_grl=1.0, domain_droprate=0.0):
        super(ACMIL_GA_singletask_DA, self).__init__()
        
        # Linear layer for dim reduction
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        
        # Attention layer
        self.attention = Attention_Gated(conf.D_inner, D, n_token)
        
        # Instance/branch classifier(s)
        self.classifier = nn.ModuleList()
        for _ in range(n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = n_masked_patch
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = mask_drop

        # ---------- Domain Adaptation head ----------
        self.enable_domain = enable_domain
        if self.enable_domain:
            self.grl = GRL(lambd=lambda_grl)
            # Reuse your 1-fc classifier head for 2-class domain prediction
            self.Domain_classifier = Classifier_1fc(conf.D_inner, 2, domain_droprate)

    def forward(self, x):  # x: [1, N_Tile, D_feat]
        x = x[0]
        x = self.dimreduction(x)  # [N, D_inner]
        
        # Attention
        A = self.attention(x)  # [K, N]
        
        # Masked patch (optional, train only)
        if self.n_masked_patch > 0 and self.training:
            A = self._apply_attention_mask(A)
        A_out = A

        # Softmax over N
        A = F.softmax(A, dim=1)  # [K, N]

        # Features per token/head
        afeat = torch.mm(A, x)   # [K, D_inner]

        # Branch (instance) predictions
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))
        branch_pred = torch.stack(outputs, dim=0)  # [K, n_class] (likely [K,1] for your setup)

        # Bag attention & bag feature
        bag_A   = F.softmax(A_out, dim=1).mean(0, keepdim=True)  # [1, N]
        bag_feat = torch.mm(bag_A, x)                            # [1, D_inner]

        # Slide label prediction (task head)
        bag_pred = self.Slide_classifier(bag_feat)               # [1, n_class]

        # ----- Domain prediction (optional) -----
        domain_pred = None
        if self.enable_domain:
            # Reverse gradients so shared feature becomes domain-invariant
            bag_feat_rev = self.grl(bag_feat)                    # [1, D_inner]
            domain_pred = self.Domain_classifier(bag_feat_rev)   # [1, 2] logits

        # Return domain_pred even if None to keep signature stable
        return branch_pred, bag_pred, A_out.unsqueeze(0), bag_A, bag_feat, domain_pred
        # (branch_pred, slide_pred, branch_attention_raw, bag_attention_softmaxed, bag_feature, domain_pred)
    
    def forward_feature(self, x, use_attention_mask=False):  ## x: [1, N, D_feat]
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  # [K, N]

        if self.n_masked_patch > 0 and use_attention_mask:
            A = self._apply_attention_mask(A)

        A_out = A
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        return bag_feat    
    
    def _apply_attention_mask(self, A):
        # Drop a random fraction of top-k attended patches per token
        k, n = A.shape
        n_masked_patch = min(self.n_masked_patch, n)
        _, indices = torch.topk(A, n_masked_patch, dim=-1)
        rand_selected = torch.argsort(torch.rand(*indices.shape, device=A.device), dim=-1)[:, :int(n_masked_patch * self.mask_drop)]
        masked_indices = indices[torch.arange(indices.shape[0], device=A.device).unsqueeze(-1), rand_selected]
        random_mask = torch.ones(k, n, device=A.device)
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
        # if print_every > 0 :
        #     if data_it % print_every == 0 or data_it == len(data_loader) - 1:
        #         log_items = [
        #             f"EPOCH: {epoch}",
        #             f"[{data_it}/{len(data_loader)}]",
        #             f"lr: {optimizer0.param_groups[0]['lr']:.6f}",
        #             f"branch_loss: {loss0.item():.4f}",
        #             f"slide_loss: {loss1.item():.4f}",
        #             f"diff_loss: {diff_loss.item():.4f}",
        #             f"total_loss: {avg_loss.item():.4f}"
        #         ]
        #         print(" | ".join(log_items))


import torch

def train_one_epoch_singletask2(model, criterion, data_loader, optimizer0, device, epoch, conf, 
                               print_every=100,
                               loss_method='none',
                               accum_steps=16,
                               use_amp=True,
                               max_norm=5.0):

    # Set the network to training mode
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    optimizer0.zero_grad(set_to_none=True)

    for data_it, data in enumerate(data_loader):

        # Get data
        image_patches = data[0].to(device, dtype=torch.float32, non_blocking=True)   # [1, N_Patches, N_FEATURE]
        labels = data[1][0].to(device)                                               # scalar or [1]
        tf = data[2].to(device, dtype=torch.float32, non_blocking=True)              # (1, N_Patches)

        # Forward pass with autocast
        with torch.cuda.amp.autocast(enabled=use_amp):
            branch_preds, slide_preds, branch_att_raw, bag_att, bag_feat = model(image_patches)

            # Compute loss
            loss, diff_loss, loss0, loss1 = compute_loss_singletask(
                branch_preds, slide_preds, labels, branch_att_raw, criterion, device, conf
            )

            # Normalize loss for gradient accumulation
            loss = loss / accum_steps

        # Backpropagate (scaled if AMP is on)
        scaler.scale(loss).backward()

        # Update running average of loss (unscaled)
        total_loss += loss.item() * accum_steps
        avg_loss = total_loss / (data_it + 1)

        # Optimizer step every accum_steps
        if (data_it + 1) % accum_steps == 0 or (data_it + 1) == len(data_loader):

            # Optional: gradient clipping (unscale first for AMP)
            if max_norm is not None:
                scaler.unscale_(optimizer0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer0)
            scaler.update()
            optimizer0.zero_grad(set_to_none=True)

        # Print progress
        if (data_it + 1) % print_every == 0:
            print(f"Epoch [{epoch}], Iter [{data_it+1}/{len(data_loader)}], "
                  f"Avg Loss: {avg_loss:.4f}")


def train_one_epoch_singletask2_DA(model, criterion, data_loader, optimizer0, device, epoch, conf, 
                               print_every=100,
                               loss_method='none',
                               accum_steps=16,
                               use_amp=True,
                               max_norm=5.0,
                               lambda_domain=0.5):
    """
    Adds domain loss:
      - expects domain label at data[4]  (0 = distant, 1 = local)
      - expects model to return domain_pred as 6th tensor (logits [1, 2]); if not present, skips domain loss
    """
    # Set the network to training mode
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_domain_loss = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce_domain = nn.CrossEntropyLoss()

    optimizer0.zero_grad(set_to_none=True)

    for data_it, data in enumerate(data_loader):

        # Get data
        image_patches = data[0].to(device, dtype=torch.float32, non_blocking=True)   # [1, N_Patches, N_FEATURE]
        labels_task = data[1][0].to(device)                                          # scalar or [1]
        tf = data[2].to(device, dtype=torch.float32, non_blocking=True)              # (1, N_Patches)
        # Domain label (0 = distant, 1 = local); shape [1]
        domain_label = None
        if len(data) > 3 and data[3] is not None:
            domain_label = data[3][0].squeeze().unique().to(device).long()                     # [1]

        # Forward pass with autocast
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(image_patches)

            # Unpack with/without domain head
            if isinstance(outputs, (list, tuple)) and len(outputs) == 6:
                branch_preds, slide_preds, branch_att_raw, bag_att, bag_feat, domain_pred = outputs
            else:
                branch_preds, slide_preds, branch_att_raw, bag_att, bag_feat = outputs
                domain_pred = None

            # ----- Task loss (your existing compute) -----
            task_loss, diff_loss, loss0, loss1 = compute_loss_singletask(
                branch_preds, slide_preds, labels_task, branch_att_raw, criterion, device, conf
            )

            # ----- Domain loss -----
            if (domain_pred is not None) and (domain_label is not None):
                # domain_pred: [1, 2] logits; domain_label: [1] with {0,1}
                d_loss = ce_domain(domain_pred, domain_label)
            else:
                # tensor(0.) on correct device for clean autograd sum
                d_loss = torch.tensor(0.0, device=device)

            total_step_loss = task_loss + lambda_domain * d_loss

            # Normalize for gradient accumulation
            loss = total_step_loss / accum_steps

        # Backpropagate (scaled if AMP is on)
        scaler.scale(loss).backward()

        # Running (unscaled) stats for logging
        total_loss += total_step_loss.item()
        total_task_loss += task_loss.item()
        total_domain_loss += d_loss.item()
        avg_loss = total_loss / (data_it + 1)
        avg_task_loss = total_task_loss / (data_it + 1)
        avg_domain_loss = total_domain_loss / (data_it + 1)

        # Optimizer step every accum_steps
        if (data_it + 1) % accum_steps == 0 or (data_it + 1) == len(data_loader):
            # Optional: gradient clipping (unscale first for AMP)
            if max_norm is not None:
                scaler.unscale_(optimizer0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer0)
            scaler.update()
            optimizer0.zero_grad(set_to_none=True)

        # Print progress
        if (data_it + 1) % print_every == 0:
            print(
                f"Epoch [{epoch}], Iter [{data_it+1}/{len(data_loader)}], "
                f"Avg Total Loss: {avg_loss:.4f} | "
                f"Avg Task: {avg_task_loss:.4f} | "
                f"Avg Domain: {avg_domain_loss:.4f}"
            )

    return avg_loss


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


@torch.no_grad()
def evaluate_singletask2(net, criterion, data_loader, device, conf, thres, cohort_name,
                        lambda_domain=0.5):
    """
    Domain adaptation-aware evaluation.
    - Expects the model to optionally return a 6th output: domain_pred logits [1, 2]
    - Expects domain label at data[4] with values {0 (distant), 1 (local)}
    - total_loss that is printed/returned = task_loss + lambda_domain * domain_loss (averaged)
    """
    net.eval()

    y_pred_prob = []
    y_pred_logit = []
    y_true = []

    # Domain collections (optional)
    dom_probs = []   # probability of domain==1 (local)
    dom_true  = []   # domain labels

    total_loss = 0.0
    total_task_loss = 0.0
    total_domain_loss = 0.0

    ce_domain = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data_it, data in enumerate(data_loader):
            image_patches = data[0].to(device, dtype=torch.float32)
            labels_task   = data[1][0].to(device)
            tf            = data[2].to(device, dtype=torch.float32)

            # Optional domain label
            domain_label = None
            if len(data) > 3 and data[3] is not None:
                domain_label = data[3].squeeze().unique().to(device).long()  # [1]
                #print(domain_label)

            # ---- Run model ----
            outputs = net(image_patches)

            # Unpack, allowing old/no-domain models
            if isinstance(outputs, (list, tuple)) and len(outputs) == 6:
                branch_preds, slide_preds, branch_att_raw, bag_att, bag_feat, domain_pred = outputs
            else:
                branch_preds, slide_preds, branch_att_raw, bag_att, bag_feat = outputs
                domain_pred = None

            # ---- Task predictions/metrics ----
            pred_prob = torch.sigmoid(slide_preds)  # task is 1-logit binary
            y_pred_prob.append(pred_prob)
            y_pred_logit.append(slide_preds)
            y_true.append(labels_task)

            # ---- Task loss ----
            task_loss, diff_loss, loss0, loss1 = compute_loss_singletask(
                branch_preds, slide_preds, labels_task, branch_att_raw, criterion, device, conf
            )

            # ---- Domain loss & metrics (optional) ----
            if (domain_pred is not None) and (domain_label is not None):
                d_loss = ce_domain(domain_pred, domain_label)  # [1,2] vs [1]
                # collect metrics
                dom_p = F.softmax(domain_pred, dim=1)[:, 1]    # P(domain==1)
                dom_probs.append(dom_p)
                dom_true.append(domain_label)
            else:
                d_loss = torch.tensor(0.0, device=device)

            total_step_loss = task_loss + lambda_domain * d_loss

            # Accumulate (CPU floats)
            total_loss       += total_step_loss.item()
            total_task_loss  += task_loss.item()
            total_domain_loss+= d_loss.item()

        # ===== Concatenate all preds/labels (task) =====
        all_pred_prob  = torch.cat(y_pred_prob,  dim=0)
        all_pred_logit = torch.cat(y_pred_logit, dim=0)
        all_labels     = torch.cat(y_true,       dim=0)
        all_pred_class = (all_pred_prob > thres).int()

        # ===== Task metrics =====
        roauc_metric = BinaryAUROC().to(device)
        roauc_metric.update(all_pred_prob.squeeze(), all_labels.squeeze())
        roc_auc = roauc_metric.compute().item()

        prauc_metric = BinaryAUPRC().to(device)
        prauc_metric.update(all_pred_prob.squeeze(), all_labels.squeeze())
        pr_auc = prauc_metric.compute().item()

        # Averages
        n_batches = len(data_loader)
        avg_total_loss   = total_loss / max(1, n_batches)
        avg_task_loss    = total_task_loss / max(1, n_batches)
        avg_domain_loss  = total_domain_loss / max(1, n_batches)

        # ===== Domain metrics (optional, only if available) =====
        domain_metrics_str = "domain_auc: N/A | domain_acc: N/A"
        if len(dom_probs) > 0 and len(dom_true) > 0:
            all_dom_probs = torch.cat(dom_probs, dim=0)     # [num_slides]
            all_dom_true  = torch.cat(dom_true,  dim=0)     # [num_slides]

            dom_auc_metric = BinaryAUROC().to(device)
            dom_auc_metric.update(all_dom_probs, all_dom_true)
            domain_auc = dom_auc_metric.compute().item()

            # Accuracy at 0.5 for readability
            dom_pred_class = (all_dom_probs > 0.5).long()
            domain_acc = (dom_pred_class == all_dom_true).float().mean().item()

            domain_metrics_str = f"domain_auc: {domain_auc:.4f} | domain_acc: {domain_acc:.4f}"

    # === Manual Logging ===
    log_items = [
        f"cohort_name: {cohort_name}",
        f"total_loss: {avg_total_loss:.4f}",   # task + lambda_domain * domain
        f"task_loss: {avg_task_loss:.4f}",
        f"domain_loss: {avg_domain_loss:.4f}",
        f"roauc: {roc_auc:.4f}",
        f"pr_auc: {pr_auc:.4f}"
        #domain_metrics_str
    ]
    print(" | ".join(log_items))

    # Keep original return signature for compatibility
    return avg_total_loss, roc_auc, pr_auc


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




