#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:38:47 2025

@author: jliu6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=-1, gamma=0, reduction='mean'):
        r'''
        if alpha = -1, gamma = 0, then it is = CE loss
        '''
        super(FocalLoss, self).__init__()
        self.alpha = alpha #alpha is for the class = 1
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits, target):
        """FocalLoss_twologits
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

#this is equaliant to above
class FocalLoss_twologits(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        '''
        Focal Loss for softmax-based binary classification (2-class).

        Args:
            alpha (float or list of float): class balancing factor. If float, applies to class 1.
            gamma (float): focusing parameter.
            reduction (str): 'mean', 'sum', or 'none'.
        '''
        super(FocalLoss_twologits, self).__init__()
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
        
        

# logits = torch.tensor([[-3, 1.0]])
# log_probs = F.log_softmax(logits, dim=1)  # [B, 2]
# probs = torch.exp(log_probs)              # [B, 2]

# ce_loss = F.cross_entropy(logits, torch.tensor([1]), reduction="none")
# pt = torch.exp(-ce_loss)  # pt = softmax prob of the true class
