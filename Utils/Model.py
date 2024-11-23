#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:27:44 2024

@author: jliu6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mutation_MIL_MT(nn.Module):
    def __init__(self, in_features = 2048):
        super().__init__()
        self.in_features = in_features  
        self.L = 2048 # 512 node fully connected layer
        self.D = 128 # 128 node attention layer
        self.K = 1

        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        # self.one_encoder = nn.Sequential()
        # for i in range(len(dim_list)-1):
        #     self.one_encoder.append(nn.Linear(dim_list[i], dim_list[i+1]))
        #     self.one_encoder.append(nn.ReLU(True))
        #     if i != (len(dim_list) - 2):
        #         self.one_encoder.append(nn.Dropout())
                
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.in_features, 1024), #linear layer
            nn.Linear(1024, 512), #linear layer
            nn.Linear(512, 256), #linear layer
            nn.Linear(256, 128), #linear layer
        )
        
        self.ln_out1 = nn.Linear(128, 1) #linear layer
        self.ln_out2 = nn.Linear(128, 1) #linear layer
        self.ln_out3 = nn.Linear(128, 1) #linear layer
        self.ln_out4 = nn.Linear(128, 1) #linear layer
        self.ln_out5 = nn.Linear(128, 1) #linear layer
        self.ln_out6 = nn.Linear(128, 1) #linear layer
        self.ln_out7 = nn.Linear(128, 1) #linear layer
        
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        r'''
        x size: [1, N_TILE ,N_FEATURE]
        '''
        #attention
        A = self.attention(x) # NxK
        A = F.softmax(A, dim=1) # softmax over N
        M = x*A
        x = M.sum(dim=1) #N_Sample, 2048

        
        #Linear
        x = self.embedding_layer(x) 
        out1 = self.ln_out1(x) 
        out2 = self.ln_out2(x) 
        out3 = self.ln_out3(x) 
        out4 = self.ln_out4(x) 
        out5 = self.ln_out5(x) 
        out6 = self.ln_out6(x) 
        out7 = self.ln_out7(x) 

        #Drop out
        out1 = self.dropout(out1)  # Apply dropout
        out2 = self.dropout(out2)  # Apply dropout
        out3 = self.dropout(out3)  # Apply dropout
        out4 = self.dropout(out4)  # Apply dropout
        out5 = self.dropout(out5)  # Apply dropout
        out6 = self.dropout(out6)  # Apply dropout
        out7 = self.dropout(out7)  # Apply dropout
        
        # predict 
        y1 = torch.sigmoid(out1)
        y2 = torch.sigmoid(out2)
        y3 = torch.sigmoid(out3)      
        y4 = torch.sigmoid(out4)
        y5 = torch.sigmoid(out5)
        y6 = torch.sigmoid(out6)
        y7 = torch.sigmoid(out7)

        y = [y1, y2, y3, y4, y5, y6, y7]
        
        return y,A
