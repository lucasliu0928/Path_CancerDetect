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
    def __init__(self, in_features = 2048, act_func = 'tanh', drop_out = 0, n_outcomes = 7, dim_out = 128):
        super().__init__()
        self.in_features = in_features  
        self.L = in_features # 2048 node fully connected layer
        self.D = 128 # 128 node attention layer
        self.K = 1
        self.n_outs = n_outcomes # number of outcomes
        self.d_out = dim_out   # dim of output layers
        self.drop_out = drop_out

        if act_func == 'leakyrelu':
            self.act_func = nn.LeakyReLU()
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()

        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K),
            nn.Tanh()
        )

        # self.one_encoder = nn.Sequential()
        # for i in range(len(dim_list)-1):
        #     self.one_encoder.append(nn.Linear(dim_list[i], dim_list[i+1]))
        #     self.one_encoder.append(nn.ReLU(True))
        #     if i != (len(dim_list) - 2):
        #         self.one_encoder.append(nn.Dropout())
                
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.in_features, 1024), #linear layer
            self.act_func,
            nn.Linear(1024, 512), #linear layer
            self.act_func,
            nn.Linear(512, 256), #linear layer
            self.act_func,
            nn.Linear(256, 128), #linear layer
        )

        #Outcome layers
        self.hidden_layers =  nn.ModuleList([nn.Linear(self.d_out, 1) for _ in range(self.n_outs)])        
        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        r'''
        x size: [1, N_TILE ,N_FEATURE]
        '''
        #attention
        A = self.attention(x) # NxK
        A = F.softmax(A, dim=1) # softmax over N
        M = x*A
        x = M.sum(dim=1) #1, 2048

        
        #Linear
        x = self.embedding_layer(x) 

        out = []
        for i in range(len(self.hidden_layers)):
            cur_out = self.hidden_layers[i](x)
            out.append(cur_out)

        #Drop out
        if self.drop_out > 0:
            for i in range(len(self.hidden_layers)):
                out[i] = self.dropout(out[i])
        
        # predict 
        for i in range(len(self.hidden_layers)):
            out[i] = torch.sigmoid(out[i])
        
        return out,A



class Mutation_MEANPOOLING_MT(nn.Module):
    def __init__(self, in_features = 2048, act_func = 'tanh'):
        super().__init__()
        self.in_features = in_features  

        if act_func == 'leakyrelu':
            self.act_func = nn.LeakyReLU()
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()
                
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.in_features, 1024), #linear layer
            self.act_func,
            nn.Linear(1024, 512), #linear layer
            self.act_func,
            nn.Linear(512, 256), #linear layer
            self.act_func,
            nn.Linear(256, 128), #linear layer
        )
        
        self.ln_out1 = nn.Linear(128, 1) #linear layer
        self.ln_out2 = nn.Linear(128, 1) #linear layer
        self.ln_out3 = nn.Linear(128, 1) #linear layer
        self.ln_out4 = nn.Linear(128, 1) #linear layer
        self.ln_out5 = nn.Linear(128, 1) #linear layer
        self.ln_out6 = nn.Linear(128, 1) #linear layer
        self.ln_out7 = nn.Linear(128, 1) #linear layer
        
        
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        r'''
        x size: [1, N_TILE ,N_FEATURE]
        '''
        x = torch.mean(x, dim=1) #N_Sample, 2048

        
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
        
        return y



class Mutation_MIL_ONE_MUT(nn.Module):
    def __init__(self, in_features = 2048, act_func = 'tanh', drop_out = False):
        super().__init__()
        self.in_features = in_features  
        self.L = 2048 # 2048 node fully connected layer
        self.D = 128 # 128 node attention layer
        self.K = 1
        self.drop_out = drop_out

        if act_func == 'leakyrelu':
            self.act_func = nn.LeakyReLU()
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()

        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

                
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.in_features, 1024), #linear layer
            self.act_func,
            nn.Linear(1024, 512), #linear layer
            self.act_func,
            nn.Linear(512, 256), #linear layer
            self.act_func,
            nn.Linear(256, 128), #linear layer
        )
        
        self.ln_out = nn.Linear(128, 1) #linear layer
        
        self.dropout = nn.Dropout(p=0.0)

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
        out = self.ln_out(x) 

        #Drop out
        if self.drop_out == True:
            out = self.dropout(out)  
        
        # predict 
        y = torch.sigmoid(out)

        return y,A




class Mutation_MEAMPOOLING_ONE_MUT(nn.Module):
    def __init__(self, in_features = 2048, act_func = 'tanh', drop_out = False):
        super().__init__()
        self.in_features = in_features  
        self.drop_out = drop_out

        if act_func == 'leakyrelu':
            self.act_func = nn.LeakyReLU()
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()

                
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.in_features, 1024), #linear layer
            self.act_func,
            nn.Linear(1024, 512), #linear layer
            self.act_func,
            nn.Linear(512, 256), #linear layer
            self.act_func,
            nn.Linear(256, 128), #linear layer
        )
        
        self.ln_out = nn.Linear(128, 1) #linear layer
        
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        r'''
        x size: [1, N_TILE ,N_FEATURE]
        '''
        x = torch.mean(x, dim=1) #N_Sample, 2048

        #Linear
        x = self.embedding_layer(x) 
        out = self.ln_out(x) 

        #Drop out
        if self.drop_out == True:
            out = self.dropout(out)  
        
        # predict 
        y = torch.sigmoid(out)

        return y



class Mutation_Multihead(nn.Module):
    def __init__(self, in_features = 2048, num_heads = 2, embed_dim = 128, act_func = 'tanh', drop_out = 0, n_outcomes = 7, dim_out = 128):
        super().__init__()
        self.in_features = in_features  
        self.embed_dim = embed_dim # 2048 feature dim
        self.num_heads = num_heads # N heads
        self.n_outs = n_outcomes # number of outcomes
        self.d_out = dim_out   # dim of output layers
        self.drop_out = drop_out

        if act_func == 'leakyrelu':
            self.act_func = nn.LeakyReLU()
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()

        
        self.attention =  nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first = True)

                
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.in_features, 1024), #linear layer
            self.act_func,
            nn.Linear(1024, 512), #linear layer
            self.act_func,
            nn.Linear(512, 256), #linear layer
            self.act_func,
            nn.Linear(256, 128), #linear layer
        )

        #Outcome layers
        self.hidden_layers =  nn.ModuleList([nn.Linear(self.d_out, 1) for _ in range(self.n_outs)])        
        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        r'''
        x size: [1, N_TILE ,N_FEATURE]
        '''        
        #Linear
        x = self.embedding_layer(x) 
        att_output, A = self.attention(x, x, x)



        out = []
        for i in range(len(self.hidden_layers)):
            cur_out = self.hidden_layers[i](att_output)
            out.append(cur_out)

        #Drop out
        if self.drop_out > 0:
            for i in range(len(self.hidden_layers)):
                out[i] = self.dropout(out[i])
        
        # predict 
        for i in range(len(self.hidden_layers)):
            out[i] = torch.sigmoid(out[i])
        
        return out,A