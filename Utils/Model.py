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
        #attention (Adapted from Attention-based Deep Multiple Instance Learning)
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


class Mutation_MIL_MT_sepAtt(nn.Module):
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
        
        # Create a list of copies of the attention layer
        self.attention_layers = nn.ModuleList([self.attention for _ in range(self.n_outs)])
                
        self.embed = nn.Sequential(
            nn.Linear(self.in_features, 1024), #linear layer
            self.act_func,
            nn.Linear(1024, 512), #linear layer
            self.act_func,
            nn.Linear(512, 256), #linear layer
            self.act_func,
            nn.Linear(256, 128), #linear layer
        )
        # Create a list of copies of the embedding_layer 
        self.embedding_layers = nn.ModuleList([self.embed for _ in range(self.n_outs)])

        #Outcome layers
        self.hidden_layers =  nn.ModuleList([nn.Linear(self.d_out, 1) for _ in range(self.n_outs)])        
        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        r'''
        x size: [1, N_TILE ,N_FEATURE]
        '''
        #attention (Adapted from Attention-based Deep Multiple Instance Learning)
        A_list = []
        out = []
        for i in range(self.n_outs):
            A = self.attention_layers[i](x) # NxK
            A = F.softmax(A, dim=1) # softmax over N
            M = x*A
            x_new = M.sum(dim=1) #1, 2048
            x_new = self.embedding_layers[i](x_new)  #Linear
            cur_out = self.hidden_layers[i](x_new)

            #Drop out
            if self.drop_out > 0:
                cur_out = self.dropout(cur_out)

            cur_out = torch.sigmoid(cur_out)
            
            A_list.append(A)
            out.append(cur_out)
        
        return out, A_list
        
import libpysal as ps
from esda.moran import Moran
class Mutation_MIL_MoransI(nn.Module):
    def __init__(self, in_features = 2048, act_func = 'tanh', drop_out = 0, n_outcomes = 7, dim_out = 5):
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
                
        self.embedding_layer = nn.Sequential(
            nn.Linear(4, 1024), #linear layer
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

    def forward(self, x, coor):
        r'''
        x size: [1, N_TILE ,N_FEATURE]
        '''
        #attention (Adapted from Attention-based Deep Multiple Instance Learning)
        A = self.attention(x) # NxK
        A = F.softmax(A, dim=1) # softmax over N

        #Assign attention scores to a matrix according to their XY cooridnates
        A_matrix = self.get_attention_matrix(coor, A)

        #Get moransI
        morans_i = self.get_moransi(A_matrix).unsqueeze(0)

        #Get percentage 
        _, _, perc = self.get_percentage_feature(A_matrix)
        normalized_prc = (perc - perc.min()) / (perc.max() - perc.min()) #norm

        #Get WSI reprensetation (combine morans I and normalized_prc)
        x = torch.cat([morans_i,normalized_prc]).float()

        # #Linear
        # x = self.embedding_layer(x) 

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
        
        return out, A_matrix

    def get_attention_matrix(self, coordinates, attention_scores):
        r'''
        Assign attention scores to a matrix according to their XY cooridnates
        '''
        #Reformat coordinates and attention 
        coordinates = coordinates.squeeze()
        attention_scores = attention_scores.view(-1) 
        
        # Define the dimensions of the matrix (assuming max coordinates for simplicity)
        max_x = coordinates[:,0].max() + 1
        max_y = coordinates[:,1].max() + 1
        
        # Initialize the matrix with zeros
        A_matrix = torch.zeros((max_x, max_y), device = attention_scores.device)
        
        # Assign values to the matrix
        for (x, y), value in zip(coordinates, attention_scores):
            A_matrix[x, y] = value
    
        return A_matrix

    def get_moransi(self, attention_matrix):
    
        #Reformat
        all_values = attention_matrix.flatten().cpu().detach().numpy()
        # Create a spatial weights matrix for the 2D grid
        w = ps.weights.lat2W(attention_matrix.shape[0], attention_matrix.shape[1])
    
        # Calculate Moran's I
        moran = Moran(all_values, w)
        moran_score = torch.tensor(moran.I, device = attention_matrix.device, dtype=torch.float64)
        p_val = moran.p_sim
        # print("Moran's I:", moran_score)
        # print("p-value:", p_val)
    
        return moran_score


    def get_percentage_feature(self, attention_matrix):
        
        intervals = torch.linspace(attention_matrix.flatten().min(), attention_matrix.flatten().max() + 1e-6, steps=4).to(attention_matrix.device)

        # Assign each element to an interval
        bucketized = torch.bucketize(attention_matrix.flatten(), intervals, right=True)
        
        # # Compute the percentage of elements in each group
        group_counts = torch.bincount(bucketized)
        total_elements = attention_matrix.flatten().size()[0]
        percentages = group_counts.float() / total_elements * 100
        percentages = percentages.double()
        
        # # Print the results
        # for i in range(1, len(intervals)):
        #     print(f"Group {i}: {percentages[i]:.2f}%")
    
        return intervals, bucketized, percentages
        

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



class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first = True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def min_max_norm(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    norm_tensor = (tensor - min_val) / (max_val - min_val)
    return norm_tensor


class Mutation_Multihead(nn.Module):
    def __init__(self, in_features = 2048, num_heads = 2, embed_dim = 128, dim_feedforward = 2048, act_func = 'tanh', drop_out = 0, n_outcomes = 7, dim_out = 128):
        super().__init__()
        self.in_features = in_features  
        self.embed_dim = embed_dim # 2048 feature dim
        self.num_heads = num_heads # N heads
        self.n_outs = n_outcomes # number of outcomes
        self.d_out = dim_out   # dim of output layers
        self.drop_out = drop_out
        self.dim_feedforward = dim_feedforward #linear layer in transofrmer

        if act_func == 'leakyrelu':
            self.act_func = nn.LeakyReLU()
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()


        #Transformer encoder layer
        self.attention =  nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first = True)
        self.linear1 = nn.Linear(self.embed_dim, self.dim_feedforward)
        self.dropout = nn.Dropout(self.drop_out)
        self.linear2 = nn.Linear(self.dim_feedforward, self.embed_dim)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout1 = nn.Dropout(self.drop_out)
        self.dropout2 = nn.Dropout(self.drop_out)

                
        # self.embedding_layer = nn.Sequential(
        #     nn.Linear(self.in_features, 1024), #linear layer
        #     self.act_func,
        #     nn.Linear(1024, 512), #linear layer
        #     self.act_func,
        #     nn.Linear(512, 256), #linear layer
        #     self.act_func,
        #     nn.Linear(256, 128), #linear layer
        # )

        #Attention for tiles
        self.attention_tiles = nn.Linear(128,1)

        #Outcome layers
        self.hidden_layers =  nn.ModuleList([nn.Linear(self.d_out, 1) for _ in range(self.n_outs)])        
        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        r'''
        x size: [1, N_TILE ,N_FEATURE]
        '''        
        #Linear
        #x = self.embedding_layer(x) 
        att_output, att_weights = self.attention(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout1(att_output) #add 
        x = self.norm1(x) #norm
        att_output2 = self.linear2(self.dropout(F.relu(self.linear1(x)))) #linear feed forward
        x = x + self.dropout2(att_output2)
        x = self.norm2(x)

        #Get tile attention
        #att_tiles = self.attention_tiles(x)
        #att_tiles = min_max_norm(self.attention_tiles(x))

        #Weight by attention tiles
        #x = att_tiles*x
        #Mean pooling
        x = torch.mean(x, dim = 1)
        
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
        
        return out,att_weights #out, att_tiles