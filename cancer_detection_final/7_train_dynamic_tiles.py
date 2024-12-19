#!/usr/bin/env python
# coding: utf-8

#NOTE: use paimg9 env
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
from torch.utils.data import DataLoader
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
from train_utils import ModelReadyData_diffdim, convert_to_dict, prediction, BCE_Weighted_Reg, compute_loss_for_all_labels
from Model import Mutation_MIL_MT
import argparse
warnings.filterwarnings("ignore")


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-itf', '--Include_Tumor_Fraction', type=str2bool, default=False, help='Include TF as an extra feature')
parser.add_argument('-ic', '--Include_Cluster', type=str2bool, default=False, help='Include Cluster Perc as an extra feature')
parser.add_argument('-e', '--n_epoch', type=int, default=100, help='training epoch')
parser.add_argument('-of', '--out_folder', type=str, help='output folder')

args = parser.parse_args()

####################################
######      USERINPUT       ########
####################################
SELECTED_LABEL = ["AR","MMR (MSH2, MSH6, PMS2, MLH1, MSH3, MLH3, EPCAM)2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]
TRAIN_SAMPLE_SIZE = "ALLTUMORTILES"
TRAIN_OVERLAP = 100
TEST_OVERLAP = 0
SELECTED_FOLD = 0
TUMOR_FRAC_THRES = 0
feature_extraction_method = 'retccl'
INCLUDE_TF = args.Include_Tumor_Fraction
INCLUDE_CLUSTER = args.Include_Cluster
N_CLUSTERS = 4
out_folder = args.out_folder  #"pred_out1210"

########################
#model Para
LEARNING_RATE = 0.00001 
BATCH_SIZE  = 1
ACCUM_SIZE = 16  # Number of steps to accumulate gradients
EPOCHS = int(args.n_epoch)
DROPOUT = 0
DIM_OUT = 128
LOSS_FUNC_NAME = "BCE_Weighted_Reg" #"BCE_Weighted_Reg"
REG_COEEF = 0.0000001
REG_TYPE = 'L1'
OPTMIZER = "ADAM"
ATT_REG_FLAG = True
SELECTED_MUTATION = "MT"

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


if SELECTED_MUTATION == "MT":
    N_LABELS = len(SELECTED_LABEL)
    LOSS_WEIGHTS_LIST = [[1, 10], [1, 50], [1, 50], [1, 100], [1, 100], [1, 10], [1, 20]]  #NEG, POS #HR changed from 100 to 50
else:
    N_LABELS = 1
    LOSS_WEIGHTS_LIST = [[1, 10], [1, 10], [1, 50], [1, 100], [1, 100], [1, 100], [1, 20]]  #NEG, POS

##################
###### DIR  ######
##################
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
folder_name = feature_extraction_method + '/MAXSS'+ str(TRAIN_SAMPLE_SIZE)  + '_TrainOL' + str(TRAIN_OVERLAP) +  '_TestOL' + str(TEST_OVERLAP) + '_TFT' + str(TUMOR_FRAC_THRES) + "/split_fold" + str(SELECTED_FOLD) + "/" 
wsi_path = proj_dir + '/data/OPX/'
in_data_path = proj_dir + 'intermediate_data/model_ready_data/feature_' + folder_name + "model_input/"
model_data_path =  in_data_path + feature_type + "/"
    
################################################
#Create output-dir
################################################
outdir0 =  proj_dir + "intermediate_data/" + out_folder + "/" + folder_name + "/DL_" + feature_type + "/" + SELECTED_MUTATION + "/"
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


################################################
#     Model ready data 
################################################
train_data = torch.load(model_data_path + 'train_data.pth')
test_data = torch.load(model_data_path + 'test_data.pth')
val_data = torch.load(model_data_path + 'val_data.pth')

train_ids = torch.load(model_data_path + 'train_ids.pth')
test_ids = torch.load(model_data_path + 'test_ids.pth')
test_info  = torch.load(model_data_path + 'test_info.pth')

# print(feature_type)
# print("TF",INCLUDE_TF)
# print("CLUSTER",INCLUDE_CLUSTER)
print(train_data.x[0].shape)
####################################################
#            Train 
####################################################
set_seed(0)

#Dataloader for training
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)


#Construct model
model = Mutation_MIL_MT(in_features = N_FEATURE, 
                        act_func = 'tanh', 
                        drop_out = DROPOUT,
                        n_outcomes = N_LABELS,
                        dim_out = DIM_OUT)

model.to(device)

#Optimizer
if OPTMIZER == "ADAM":
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif OPTMIZER == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

#Loss
if LOSS_FUNC_NAME == "BCE_Weighted_Reg":
    loss_func = BCE_Weighted_Reg(REG_COEEF, REG_TYPE, model, reduction = 'mean', att_reg_flag = ATT_REG_FLAG)
elif LOSS_FUNC_NAME == "BCELoss":
    loss_func = torch.nn.BCELoss()
    

#Model para
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")
#print(model)


#OUTPUT MODEL hyper-para
hyper_df = pd.DataFrame({"Target_Mutation": SELECTED_MUTATION,
                         "TRAIN_OVERLAP": TRAIN_OVERLAP,
                         "TEST_OVERLAP": TEST_OVERLAP,
                         "TRAIN_SAMPLE_SIZE": TRAIN_SAMPLE_SIZE,
                         "TUMOR_FRAC_THRES": TUMOR_FRAC_THRES,
                         "N_FEATURE": N_FEATURE,
                         "N_LABELS": N_LABELS,
                         "BATCH_SIZE": BATCH_SIZE,
                         "ACCUM_SIZE": ACCUM_SIZE,
                         "N_EPOCH": EPOCHS,
                         "OPTMIZER": OPTMIZER,
                         "LEARNING_RATE": LEARNING_RATE,
                         "DROPOUT": DROPOUT,
                         "DIM_OUT": DIM_OUT,
                         "REG_TYPE": REG_TYPE,
                         "REG_COEEF": REG_COEEF,
                         "LOSS_FUNC_NAME": LOSS_FUNC_NAME,
                         "LOSS_WEIGHTS_LIST": str(LOSS_WEIGHTS_LIST),
                         "ATT_REG_FLAG": ATT_REG_FLAG,
                         "NUM_MODEL_PARA": total_params}, index = [0])
hyper_df.to_csv(outdir2 + "hyperpara_df.csv")


log_message("Start Training", outdir3 + "training_log.txt")
####################################################################################
#Training
####################################################################################
train_loss = []
valid_loss = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    ct = 0
    optmizer_loss = 0
    for x,y,tf in train_loader:
        ct += 1
        #optimizer.zero_grad() #zero the grad
        yhat_list, train_att_list = model(x.to(device)) #Forward

        #compute loss
        loss = compute_loss_for_all_labels(yhat_list, y, LOSS_WEIGHTS_LIST, LOSS_FUNC_NAME, loss_func, device, tf , train_att_list, SELECTED_MUTATION, SELECTED_LABEL)

        running_loss += loss.detach().item() #acuumalated batch loss
        optmizer_loss += loss #accumalted loss for optimizer
       
        #Optimize
        if ct % ACCUM_SIZE == 0:
            optmizer_loss = optmizer_loss/ACCUM_SIZE
            optmizer_loss.backward() 
            optimizer.step()  # Optimize
            optmizer_loss = 0
            optimizer.zero_grad() #gradient reset

    #Training loss 
    epoch_loss = running_loss/len(train_loader) #accumulated loss/total # batches (averaged loss over batches)
    train_loss.append(epoch_loss)

    #Validation
    model.eval()
    with torch.no_grad():
        val_running_loss = 0
        for x_val,y_val,tf_val in val_loader:
            val_yhat_list, val_att_list = model(x_val.to(device))
            val_loss = compute_loss_for_all_labels(val_yhat_list, y_val, LOSS_WEIGHTS_LIST, LOSS_FUNC_NAME, loss_func, device, tf_val, val_att_list, SELECTED_MUTATION, SELECTED_LABEL)
            val_running_loss += val_loss.detach().item() 
        val_epoch_loss = val_running_loss/len(val_loader) 
        valid_loss.append(val_epoch_loss)

    if epoch % 10 == 0:
        print("Epoch"+ str(epoch) + ":",
              "Train-LOSS:" + "{:.5f}".format(train_loss[epoch]) + ", " +
              "Valid-LOSS:" +  "{:.5f}".format(valid_loss[epoch]))
    
    #Save model parameters
    torch.save(model.state_dict(), outdir1 + "model" + str(epoch))


#Plot LOSS
plot_LOSS(train_loss,valid_loss, outdir1)
log_message("End Training", outdir3 + "training_log.txt")

#SAVE VALIDATION LOSS
valid_loss_df  = pd.DataFrame({"VALID_LOSS": valid_loss})
valid_loss_df.to_csv(outdir1 + "Valid_LOSS.csv")



####################################################################################
# Testing
####################################################################################
#Load model
valid_loss_df = pd.read_csv(outdir1 + "Valid_LOSS.csv")
min_index = valid_loss_df['VALID_LOSS'].idxmin()
print(min_index)
model = Mutation_MIL_MT(in_features = N_FEATURE, 
                        act_func = 'tanh', 
                        drop_out = DROPOUT,
                        n_outcomes = N_LABELS,
                        dim_out = DIM_OUT)
state_dict = torch.load(outdir1 + "model" + str(min_index))
model.load_state_dict(state_dict)
model.to(device)


#Loss function
loss_func = torch.nn.BCELoss()
THRES = 0.5

#predicts
test_pred_prob, test_true_label, test_att, test_loss = prediction(test_loader, model, N_LABELS, loss_func, device, SELECTED_MUTATION, SELECTED_LABEL, attention = True)
print("Test-Loss TOTAL: " + "{:.5f}".format(test_loss))


#Prediction df
pred_df_list = []
for i in range(0,N_LABELS):
    if N_LABELS > 1:
        cur_pred_df = pd.DataFrame({"SAMPLE_IDs":  test_ids, 
                                              "Y_True": [l[i] for l in test_true_label], 
                                              "Pred_Prob" :  [l[i] for l in test_pred_prob],
                                              #"Pred_Prob" :  test_pred_prob,
                                              "OUTCOME": SELECTED_LABEL[i]})
    else:
        cur_pred_df = pd.DataFrame({"SAMPLE_IDs":  test_ids, 
                                    "Y_True": [l[i] for l in test_true_label], 
                                    "Pred_Prob" :  test_pred_prob,
                                    "OUTCOME": SELECTED_MUTATION})
        
    pred_df_list.append(cur_pred_df)
pred_df = pd.concat(pred_df_list)

#Add Predict class
pred_df['Pred_Class'] = 0
pred_df.loc[pred_df['Pred_Prob'] > THRES,'Pred_Class'] = 1
pred_df.to_csv(outdir4 + "/pred_df.csv",index = False)


#Compute performance
if SELECTED_MUTATION == "MT":
    perf_df = compute_performance_each_label(SELECTED_LABEL, pred_df, "SAMPLE_LEVEL")
else:
    perf_df = compute_performance_each_label([SELECTED_MUTATION], pred_df, "SAMPLE_LEVEL")
perf_df.to_csv(outdir5 + "/perf.csv",index = True)

print(perf_df.iloc[:,[0,5,6,7,8,9]])
print("AVG AUC:", round(perf_df['AUC'].mean(),2))
print("AVG PRAUC:", round(perf_df['PR_AUC'].mean(),2))
