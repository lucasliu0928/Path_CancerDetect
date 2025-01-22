#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from train_utils import ModelReadyData_diffdim, add_tile_xy, convert_to_dict, prediction, BCE_Weighted_Reg, BCE_Weighted_Reg_focal, compute_loss_for_all_labels
from Model import Mutation_MIL_MoransI
warnings.filterwarnings("ignore")
#get_ipython().run_line_magic('matplotlib', 'inline')


####################################
######      USERINPUT       ########
####################################
SELECTED_LABEL = ["AR","MMR (MSH2, MSH6, PMS2, MLH1, MSH3, MLH3, EPCAM)2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]
TRAIN_SAMPLE_SIZE = "ALLTUMORTILES"
TRAIN_OVERLAP = 100
TEST_OVERLAP = 0
SELECTED_FOLD = 0
TUMOR_FRAC_THRES = 0.9
feature_extraction_method = 'retccl'
INCLUDE_TF = False
INCLUDE_CLUSTER = False
N_CLUSTERS = 4
focal_gamma = 2


####
#model Para
LEARNING_RATE = 0.00001 
BATCH_SIZE  = 1
ACCUM_SIZE = 16  # Number of steps to accumulate gradients
EPOCHS = 500
DROPOUT = 0
DIM_OUT = 5

if INCLUDE_TF == False and INCLUDE_CLUSTER == False:
    N_FEATURE = 2048
elif INCLUDE_TF == True and INCLUDE_CLUSTER == False:
    N_FEATURE = 2049
elif INCLUDE_TF == False and INCLUDE_CLUSTER == True:
    N_FEATURE = 2049
elif INCLUDE_TF == True and INCLUDE_CLUSTER == True:
    N_FEATURE = 2050
            

LOSS_FUNC_NAME = "BCE_Weighted_Reg_focal" #"BCE_Weighted_Reg", "BCE_Weighted_Reg_focal"
REG_COEEF = 0.0000001
REG_TYPE = 'L1'
OPTMIZER = "ADAM"
ATT_REG_FLAG = False
SELECTED_MUTATION = "MT"

if SELECTED_MUTATION == "MT":
    N_LABELS = len(SELECTED_LABEL)
    LOSS_WEIGHTS_LIST = [[1, 50], [1, 100], [1, 50], [1, 100], [1, 100], [1, 50], [1, 50]]  #NEG, POS
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

if INCLUDE_TF == False and INCLUDE_CLUSTER == False:
    feature_type = "emb_only"
elif INCLUDE_TF == True and INCLUDE_CLUSTER == False:
    feature_type = "emb_and_tf"
elif INCLUDE_TF == False and INCLUDE_CLUSTER == True:
    feature_type = "emb_and_cluster" + str(N_CLUSTERS)
elif INCLUDE_TF == True and INCLUDE_CLUSTER == True:
    feature_type = "emb_and_tf_and_cluster" + str(N_CLUSTERS) 

model_data_path =  in_data_path + feature_type + "/"
    
################################################
#Create output-dir
################################################
outdir0 =  proj_dir + "intermediate_data/pred_out011325/" + folder_name + "/DL_" + feature_type + "/" + SELECTED_MUTATION + "/"
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
test_data_old = torch.load(model_data_path + 'test_data.pth')
test_data_add = torch.load(model_data_path + 'newMSI_test_data.pth')
val_data = torch.load(model_data_path + 'val_data.pth')

test_ids_old = torch.load(model_data_path + 'test_ids.pth')
test_ids_add = torch.load(model_data_path + 'newMSI_test_ids.pth')
test_info_old  = torch.load(model_data_path + 'test_info.pth')
test_info_add  = torch.load(model_data_path + 'newMSI_test_info.pth')

train_info = torch.load(model_data_path + 'train_info.pth')
val_info = torch.load(model_data_path + 'val_info.pth')

#Add train info to train_data
train_data = add_tile_xy(train_data, train_info)
val_data = add_tile_xy(val_data, val_info)



################################################
#Exclude OPX_085, Prostate cancer find in colorectal site, patterns are for CRC, not for prostate
################################################
exc_idx = test_ids_old.index('OPX_085')
inc_idx = [i for i in range(len(test_data_old)) if i not in [exc_idx]]

#Update old testset
test_data_old = Subset(test_data_old, inc_idx)
removed_id =   test_ids_old.pop(exc_idx)  
removed_info = test_info_old.pop(exc_idx)  

################################################
#Combine old and new test data
################################################
test_data  = ConcatDataset([test_data_old, test_data_add])
test_ids = test_ids_old +  test_ids_add
test_info = test_info_old +  test_info_add

test_data = add_tile_xy(test_data, test_info)




#count labels in test
test_label_counts = [dt[1] for dt in test_data]
test_label_counts = torch.concat(test_label_counts)
count_ones = (test_label_counts == 1).sum(dim=0)
print(count_ones)
perc_ones = count_ones/test_label_counts.shape[0] * 100
print(perc_ones)


####################################################
#            Train 
####################################################
set_seed(0)

#Dataloader for training
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)


#Construct model
model = Mutation_MIL_MoransI(in_features = N_FEATURE, 
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
elif LOSS_FUNC_NAME == "BCE_Weighted_Reg_focal":
    loss_func = BCE_Weighted_Reg_focal(REG_COEEF, REG_TYPE, model, gamma = focal_gamma, reduction = 'mean', att_reg_flag = ATT_REG_FLAG)
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
    for x,y,tf, coor in train_loader:
        ct += 1
        #optimizer.zero_grad() #zero the grad
        yhat_list, train_att_list = model(x.to(device), coor.to(device)) #Forward

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
        for x_val,y_val,tf_val,coor_val in val_loader:
            val_yhat_list, val_att_list = model(x_val.to(device),coor_val.to(device))
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