#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 20:54:31 2025

@author: jliu6
"""

#!/usr/bin/env python
# coding: utf-8

#NOTE: use python env acmil in ACMIL folder
import sys
import os
import numpy as np
import matplotlib
%matplotlib inline
matplotlib.use('Agg')
import pandas as pd
import warnings
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists, set_seed, str2bool
from Eval import boxplot_predprob_by_mutationclass, get_performance, plot_roc_curve
from Eval import bootstrap_ci_from_df, calibrate_probs_isotonic
from train_utils import FocalLoss, get_feature_idexes, get_train_test_val_data, get_external_validation_data, load_model_ready_data
from ACMIL import ACMIL_GA_MultiTask, predict_v2, train_one_epoch_multitask, evaluate_multitask
warnings.filterwarnings("ignore")
from train_utils import FocalLoss, predict_clustercnn
from Model import RESNET_CLUSTER
from train_utils import plot_cluster_image

#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
from utils.utils import save_model, Struct
import yaml
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse
from architecture.transformer import ACMIL_GA #ACMIL_GA
from architecture.transformer import ACMIL_MHA
import wandb
import torch.nn.functional as F

def resize_input(input_matrix):
    input_matrix = input_matrix.unsqueeze(0).unsqueeze(0)    # shape [1, 1, H, W]
    input_matrix = F.interpolate(input_matrix, size=(250, 250), mode='bilinear', align_corners=False)
    input_matrix = input_matrix.squeeze(0)  # back to [1, 250, 250]
    # min_v = input_matrix.min()
    # max_v = input_matrix.max()
    
    # input_matrix = (input_matrix - min_v) / (max_v - min_v)
    return input_matrix    
        
#Run: python3 -u 7_train_ACMIL.py --train_cohort OPX --mutation MT 

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--learning_method', default='clusterCNN', type=str, help=': e.g., acmil, abmil')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--train_cohort', default= 'OPX', type=str, help='TCGA_PRAD or OPX')
parser.add_argument('--valid_cohort', default='TCGA_PRAD', type=str, help='data set name: TAN_TMA_Cores or OPX or TCGA_PRAD')
parser.add_argument('--mutation', default='MT', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_overlap', default=100, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--out_folder', default= 'pred_out_041025_clustering', type=str, help='out folder name')



# ===============================================================
#     Model Para
# ===============================================================
parser.add_argument('--BATCH_SIZE', default=16, type=int, help='batch size')
#parser.add_argument('--DROPOUT', default=0, type=int, help='drop out rate')
parser.add_argument('--DIM_OUT', default=128, type=int, help='')


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    ####################################
    ######      USERINPUT       ########
    ####################################
    ALL_LABEL = ["AR","HR","PTEN","RB1","TP53","TMB","MSI_POS"]

            
    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'        

    ######################
    #Create output-dir
    ######################
    folder_name1 = args.fe_method + '/TrainOL' + str(args.train_overlap) +  '_TestOL' + str(args.test_overlap) + '_TFT' + str(args.tumor_frac)  + "/"
    outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                           'trainCohort_' + args.train_cohort,
                           args.learning_method,
                           folder_name1,
                           args.mutation)
    outdir1 =  outdir0  + "/saved_model/"
    outdir2 =  outdir0  + "/model_para/"
    outdir3 =  outdir0  + "/logs/"
    outdir4 =  outdir0  + "/predictions/"
    outdir5 =  outdir0  + "/perf/"
    outdir_list = [outdir0,outdir1,outdir2,outdir3,outdir4,outdir5]
    
    for out_path in outdir_list:
        create_dir_if_not_exists(out_path)

    ##################
    #Select GPU
    ##################
    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    ################################################
    #     Model ready data 
    ################################################
    data_path = os.path.join(proj_dir, 'intermediate_data/5_model_ready_data_cluster', 
                             args.train_cohort, 
                             'feature_' + args.fe_method,
                             'TFT' + str(args.tumor_frac))
    train_data = torch.load(os.path.join(data_path, args.train_cohort + '_train_data.pth'))
    test_data = torch.load(os.path.join(data_path, args.train_cohort + '_test_data.pth'))
    val_data = torch.load(os.path.join(data_path, args.valid_cohort + '_val_data.pth'))

    #reise input
    train_data2 = [(torch.FloatTensor(resize_input(x[0])), x[1], x[2], x[3]) for x in train_data]
    test_data2 = [(torch.FloatTensor(resize_input(x[0])), x[1], x[2], x[3]) for x in test_data]
    val_data2 = [(torch.FloatTensor(resize_input(x[0])), x[1], x[2], x[3]) for x in val_data]

        
    # Move the model to the device (GPU or CPU)
    model = RESNET_CLUSTER()
    model = model.to(device)
    
    # Define loss function and optimizer
    focal_alpha = 0.2
    focal_gamma = 5
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
    
        
    ####################################################
    #Dataloader for training
    ####################################################
    set_seed(0)
    train_loader = DataLoader(dataset=train_data2, batch_size=args.BATCH_SIZE, shuffle=False)
    n_batches = len(train_loader)
    num_epochs = 100
    momentum = 0.9
    learning_rate = 0.1 
    weight_decay = 0.0005
    use_scheduler = False # scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(1, num_epochs+1):        
        model.train()  # needed if we switch to eval within each epoch
    
        running_loss = 0.0
        correct = 0
        total = 0
    
        for batch_idx, (images, labels, sp_id, pt_id) in enumerate(train_loader, 1):
            
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            #forward
            slide_preds = model(images)
           
            
            loss = 0
            
            for i in range(len(slide_preds)):
                
                if args.train_cohort == 'OPX':
                    pred_prob = torch.sigmoid(slide_preds[i]) #[BS, 1]
                    pred = pred_prob.round() ##[BS, 1]
                    loss += criterion(slide_preds[i], labels[:,:,i])
                elif args.train_cohort == 'TCGA_PRAD':
                    if i != 5:
                        pred_prob = torch.sigmoid(slide_preds[i]) #[BS, 1]
                        pred = pred_prob.round() ##[BS, 1]
                        loss += criterion(slide_preds[i], labels[:,:,i])


            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            
            #loss
            running_loss += loss.item()
   
        # Train losses
        train_loss = running_loss / n_batches
    
        if epoch%10 == 0:
            print('epoch',epoch, ":" ,train_loss)



#TEST
SELECTED_LABEL = ["AR","HR","PTEN","RB1","TP53","TMB","MSI_POS"]
test_loader = DataLoader(dataset=test_data2, batch_size=1, shuffle=False)
test_ids = [x[-2] for x in test_data2]

y_pred_test, y_predprob_task_test, y_true_task_test = predict_clustercnn(model, test_loader, criterion, device, n_task = 7)


pred_df_list = []
perf_df_list = []
for i in range(7):
    if args.train_cohort == 'OPX':
        #Calibration
        pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], test_ids, SELECTED_LABEL[i], THRES = np.quantile(y_predprob_task_test[i],0.5))
    elif args.train_cohort == 'TCGA_PRAD':
        if i != 5:
            pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], test_ids, SELECTED_LABEL[i], THRES = np.quantile(y_predprob_task_test[i],0.5))

    pred_df_list.append(pred_df)
    perf_df_list.append(perf_df)

all_perd_df = pd.concat(pred_df_list)
all_perf_df = pd.concat(perf_df_list)
all_perd_df.to_csv(outdir4 + args.train_cohort + "_pred_df.csv",index = False)
all_perf_df.to_csv(outdir5 + args.train_cohort +  "_perf.csv",index = True)
print(all_perf_df)

# ###############################################################################################
# #Plot clustered input image
# ###############################################################################################
# for target_label in SELECTED_LABEL:
#     cond1 = all_perd_df['Y_True'] == all_perd_df['Pred_Class']
#     cond2 = all_perd_df['OUTCOME'] == target_label
#     cond3 = all_perd_df['Y_True'] = 1
#     true_pos_ids = list(all_perd_df.loc[cond1 & cond2 & cond3,'SAMPLE_IDs'])
#     plot_path = os.path.join(outdir4, target_label)
#     create_dir_if_not_exists(plot_path)
    
#     for pt in true_pos_ids:
#         matches = [item for item in test_data if item[2] == pt]
#         plot_cluster_image(matches[0][0], '/' + pt , plot_path, colorbar = False)
        
# ###############################################################################################
# #MoranI
# ###############################################################################################
# train_ids = [x[-2] for x in train_data]
# moranI_list = []
# for pt in train_ids:
#     matches = [item for item in train_data if item[2] == pt]    
#     import libpysal as ps
#     from esda.moran import Moran
#     values = matches[0][0]
#     flat_values = values.flatten()    
#     # Create a spatial weights matrix for the 2D grid
#     w = ps.weights.lat2W(values.shape[0], values.shape[1])
#     # Calculate Moran's I
#     moran = Moran(flat_values, w)
#     print("Moran's I:", moran.I)
#     print("p-value:", moran.p_sim)
#     cur_moran_df = pd.DataFrame({'moranI': moran.I, 'SAMPLE_ID': pt,
#                                  'AR': matches[0][1][0][0].numpy().item(),
#                                  'HR': matches[0][1][0][1].numpy().item(),
#                                  'PTEN': matches[0][1][0][2].numpy().item(),
#                                  'RB1': matches[0][1][0][3].numpy().item(),
#                                  'TP53': matches[0][1][0][4].numpy().item(),
#                                  'TMB': matches[0][1][0][5].numpy().item(),
#                                  'MSI_POS': matches[0][1][0][6].numpy().item()}, index = [0])
#     moranI_list.append(cur_moran_df)

# moranI_df = pd.concat(moranI_list)
# moranI_df.to_csv(outdir4 + args.train_cohort +  "_moransI_train.csv",index = True)

# # List of gene columns (update as needed)
# genes = ['AR', 'HR', 'PTEN', 'RB1', 'TP53', 'TMB', 'MSI_POS']
# from scipy.stats import mannwhitneyu
# import matplotlib.pyplot as plt
# # Boxplot data setup
# data_0 = []  # Y = 0
# data_1 = []  # Y = 1
# p_values = []

# for gene in genes:
#     group0 = moranI_df[moranI_df[gene] == 0]['moranI']
#     group1 = moranI_df[moranI_df[gene] == 1]['moranI']
#     data_0.append(group0)
#     data_1.append(group1)
    
#     # Mann-Whitney U test
#     if len(group0) > 0 and len(group1) > 0:
#         stat, p = mannwhitneyu(group0, group1, alternative='two-sided')
#         p_values.append(p)
#     else:
#         p_values.append(None)

# # Set up positions
# positions_0 = [i * 2.0 - 0.3 for i in range(len(genes))]
# positions_1 = [i * 2.0 + 0.3 for i in range(len(genes))]
# mid_positions = [(x + y) / 2 for x, y in zip(positions_0, positions_1)]

# plt.figure(figsize=(12, 6))

# # Boxplots
# bp0 = plt.boxplot(data_0, positions=positions_0, widths=0.5, patch_artist=True)
# bp1 = plt.boxplot(data_1, positions=positions_1, widths=0.5, patch_artist=True)

# # Colors
# # Colors
# for box in bp0['boxes']:
#     box.set(facecolor='skyblue')
# for box in bp1['boxes']:
#     box.set(facecolor='lightcoral')

# for i, (x0, x1, p, y0s, y1s) in enumerate(zip(positions_0, positions_1, p_values, data_0, data_1)):
#     if p is not None:
#         # Get the lowest point among both boxes
#         y_min = min(min(y0s), min(y1s))
#         bracket_height = y_min - 0.015   # horizontal line
#         cap_height = y_min - 0.005       # vertical caps (arms)
#         text_height = y_min - 0.03       # label below bracket

#         # Make sure it doesn’t go below y-axis limit
#         bracket_height = max(bracket_height, 0.52)
#         cap_height = max(cap_height, 0.53)
#         text_height = max(text_height, 0.51)

#         # Draw upward-facing bracket under the boxplot
#         plt.plot([x0, x0, x1, x1],
#                  [cap_height, bracket_height, bracket_height, cap_height],
#                  lw=1.5, c='black')

#         # Label just under the bracket
#         p_label = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
#         plt.text((x0 + x1) / 2, text_height, p_label,
#                  ha='center', va='top', fontsize=10)


# # X-axis setup
# mid_positions = [(x + y) / 2 for x, y in zip(positions_0, positions_1)]
# plt.xticks(mid_positions, genes, rotation=45)

# plt.title("Moran's I by Gene Alteration Status")
# plt.ylabel("Moran's I")

# # ✅ Move legend BELOW the x-axis
# plt.legend([bp0["boxes"][0], bp1["boxes"][0]], ["Y=0", "Y=1"],
#            loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.ylim(0.5, 1)
# plt.tight_layout()  # important to prevent clipping
# plt.savefig(outdir4 + 'moransI_train.png', dpi=300, bbox_inches='tight')
# plt.show()
            

#bootstrap perforance
ci_list = []
for i in range(7):
    print(i)
    if args.train_cohort == 'OPX':
        cur_pred_df = all_perd_df.loc[all_perd_df['OUTCOME'] == SELECTED_LABEL[i]]
        cur_ci_df = bootstrap_ci_from_df(cur_pred_df, y_true_col='Y_True', y_pred_col='Pred_Class', y_prob_col='Pred_Prob', num_bootstrap=1000, ci=95, seed=42)
        cur_ci_df['OUTCOME'] = SELECTED_LABEL[i]
    elif args.train_cohort == 'TCGA_PRAD':
        if i != 5:
            cur_pred_df = all_perd_df.loc[all_perd_df['OUTCOME'] == SELECTED_LABEL[i]]
            cur_ci_df = bootstrap_ci_from_df(cur_pred_df, y_true_col='Y_True', y_pred_col='Pred_Class', y_prob_col='Pred_Prob', num_bootstrap=1000, ci=95, seed=42)
            cur_ci_df['OUTCOME'] = SELECTED_LABEL[i]
    ci_list.append(cur_ci_df)
ci_final_df = pd.concat(ci_list)
print(ci_final_df)
ci_final_df.to_csv(outdir5 + args.train_cohort + "_TEST_perf_bootstrap.csv",index = True)

            
            
#VAL
SELECTED_LABEL = ["AR","HR","PTEN","RB1","TP53","TMB","MSI_POS"]
val_loader = DataLoader(dataset=val_data2, batch_size=1, shuffle=False)
val_ids = [x[-2] for x in val_data2]

y_pred_test, y_predprob_task_test, y_true_task_test = predict_clustercnn(model, val_loader, criterion, device, n_task = 7)


pred_df_list = []
perf_df_list = []
for i in range(7):
    if args.valid_cohort == 'OPX':
        pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], val_ids, SELECTED_LABEL[i], THRES = np.quantile(y_predprob_task_test[i],0.5))
    elif args.valid_cohort == 'TCGA_PRAD':
        #Calibration
        if i != 5:
            pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], val_ids, SELECTED_LABEL[i], THRES = np.quantile(y_predprob_task_test[i],0.5))
    pred_df_list.append(pred_df)
    perf_df_list.append(perf_df)

all_perd_df = pd.concat(pred_df_list)
all_perf_df = pd.concat(perf_df_list)
all_perd_df.to_csv(outdir4 + args.valid_cohort + "_pred_df.csv",index = False)
all_perf_df.to_csv(outdir5 + args.valid_cohort +  "_perf.csv",index = True)
print(all_perf_df)

#bootstrap perforance
ci_list = []
for i in range(7):
    print(i)
    if args.valid_cohort == 'OPX':
        cur_pred_df = all_perd_df.loc[all_perd_df['OUTCOME'] == SELECTED_LABEL[i]]
        cur_ci_df = bootstrap_ci_from_df(cur_pred_df, y_true_col='Y_True', y_pred_col='Pred_Class', y_prob_col='Pred_Prob', num_bootstrap=1000, ci=95, seed=42)
        cur_ci_df['OUTCOME'] = SELECTED_LABEL[i]
    elif args.valid_cohort == 'TCGA_PRAD':
        if i != 5:
            cur_pred_df = all_perd_df.loc[all_perd_df['OUTCOME'] == SELECTED_LABEL[i]]
            cur_ci_df = bootstrap_ci_from_df(cur_pred_df, y_true_col='Y_True', y_pred_col='Pred_Class', y_prob_col='Pred_Prob', num_bootstrap=1000, ci=95, seed=42)
            cur_ci_df['OUTCOME'] = SELECTED_LABEL[i]
    ci_list.append(cur_ci_df)
ci_final_df = pd.concat(ci_list)
print(ci_final_df)
ci_final_df.to_csv(outdir5 + args.valid_cohort + "_TEST_perf_bootstrap.csv",index = True)


#TODO:
#plot predicted true postive tile clustereds
#Train and text on OPX, attention module
#Send cancer decetion results for Michale to check
#Make slides how to use cluterings and morans I
#Make slides for interpretation anlaysis
