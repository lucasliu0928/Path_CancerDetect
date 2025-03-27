#!/usr/bin/env python
# coding: utf-8

#NOTE: use python env acmil in ACMIL folder
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
from Utils import get_downsample_factor, get_image_at_target_mag
from Utils import do_mask_original
import cv2
from Eval import compute_performance, plot_LOSS, compute_performance_each_label, get_attention_and_tileinfo, get_performance, plot_roc_curve
from Eval import bootstrap_ci_from_df, calibrate_probs_isotonic
from train_utils import pull_tiles, FocalLoss, get_feature_idexes
from train_utils import convert_to_dict, prediction_sepatt, BCE_Weighted_Reg, BCE_Weighted_Reg_focal, compute_loss_for_all_labels_sepatt
from ACMIL import ACMIL_GA_MultiTask, predict_v2, train_one_epoch_multitask_V2, evaluate_multitask_V2
warnings.filterwarnings("ignore")


#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
from utils.utils import save_model, Struct, set_seed
import yaml
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint

import argparse
import torch
from torch.utils.data import DataLoader

from utils.utils import save_model, Struct, set_seed
from datasets.datasets import build_HDF5_feat_dataset
from architecture.transformer import ACMIL_GA #ACMIL_GA
from architecture.transformer import ACMIL_MHA
import torch.nn.functional as F
import wandb


#Run: python3 -u 7_train_dynamic_tiles_ACMIL_AddReg_working-MultiTasking_NewFeature_TCGA_ACMIL_UpdatedOPX.py --SELECTED_FOLD 0 --loss_method ''

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--SELECTED_FOLD', default='0', type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--loss_method', default='', type=str, help='ATTLOSS or ''')



if __name__ == '__main__':
    
    args = parser.parse_args()
    ####################################
    ######      USERINPUT       ########
    ####################################
    ALL_LABEL = ["AR","HR","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]
    TUMOR_FRAC_THRES = 0.9 
    feature_extraction_method = 'uni2' #retccl, uni1, prov_gigapath
    learning_method = "acmil"
    SELECTED_FEATURE = get_feature_idexes(feature_extraction_method, include_tumor_fraction = False)
    N_FEATURE = len(SELECTED_FEATURE)
    
    
    #For RB1: gamma = 2, focal_alpha = 0.1
    
    
    # Define different values for alpha and gamma
    alpha_values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6]  # Example alpha values
    gamma_values = [2,    2,    2,   2,  2,   2,  10]  # Example gamma values
    
    # focal_gamma = 5 #10 seems good too
    # focal_alpha = 0.80
    #Best before
    focal_gamma = 10
    focal_alpha = 0.6
    loss_method = args.loss_method #ATTLOSS
    
    ################################
    #model Para
    BATCH_SIZE  = 1
    DROPOUT = 0
    DIM_OUT = 128
    SELECTED_MUTATION = "MT"
    SELECTED_FOLD = args.SELECTED_FOLD
    arch = 'ga_mt' #ga_mt or ga
    
        
    ################################
    # get config
    config_dir = "myconf.yml"
    with open(config_dir, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        #c.update(vars(args))
        conf = Struct(**c)
    
    conf.train_epoch = 100
    conf.D_feat = N_FEATURE
    conf.D_inner = DIM_OUT
    
    if learning_method == 'abmil':
        conf.n_token = 1
        conf.mask_drop = 0
        conf.n_masked_patch = 0
    elif learning_method == 'acmil':
        conf.n_token = 3
        conf.mask_drop = 0.6
        conf.n_masked_patch = 0
        
    conf.n_class = 1
    conf.wandb_mode = 'disabled'
    conf.n_task = 7
    conf.lr = 0.001
    
    # Print all key-value pairs in the conf object
    for key, value in conf.__dict__.items():
        print(f"{key}: {value}")
        
    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    folder_name_overlap = "IMSIZE250_OL100"
    folder_name_nonoverlap = "IMSIZE250_OL0"
    feature_path_opx_train =  os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', "OPX", folder_name_overlap, 'feature_' + feature_extraction_method, 'TFT' + str(TUMOR_FRAC_THRES))
    feature_path_opx_test =  os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', "OPX", folder_name_nonoverlap, 'feature_' + feature_extraction_method, 'TFT' + str(TUMOR_FRAC_THRES))
    train_val_test_id_path =  os.path.join(proj_dir + 'intermediate_data/6_Train_TEST_IDS', 'TrainOL100_TestOL0_TFT' + str(TUMOR_FRAC_THRES))
    
    
    ######################
    #Create output-dir
    ################################################
    folder_name1 = feature_extraction_method + '/TrainOL100_TestOL0_TFT' + str(TUMOR_FRAC_THRES)  + "/"
    outdir0 =  proj_dir + "intermediate_data/pred_out032025_ACMIL_noATTLOSS" + "/" + folder_name1 + 'FOLD' + str(SELECTED_FOLD) + '/' + SELECTED_MUTATION + "/" 
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
    opx_data_ol100 = torch.load(feature_path_opx_train + '/OPX_data.pth')
    opx_data_ol0 = torch.load(feature_path_opx_test + '/OPX_data.pth')
    
    
    ################################################
    #Get train, test IDs
    ################################################
    train_test_val_id_df = pd.read_csv(os.path.join(train_val_test_id_path, "train_test_split.csv"))
    train_ids_all = list(train_test_val_id_df.loc[train_test_val_id_df['FOLD' + str(SELECTED_FOLD)] == 'TRAIN', 'SAMPLE_ID'])
    test_ids_all = list(train_test_val_id_df.loc[train_test_val_id_df['FOLD' + str(SELECTED_FOLD)] == 'TEST', 'SAMPLE_ID'])
    val_ids_all = list(train_test_val_id_df.loc[train_test_val_id_df['FOLD' + str(SELECTED_FOLD)] == 'VALID', 'SAMPLE_ID'])
    
    
    print(len(train_ids_all))
    print(len(test_ids_all))
    print(len(val_ids_all))
    
    
    
    ################################################
    #IDS
    ################################################
    opx_ids_ol100 = [x[-2] for x in opx_data_ol100]
    opx_ids_ol0 = [x[-2] for x in opx_data_ol0]
    
 
    ################################################
    #Get Train, test, val data
    ################################################
    #Train:
    inc_idx = [opx_ids_ol100.index(x) for x in train_ids_all]
    train_data = Subset(opx_data_ol100, inc_idx)
    train_ids =  list(Subset(opx_ids_ol100, inc_idx))
    
    #Val:
    inc_idx = [opx_ids_ol100.index(x) for x in val_ids_all]
    val_data = Subset(opx_data_ol100, inc_idx)
    val_ids =  list(Subset(opx_ids_ol100, inc_idx))
    
    #Test:
    inc_idx = [opx_ids_ol0.index(x) for x in test_ids_all]
    test_data = Subset(opx_data_ol0, inc_idx)
    test_ids =  list(Subset(opx_ids_ol0, inc_idx))
    

    ####################################################
    #Dataloader for training
    ####################################################
    train_data2 = [item[:-3] for item in train_data] #only keep the needed for training
    test_data2 = [item[:-3] for item in test_data] #only keep the needed for training
    val_data2 = [item[:-3] for item in val_data] #only keep the needed for training
    
    train_loader = DataLoader(dataset=train_data2, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_data2, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(dataset=val_data2, batch_size=BATCH_SIZE, shuffle=False)
    
    
    ####################################################
    # define network
    ####################################################
    if arch == 'ga':
        model = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop)
    elif arch == 'ga_mt':
        model = ACMIL_GA_MultiTask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
    else:
        model = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
    model.to(device)
    
    
    # Create a list of FocalLoss criteria with different alpha and gamma
    criterion = [FocalLoss(alpha=a, gamma=g, reduction='mean') for a, g in zip(alpha_values, gamma_values)]
    #criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
    
    # define optimizer, lr not important at this point
    optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.wd)
    
    ckpt_dir = outdir1 + SELECTED_MUTATION + "/"
    create_dir_if_not_exists(ckpt_dir)
    

    ####################################################
    #            Train 
    ####################################################
    set_seed(0)
    # define optimizer, lr not important at this point
    optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=conf.lr, weight_decay=conf.wd)
    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    train_epoch = conf.train_epoch
    for epoch in range(train_epoch):
        # train_one_epoch_multitask(model, criterion, train_loader, optimizer0, device, epoch, conf, loss_method)
        # val_auc, val_acc, val_f1, val_loss = evaluate_multitask(model, criterion, val_loader, device, conf, 'Val')
        # test_auc, test_acc, test_f1, test_loss = evaluate_multitask(model, criterion, test_loader, device, conf, 'Test')
        train_one_epoch_multitask_V2(model, criterion, train_loader, optimizer0, device, epoch, conf, loss_method)
        val_auc, val_acc, val_f1, val_loss = evaluate_multitask_V2(model, criterion, val_loader, device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss = evaluate_multitask_V2(model, criterion, test_loader, device, conf, 'Test')
    
        save_model(conf=conf, model=model, optimizer=optimizer0, epoch=epoch,
            save_path=os.path.join(ckpt_dir + 'checkpoint_' + 'epoch' + str(epoch) + '.pth'))
    print("Results on best epoch:")
    print(best_state)
    wandb.finish()
    
    
    
    # ###################################################
    # #           Test 
    # ###################################################
    # # define network
    # if arch == 'ga':
    #     model2 = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop)
    # elif arch == 'ga_mt':
    #     model2 = ACMIL_GA_MultiTask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
    # else:
    #     model2 = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
    # model2.to(device)
    
    # # Load the checkpoint
    # #checkpoint = torch.load(ckpt_dir + 'checkpoint-best.pth')
    # checkpoint = torch.load(ckpt_dir + 'checkpoint_epoch99.pth')
    
    # # Load the state_dict into the model
    # model2.load_state_dict(checkpoint['model'])
    
    y_pred_tasks_test, y_predprob_task_test, y_true_task_test = predict_v2(model, test_loader, device, conf, 'Test')
    y_pred_tasks_val,  y_predprob_task_val, y_true_task_val = predict_v2(model, val_loader, device, conf, 'Test')
    
    
    pred_df_list = []
    perf_df_list = []
    for i in range(conf.n_task):
        #Calibration
        prob_calibrated = calibrate_probs_isotonic(y_true_task_val[i], y_predprob_task_val[i], y_predprob_task_test[i])
        pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], test_ids, ALL_LABEL[i], THRES = np.quantile(y_predprob_task_test[i],0.5))
        pred_df_list.append(pred_df)
        perf_df_list.append(perf_df)
    
    all_perd_df = pd.concat(pred_df_list)
    all_perf_df = pd.concat(perf_df_list)
    print(all_perf_df)
    
    all_perd_df.to_csv(outdir4 + "/n_token" + str(conf.n_token) + "_TEST_pred_df.csv",index = False)
    all_perf_df.to_csv(outdir5 + "/n_token" + str(conf.n_token) + "_TEST_perf.csv",index = True)
    print(round(all_perf_df['AUC'].mean(),2))

    #bootstrap perforance
    ci_list = []
    for i in range(conf.n_task):
        print(i)
        cur_pred_df = all_perd_df.loc[all_perd_df['OUTCOME'] == ALL_LABEL[i]]
        cur_ci_df = bootstrap_ci_from_df(cur_pred_df, y_true_col='Y_True', y_pred_col='Pred_Class', y_prob_col='Pred_Prob', num_bootstrap=1000, ci=95, seed=42)
        cur_ci_df['OUTCOME'] = ALL_LABEL[i]
        ci_list.append(cur_ci_df)
    ci_final_df = pd.concat(ci_list)
    ci_final_df.to_csv(outdir5 + "/n_token" + str(conf.n_token) + "_TEST_perf_bootstrap.csv",index = True)
    
    
    pred_msi = all_perd_df.loc[all_perd_df['OUTCOME'] == 'MSI_POS']
    pred_msi
    
    
    # Separate Pred_Prob values based on Y_True values
    pred_prob_0 = pred_msi[pred_msi["Y_True"] == 0]["Pred_Prob"]
    pred_prob_1 = pred_msi[pred_msi["Y_True"] == 1]["Pred_Prob"]
    
    # Create the boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot([pred_prob_0, pred_prob_1], labels=["Y_True = 0", "Y_True = 1"])
    
    # Add labels and title
    plt.xlabel("Y_True")
    plt.ylabel("Predicted Probability (Pred_Prob)")
    plt.title("Boxplot of Pred_Prob for Y_True = 0 and 1")
    
    # Show the plot
    plt.show()
    
    
    plot_roc_curve(list(pred_msi['Pred_Prob']),list(pred_msi['Y_True']))
    
    
    #PLOT:
    pred_df = all_perd_df
    SELECTED_LABEL = ['MSI_POS']
    #Get True Postives
    true_postive_ids = {}
    for label in SELECTED_LABEL:
        cond = (pred_df['Y_True'] == pred_df['Pred_Class']) & (pred_df['Y_True'] == 1) & (pred_df['OUTCOME'] == label)
        cur_pred_df = pred_df.loc[cond]
        cur_ids = list(cur_pred_df['SAMPLE_IDs'])
        true_postive_ids[label] = cur_ids
    
    #Get true nagative
    true_negative_ids = {}
    for label in SELECTED_LABEL:
        cond = (pred_df['Y_True'] == pred_df['Pred_Class']) & (pred_df['Y_True'] == 0) & (pred_df['OUTCOME'] == label)
        cur_pred_df = pred_df.loc[cond]
        cur_ids = list(cur_pred_df['SAMPLE_IDs'])
        true_negative_ids[label] = cur_ids
    
    ####################################################################################
    #Atention scores
    ####################################################################################
    save_image_size = 250
    pixel_overlap = 0
    mag_extract = 20
    limit_bounds = True
    TOP_K = 5
    mag_target_prob = 2.5
    smooth = True
    mag_target_tiss = 1.25
    
    def get_attention_and_tileinfo(pt_label_df, patient_att_score):    
        #Get label
        pt_label_df.reset_index(drop = True, inplace = True)
    
        #Get attention
        cur_att  = pd.DataFrame({'ATT':list(minmax_normalize(patient_att_score))})
        cur_att.reset_index(drop = True, inplace = True)
    
        #Comb
        cur_att_df = pd.concat([pt_label_df,cur_att], axis = 1)
        cur_att_df.reset_index(drop = True, inplace = True)
    
        return cur_att_df
    
    

    #Load all test data
    opx_data_ol0 = torch.load(feature_path_opx_test + '/OPX_data.pth')
    opx_ids_ol0 = [x[-2] for x in opx_data_ol0]
    

    selected_ids = true_postive_ids[SELECTED_LABEL[0]]
    wsi_path = proj_dir + '/data/OPX/'
    #branches = 2
    
    for pt in selected_ids:
        i =  opx_ids_ol0.index(pt)
        print(pt)
        print(opx_ids_ol0[i])
    
        save_location = outdir4 + SELECTED_LABEL[0] + "/"
        save_location =  save_location  + pt + "/"
        create_dir_if_not_exists(save_location)
        
        _file = wsi_path + pt + ".tif"
        oslide = openslide.OpenSlide(_file)
        save_name = str(Path(os.path.basename(_file)).with_suffix(''))
    
    
        first_batch = opx_data_ol0[i]
        feat = first_batch[0].unsqueeze(0).to(device)
        sub_preds, slide_preds, attn = model(feat)
        label_index = ALL_LABEL.index(SELECTED_LABEL[0])
    
        #Get attention
        cur_att = attn[label_index] #att no softmax 
        #cur_att_softmax = torch.softmax(cur_att, dim=-1) #att softmax over tiles
    
        #Mean
        cur_pt_att = cur_att.mean(dim = 1).squeeze().cpu().detach().numpy() #Mean aross channels without softmax
        #cur_pt_att = cur_att_softmax.mean(dim = 1).squeeze().cpu().detach().numpy()  #Mean aross channels with softmax
        
        #cur_pt_att = cur_att[0,branches,:].cpu().detach().numpy() #branch
        
        #Get all tile info include noncancer tile
        alltileinfo_dir = proj_dir + 'intermediate_data/2_cancer_detection/OPX/' + "IMSIZE250_OL0" + "/"
        tile_info_df = pd.read_csv(alltileinfo_dir + pt + "/ft_model/"  + save_name + "_TILE_TUMOR_PERC.csv")
        #Combine current pt_info an all tile info
        #cur_pt_info = tile_info_df.merge(cur_pt_info, on = list(tile_info_df.columns), how = "left")
        cur_pt_info = first_batch[3]
        cur_att_df = get_attention_and_tileinfo(cur_pt_info, cur_pt_att)
        #cur_att_df.loc[pd.isna(cur_att_df['ATT']),'ATT'] = 0.0001
        
        
        #Generate tiles
        tiles, tile_lvls, physSize, base_mag = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)
        
        #get level 0 size in px
        l0_w = oslide.level_dimensions[0][0]
        l0_h = oslide.level_dimensions[0][1]
        
        #1.25x tissue detection for mask
        if 'OPX' in pt:
            rad_tissue = 5
        elif '(2017-0133)' in pt:
            rad_tissue = 2
        lvl_resize_tissue = get_downsample_factor(base_mag,target_magnification = mag_target_tiss) #downsample factor
        lvl_img = get_image_at_target_mag(oslide,l0_w, l0_h,lvl_resize_tissue)
        tissue, he_mask = do_mask_original(lvl_img, lvl_resize_tissue, rad = rad_tissue)
        
        #2.5x for probability maps
        lvl_resize = get_downsample_factor(base_mag,target_magnification = mag_target_prob) #downsample factor
        x_map = np.zeros((int(np.ceil(l0_h/lvl_resize)),int(np.ceil(l0_w/lvl_resize))), float)
        x_count = np.zeros((int(np.ceil(l0_h/lvl_resize)),int(np.ceil(l0_w/lvl_resize))), float)
        
        
        for index, row in cur_att_df.iterrows():
            cur_xy = row['TILE_XY_INDEXES'].strip("()").split(", ")
            x ,y = int(cur_xy[0]) , int(cur_xy[1])
            
            #Extract tile for prediction
            lvl_in_deepzoom = tile_lvls.index(mag_extract)
            tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords(tiles, lvl_in_deepzoom, x, y) #get tile coords
            map_xstart, map_xend, map_ystart, map_yend = get_map_startend(tile_starts,tile_ends,lvl_resize) #Get current tile position in map
        
            #Store predicted probabily in map and count
            try: 
                x_count[map_xstart:map_xend,map_ystart:map_yend] += 1
                x_map[map_xstart:map_xend,map_ystart:map_yend] += row['ATT']
            except:
                pass
        
        print('post-processing')
        x_count = np.where(x_count < 1, 1, x_count)
        x_map = x_map / x_count
        x_map[x_map>1]=1
        
        #Get the following before smooth
        he_mask = cv2.resize(np.uint8(he_mask),(x_map.shape[1],x_map.shape[0])) #resize to output image size
        cond1 = he_mask < 1 #Background
        cond2 = (he_mask == 1) & (x_map == 0) #is tissue, but not selected
        smooth = True
        
        if smooth == True:
            #x_sm = filters.gaussian(x_map, sigma=0)
            x_sm = np.where(x_map != 0, filters.gaussian(x_map, sigma=10), x_map)
        if smooth == False:
            x_sm = x_map
        
        #TODO:
        #get cancer_mask:
        # cancer_mask == 
        # x_sm[(he_mask == 1) & (x_sm == 0)] = 0.1 #If tissue map value > 1, then x_sm = 1
        x_sm[cond1] = 0 #Background
        x_sm[cond2] = 0.1 #Is tissue, but not selected 
        
        # Define the colors for the sequential colormap (black to fluorescent green)
        colors = ["#4B0082", "#39FF14"]  # Black to Fluorescent Green
        # Create the sequential colormap
        cmap_name = "black_to_fluorescent_green"
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.colors import ListedColormap
        sequential_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
        cmap =  plt.cm.Spectral_r #sequential_cmap # plt.cm.YlGn_r
        cmap_colors = cmap(np.arange(cmap.N))
        cmap_colors[0] = np.array([0.95, 0.95, 0.95, 1]) #np.array([1, 1, 1, 1])  # Set the first color (corresponding to 0) to white
        cmap_colors[1] = np.array([0, 0, 0.545, 1])  # RGB for dark blue
        custom_cmap = ListedColormap(cmap_colors)
        
        plt.imshow(x_sm, cmap=custom_cmap) #Spectral_r
        plt.colorbar()
        plt.savefig(os.path.join(save_location, save_name + '_attention.png'), dpi=500,bbox_inches='tight')
        #plt.show()
        plt.close()
        
        
        #Top attented tiles
        save_location2 = save_location + "top_tiles/"
        create_dir_if_not_exists(save_location2)
        
        #Get a Attention, and corresponding tiles
        cur_att_df= cur_att_df.sort_values(by = ['ATT'], ascending = False) 
        cur_pulled_img_obj = pull_tiles(cur_att_df.iloc[0:TOP_K], tiles, tile_lvls)
                
        for i in range(TOP_K):
            cur_pulled_img = cur_pulled_img_obj[i][0] #image
            cur_pulled_att = cur_pulled_img_obj[i][1] #attentiom
            cur_pulled_coord = cur_pulled_img_obj[i][2].strip("()").split(", ")  #att tile map coordiates
            coord_save_name = '[xs' + cur_pulled_coord[0] + '_xe' + cur_pulled_coord[1] + '_ys' + cur_pulled_coord[2] + '_ye' + cur_pulled_coord[3] + "]"
            tile_save_name = "ATT" + str(round(cur_pulled_att,2)) + "_MAPCOORD" +  coord_save_name +  ".png"
            cur_pulled_img.save(os.path.join(save_location2, tile_save_name))
        
        #Bot attented tiles
        save_location2 = save_location + "bot_tiles/"
        create_dir_if_not_exists(save_location2)
        
        #Get a Attention, and corresponding tiles
        cur_att_df= cur_att_df.sort_values(by = ['ATT'], ascending = True) 
        cur_pulled_img_obj = pull_tiles(cur_att_df.iloc[0:TOP_K], tiles, tile_lvls)
        
        for i in range(TOP_K):
            cur_pulled_img = cur_pulled_img_obj[i][0] #image
            cur_pulled_att = cur_pulled_img_obj[i][1] #attentiom
            cur_pulled_coord = cur_pulled_img_obj[i][2].strip("()").split(", ")  #att tile map coordiates
            coord_save_name = '[xs' + cur_pulled_coord[0] + '_xe' + cur_pulled_coord[1] + '_ys' + cur_pulled_coord[2] + '_ye' + cur_pulled_coord[3] + "]"
            tile_save_name = "ATT" + str(round(cur_pulled_att,2)) + "_MAPCOORD" +  coord_save_name +  ".png"
            cur_pulled_img.save(os.path.join(save_location2, tile_save_name))
    
