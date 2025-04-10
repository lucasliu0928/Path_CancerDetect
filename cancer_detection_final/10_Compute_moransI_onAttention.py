#!/usr/bin/env python
# coding: utf-8

#NOTE: use python env acmil in ACMIL folder
"""
Created on Wed Apr  9 14:02:47 2025

@author: jliu6
"""

import pandas as pd
import libpysal as ps
from esda.moran import Moran
import geopandas as gpd
import matplotlib.pyplot as plt
from splot.esda import moran_scatterplot
import numpy as np
import sys
import os
import numpy as np
import openslide
#%matplotlib inline
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import warnings
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from skimage import filters
sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists
from Utils import generate_deepzoom_tiles, extract_tile_start_end_coords, get_map_startend
from Utils import get_downsample_factor
from Utils import minmax_normalize, set_seed
from Utils import get_image_at_target_mag
from Utils import do_mask_original
from Eval import plot_LOSS, compute_performance_each_label, get_attention_and_tileinfo
from Eval import boxplot_predprob_by_mutationclass, get_performance, plot_roc_curve
from Eval import bootstrap_ci_from_df, calibrate_probs_isotonic
from train_utils import pull_tiles, FocalLoss, get_feature_idexes, get_partial_data, get_train_test_val_data, get_external_validation_data
from ACMIL import ACMIL_GA_MultiTask, predict_v2, train_one_epoch_multitask, evaluate_multitask
warnings.filterwarnings("ignore")


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


#Run: python3 -u 7_train_dynamic_tiles_ACMIL_AddReg_working-MultiTasking_NewFeature_TCGA_ACMIL_UpdatedOPX.py --train_cohort TCGA_PRAD --SELECTED_MUTATION AR

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--SELECTED_FOLD', default=2, type=int, help='select fold')
parser.add_argument('--loss_method', default='', type=str, help='ATTLOSS or ''')
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--feature_extraction_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--learning_method', default='acmil', type=str, help=': e.g., acmil, abmil')
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--out_folder', default= 'pred_out_040725', type=str, help='out folder name')
parser.add_argument('--train_cohort', default= 'OPX', type=str, help='TCGA_PRAD or OPX')
parser.add_argument('--SELECTED_MUTATION', default='MT', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')





# ===============================================================
#     Model Para
# ===============================================================
parser.add_argument('--BATCH_SIZE', default=1, type=int, help='batch size')
#parser.add_argument('--DROPOUT', default=0, type=int, help='drop out rate')
parser.add_argument('--DIM_OUT', default=128, type=int, help='')
parser.add_argument('--arch', default='ga_mt', type=str, help='e.g., ga_mt, or ga')
parser.add_argument('--use_sep_cri', action= 'store_true', default=False, help='use seperate focal parameters for each mutation')


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    ####################################
    ######      USERINPUT       ########
    ####################################
    ALL_LABEL = ["AR","HR","PTEN","RB1","TP53","TMB","MSI_POS"]
    SELECTED_FEATURE = get_feature_idexes(args.feature_extraction_method, include_tumor_fraction = False)
    N_FEATURE = len(SELECTED_FEATURE)
            
    # ===============================================================
    #     Get config
    # ===============================================================
    config_dir = "myconf.yml"
    with open(config_dir, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        conf = Struct(**c)
    conf.train_epoch = 100
    conf.D_feat = N_FEATURE
    conf.D_inner = args.DIM_OUT
    conf.n_class = 1
    conf.wandb_mode = 'disabled'
    if args.SELECTED_MUTATION == 'MT':
        SELECTED_LABEL = ALL_LABEL
        conf.n_task = 7
        select_label_index = 'ALL'
    else:
        conf.n_task = 1
        SELECTED_LABEL = [args.SELECTED_MUTATION]
        select_label_index = ALL_LABEL.index(SELECTED_LABEL[0])
    conf.lr = 0.001
    
    if args.learning_method == 'abmil':
        conf.n_token = 1
        conf.mask_drop = 0
        conf.n_masked_patch = 0
    elif args.learning_method == 'acmil':
        conf.n_token = 3
        conf.mask_drop = 0.6
        conf.n_masked_patch = 0
        
    # Print all key-value pairs in the conf object
    for key, value in conf.__dict__.items():
        print(f"{key}: {value}")
        
    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    folder_name_overlap = "IMSIZE250_OL100"
    folder_name_nonoverlap = "IMSIZE250_OL0"
    feature_path_opx_train =  os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', args.train_cohort, folder_name_overlap, 'feature_' + args.feature_extraction_method, 'TFT' + str(args.TUMOR_FRAC_THRES))
    feature_path_opx_test =  os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', args.train_cohort, folder_name_nonoverlap, 'feature_' + args.feature_extraction_method, 'TFT' + str(args.TUMOR_FRAC_THRES))
    train_val_test_id_path =  os.path.join(proj_dir + 'intermediate_data/3B_Train_TEST_IDS', args.train_cohort ,'TFT' + str(args.TUMOR_FRAC_THRES))
    feature_path_tcga = os.path.join(proj_dir + 'intermediate_data/5_model_ready_data', "TCGA_PRAD", folder_name_overlap, 'feature_' + args.feature_extraction_method, 'TFT' + str(args.TUMOR_FRAC_THRES))
    
    ######################
    #Create output-dir
    ######################
    folder_name1 = args.feature_extraction_method + '/TrainOL100_TestOL0_TFT' + str(args.TUMOR_FRAC_THRES)  + "/"
    outdir0 =  proj_dir + "intermediate_data/" + args.out_folder + "/" + args.train_cohort + "/" + folder_name1 + 'FOLD' + str(args.SELECTED_FOLD) + '/' + args.SELECTED_MUTATION + "/" 
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
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    ################################################
    #     Model ready data 
    ################################################
    data_ol100 = torch.load(os.path.join(feature_path_opx_train, args.train_cohort + '_data.pth'))
    data_ol0  = torch.load(os.path.join(feature_path_opx_test, args.train_cohort + '_data.pth'))

    #Get Train, test, val data
    train_test_val_id_df = pd.read_csv(os.path.join(train_val_test_id_path, "train_test_split.csv"))
    train_test_val_id_df.rename(columns = {'TMB_HIGHorINTERMEDITATE': 'TMB'}, inplace = True)
    (train_data, train_ids), (val_data, val_ids), (test_data, test_ids) = get_train_test_val_data(data_ol100, data_ol0, train_test_val_id_df, args.SELECTED_FOLD, select_label_index)

    #Get postive freqency to choose alpha or gamma
    check = train_test_val_id_df.loc[train_test_val_id_df['FOLD' + str(0)] == 'TRAIN']
    #check = train_test_val_id_df.loc[train_test_val_id_df['TRAIN_OR_TEST'] == 'TRAIN']

    # print(check['AR'].value_counts()/160) 
    # print(check['HR'].value_counts()/160) 
    # print(check['PTEN'].value_counts()/160) 
    # print(check['RB1'].value_counts()/160) 
    # print(check['TP53'].value_counts()/160) 
    # print(check['TMB'].value_counts()/160) 
    # print(check['MSI_POS'].value_counts()/160) 
    
    #External validation
    tcga_data = torch.load(os.path.join(feature_path_tcga, 'TCGA_PRAD' + '_data.pth'))
    tcga_data, tcga_ids = get_external_validation_data(tcga_data, select_label_index)

    # Define different values for alpha and gamma
    if args.use_sep_cri:
        alpha_values = [0.6, 0.1, 0.1, 0.1, 0.1, 0.75, 0.75]  # Example alpha values
        gamma_values = [10,    2,    2,   2,  2,   15,  15]   # Example gamma values
    else:
        #Best before
        #For RB1: gamma = 2, focal_alpha = 0.1
        #For MSI: gamma = 10, focal_alpha = 0.6
        # focal_gamma = 2   #harder eaxmaple
        # focal_alpha = 0.8 #postive ratio
        gamma_list = [8] #,12,13,14,15,16,17,18,19,20,25,30,40,50]
        alpha_list = [0.9]
        #gamma_list = [5] #,12,13,14,15,16,17,18,19,20,25,30,40,50]
        #alpha_list = [0.5]
        # 'GAMMA_5_ALPHA_0.5'
        # 'GAMMA_8_ALPHA_0.9'
    for focal_gamma in gamma_list:
        for focal_alpha in alpha_list:
            outdir11 = outdir1 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir22 = outdir2 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir33 = outdir3 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir44 = outdir4 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir55 = outdir5 + 'GAMMA_' + str(focal_gamma) + '_ALPHA_' + str(focal_alpha) + '/'
            outdir_list = [outdir11,outdir22,outdir33,outdir44,outdir55]
            for out_path in outdir_list:
                create_dir_if_not_exists(out_path)
        
            ####################################################
            #Dataloader for training
            ####################################################
            train_loader = DataLoader(dataset=train_data, batch_size=args.BATCH_SIZE, shuffle=False)
            test_loader  = DataLoader(dataset=test_data, batch_size=args.BATCH_SIZE, shuffle=False)
            val_loader   = DataLoader(dataset=val_data, batch_size=args.BATCH_SIZE, shuffle=False)
            tcga_loader  = DataLoader(dataset=tcga_data, batch_size=args.BATCH_SIZE, shuffle=False)
            
            
            ###################################################
            #           Test 
            ###################################################   
            # define network
            if args.arch == 'ga':
                model2 = ACMIL_GA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop)
            elif args.arch == 'ga_mt':
                model2 = ACMIL_GA_MultiTask(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop= conf.mask_drop, n_task = conf.n_task)
            else:
                model2 = ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop)
            model2.to(device)
            
            # Load the checkpoint
            #checkpoint = torch.load(ckpt_dir + 'checkpoint-best.pth')
            checkpoint = torch.load(outdir11 + 'checkpoint_epoch99.pth')
            
            # Load the state_dict into the model
            model2.load_state_dict(checkpoint['model'])
        
            y_pred_tasks_test, y_predprob_task_test, y_true_task_test = predict_v2(model2, test_loader, device, conf, 'Test')
            y_pred_tasks_val,  y_predprob_task_val, y_true_task_val = predict_v2(model2, val_loader, device, conf, 'Test')
            
            # from sklearn.linear_model import LogisticRegression
            # import numpy as np
        
            # # Step 1: Fit Platt scaling (logistic regression) on validation set
            # platt_model = LogisticRegression()
            # platt_model.fit(np.array(y_predprob_task_test[i]).reshape(-1, 1), np.array(y_true_task_test[i]))
            # calibrated_probs = platt_model.predict_proba(np.array(y_predprob_task_test[i]).reshape(-1, 1))[:, 1]
        
            
            pred_df_list = []
            perf_df_list = []
            for i in range(conf.n_task):
                #Calibration
                prob_calibrated = calibrate_probs_isotonic(y_true_task_val[i], y_predprob_task_val[i], y_predprob_task_test[i])
                pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], test_ids, SELECTED_LABEL[i], THRES = np.quantile(y_predprob_task_test[i],0.5))
                pred_df_list.append(pred_df)
                perf_df_list.append(perf_df)
            
            all_perd_df = pd.concat(pred_df_list)
            all_perf_df = pd.concat(perf_df_list)
            print(all_perf_df)
            
            all_perd_df.to_csv(outdir44 + "/n_token" + str(conf.n_token) + "_TEST_pred_df.csv",index = False)
            all_perf_df.to_csv(outdir55 + "/n_token" + str(conf.n_token) + "_TEST_perf.csv",index = True)
            print(round(all_perf_df['AUC'].mean(),2))
        

            ###TODO
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
            wsi_path = proj_dir + '/data/OPX/'
            
            selected_ids = test_ids
            s_label = SELECTED_LABEL[6]
            moran_list = []
            for pt in selected_ids:
                i =  opx_ids_ol0.index(pt)
                print(pt)
            
                save_location =   outdir4 + s_label +"/"  + pt + "/"
                create_dir_if_not_exists(save_location)
                #TODO                
                
                _file = wsi_path + pt + ".tif"
                oslide = openslide.OpenSlide(_file)
                save_name = str(Path(os.path.basename(_file)).with_suffix(''))
            
            
                first_batch = opx_data_ol0[i]
                feat = first_batch[0].unsqueeze(0).to(device)
                sub_preds, slide_preds, attn = model2(feat)
                label_index = ALL_LABEL.index(s_label)
            
                #Get attention
                cur_att = attn[label_index] #att no softmax 
                #cur_att_softmax = torch.softmax(cur_att, dim=-1) #att softmax over tiles
            
                #Mean
                cur_pt_att = cur_att.mean(dim = 1).squeeze().cpu().detach().numpy() #Mean aross channels without softmax
                #cur_pt_att = cur_att_softmax.mean(dim = 1).squeeze().cpu().detach().numpy()  #Mean aross channels with softmax
                
                #cur_pt_att = cur_att[0,branches,:].cpu().detach().numpy() #branch
                
                #Get all tile info include noncancer tile
                alltileinfo_dir = proj_dir + 'intermediate_data/3A_otherinfo/OPX/' + "IMSIZE250_OL0" + "/"
                tile_info_df = pd.read_csv(alltileinfo_dir + "all_tile_info.csv")
                cur_tile_info_df = tile_info_df.loc[tile_info_df['SAMPLE_ID'] == pt].copy()
                #cur_tile_info_df.drop(columns = ['AR', 'HR', 'PTEN', 'RB1', 'TP53','TMB_HIGHorINTERMEDITATE', 'MSI_POS'], inplace = True)
                
                
                #Comb att and tile info
                cur_pt_info = first_batch[3]
                cur_att_df = get_attention_and_tileinfo(cur_pt_info, cur_pt_att)
                #cur_att_df.loc[pd.isna(cur_att_df['ATT']),'ATT'] = 0.0001
                
                #Combime att with all tile info
                cur_comb_df = cur_tile_info_df.merge(cur_att_df, how = 'left', 
                                                     on = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP',
                                                           'LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0',
                                                           'WHITE_SPACE','TISSUE_COVERAGE','pred_map_location','PATIENT_ID','SITE_LOCAL'])
                
                cur_comb_df.loc[pd.isna(cur_comb_df['ATT']),'ATT'] = 0
                cur_comb_df_att_coor_df = cur_comb_df[['TILE_XY_INDEXES','ATT', 'AR', 'HR', 'PTEN', 'RB1', 'TP53','TMB_HIGHorINTERMEDITATE', 'MSI_POS']].copy()
                # Step 1: Convert TILE_XY_INDEXES strings to tuples
                cur_comb_df_att_coor_df['TILE_XY_INDEXES'] = cur_comb_df_att_coor_df['TILE_XY_INDEXES'].apply(eval)  # or use ast.literal_eval for safety
                cur_comb_df_att_coor_df[['tile_x', 'tile_y']] = pd.DataFrame(cur_comb_df_att_coor_df['TILE_XY_INDEXES'].tolist(), index=cur_comb_df_att_coor_df.index)
                # Step 3: Normalize grid to (0, 0)
                cur_comb_df_att_coor_df['row'] = cur_comb_df_att_coor_df['tile_y'] - cur_comb_df_att_coor_df['tile_y'].min()
                cur_comb_df_att_coor_df['col'] = cur_comb_df_att_coor_df['tile_x'] - cur_comb_df_att_coor_df['tile_x'].min()
                
                n_rows = cur_comb_df_att_coor_df['row'].max() + 1
                n_cols = cur_comb_df_att_coor_df['col'].max() + 1
                
                # Step 4: Build 2D matrix of ATT
                grid = np.full((n_rows, n_cols), np.nan)
                for _, row in cur_comb_df_att_coor_df.iterrows():
                    grid[int(row['row']), int(row['col'])] = row['ATT']
                values = np.nan_to_num(grid, nan=0.0)
                flat_values = values.flatten()    
                # Create a spatial weights matrix for the 2D grid
                w = ps.weights.lat2W(values.shape[0], values.shape[1])
                # Calculate Moran's I
                moran = Moran(flat_values, w)
                print("Moran's I:", moran.I)
                print("p-value:", moran.p_sim)
                cur_moran_df = pd.DataFrame({'moranI': moran.I, 'SAMPLE_ID': pt,
                                             'AR': cur_comb_df_att_coor_df['AR'].unique().item(),
                                             'HR': cur_comb_df_att_coor_df['HR'].unique().item(),
                                             'PTEN': cur_comb_df_att_coor_df['PTEN'].unique().item(),
                                             'RB1': cur_comb_df_att_coor_df['RB1'].unique().item(),
                                             'TP53': cur_comb_df_att_coor_df['TP53'].unique().item(),
                                             'TMB_HIGHorINTERMEDITATE': cur_comb_df_att_coor_df['TMB_HIGHorINTERMEDITATE'].unique().item(),
                                             'MSI_POS': cur_comb_df_att_coor_df['MSI_POS'].unique().item()}, index = [0])
                moran_list.append(cur_moran_df)
                
            
                #Generate tiles
                tiles, tile_lvls, physSize, base_mag = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)
                
                #get level 0 size in px
                l0_w = oslide.level_dimensions[0][0]
                l0_h = oslide.level_dimensions[0][1]
                
                #1.25x tissue detection for mask
                from Utils import get_downsample_factor, get_image_at_target_mag
                from Utils import do_mask_original,check_tissue,whitespace_check
                import cv2
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
                plt.show()
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
    



            moran_df = pd.concat(moran_list)
            moran_df.to_csv(outdir44 + "/n_token" + str(conf.n_token) + "_TEST_moranI.csv",index = True)

            import pandas as pd
            import matplotlib.pyplot as plt
            %matplotlib inline
            
            # Example: assuming your DataFrame is named df
            # df = pd.read_csv("your_data.csv")
            
            # List of gene columns (update as needed)
            genes = ['AR', 'HR', 'PTEN', 'RB1', 'TP53', 'TMB_HIGHorINTERMEDITATE', 'MSI_POS']
            
            # Boxplot data setup
            data_0 = []  # Y = 0
            data_1 = []  # Y = 1
            
            for gene in genes:
                data_0.append(moran_df[moran_df[gene] == 0]['moranI'])
                data_1.append(moran_df[moran_df[gene] == 1]['moranI'])
            
            # Set up positions for side-by-side boxplots
            positions_0 = [i * 2.0 - 0.3 for i in range(len(genes))]
            positions_1 = [i * 2.0 + 0.3 for i in range(len(genes))]
            
            plt.figure(figsize=(12, 6))
            
            # Boxplots for Y=0 (left)
            bp0 = plt.boxplot(data_0, positions=positions_0, widths=0.5, patch_artist=True)
            # Boxplots for Y=1 (right)
            bp1 = plt.boxplot(data_1, positions=positions_1, widths=0.5, patch_artist=True)
            
            # Color the boxes
            for box in bp0['boxes']:
                box.set(facecolor='skyblue')
            for box in bp1['boxes']:
                box.set(facecolor='lightcoral')
            
            # X-axis setup
            mid_positions = [(x + y) / 2 for x, y in zip(positions_0, positions_1)]
            plt.xticks(mid_positions, genes, rotation=45)
            
            plt.title("Moran's I by Gene Alteration Status")
            plt.ylabel("Moran's I")
            plt.legend([bp0["boxes"][0], bp1["boxes"][0]], ["Y=0", "Y=1"])
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
  
            # ##############################################################################################################################
            # # TCGA
            # ##############################################################################################################################
            # y_pred_tasks_test, y_predprob_task_test, y_true_task_test = predict_v2(model2, tcga_loader, device, conf, 'TCGA')


            # pred_df_list = []
            # perf_df_list = []
            # for i in range(conf.n_task):
            #     if i != 5 :
            #         pred_df, perf_df = get_performance(y_predprob_task_test[i], y_true_task_test[i], tcga_ids, SELECTED_LABEL[i], 
            #                                            THRES = np.quantile(y_predprob_task_test[i], 0.5))
            #         pred_df_list.append(pred_df)
            #         perf_df_list.append(perf_df)
            # all_perd_df = pd.concat(pred_df_list)
            # all_perf_df = pd.concat(perf_df_list)
            # print(all_perf_df)
            # all_perd_df.to_csv(outdir44 + "/n_token" + str(conf.n_token) + "_TCGA_pred_df.csv",index = False)
            # all_perf_df.to_csv(outdir55 + "/n_token" + str(conf.n_token) + "_TCGA_perf.csv",index = True)
            # print(round(all_perf_df['AUC'].mean(),2))
            
            



            
