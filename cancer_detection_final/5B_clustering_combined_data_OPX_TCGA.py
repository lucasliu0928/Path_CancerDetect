#!/usr/bin/env python
# coding: utf-8

#NOTE: use python env acmil in ACMIL folder
import sys
import os
import matplotlib
#%matplotlib inline
matplotlib.use('Agg')
import pandas as pd
import warnings
import torch
import time
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists
from cluster_utils import run_clustering_on_training, get_cluster_labels, get_label_feature_info_comb
warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse
from train_utils import get_feature_idexes, get_list_for_modelreadydata
from train_utils import plot_cluster_image, ModelReadyData_clustering
import joblib



############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Cluster data extraction")
parser.add_argument('--train_overlap', default=0, type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--test_overlap', default=0, type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--save_image_size', default=250, type=int, help='the size of extracted tiles')
parser.add_argument('--train_cohort', default='TCGA_PRAD', type=str, help='data set name: TAN_TMA_Cores or OPX or TCGA_PRAD')
parser.add_argument('--valid_cohort', default='OPX', type=str, help='data set name: TAN_TMA_Cores or OPX or TCGA_PRAD')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold') #NOte: this does not filter out any tiles, only for selection of folders



# ===============================================================
#     Model Para
# ===============================================================
if __name__ == '__main__':
    
    args = parser.parse_args()
    
    ####################################
    #Select label and feature index
    ####################################
    SELECTED_LABEL = ['AR', 'HR', 'PTEN', 'RB1', 'TP53', 'TMB_HIGHorINTERMEDITATE', 'MSI_POS']
    SELECTED_FEATURE = get_feature_idexes(args.fe_method,include_tumor_fraction = False)

    
    ##################
    ###### DIR  ######
    ##################
    folder_name_train = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.train_overlap)
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    info_path_train =   os.path.join(proj_dir,'intermediate_data','3A_otherinfo', args.train_cohort, folder_name_train)
    feature_path_train = os.path.join(proj_dir,'intermediate_data','4_tile_feature', args.train_cohort, folder_name_train)
    
    folder_name_test = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.test_overlap)
    info_path_test =   os.path.join(proj_dir,'intermediate_data','3A_otherinfo', args.train_cohort, folder_name_test)
    feature_path_test = os.path.join(proj_dir,'intermediate_data','4_tile_feature', args.train_cohort, folder_name_test)
    
    info_path_val = os.path.join(proj_dir,'intermediate_data','3A_otherinfo', args.valid_cohort, folder_name_test)
    feature_path_val = os.path.join(proj_dir,'intermediate_data','4_tile_feature', args.valid_cohort, folder_name_test)


    train_val_test_id_path =  os.path.join(proj_dir + 'intermediate_data/3B_Train_TEST_IDS', 
                                           args.train_cohort ,
                                           'TFT' + str(args.tumor_frac))
    

    
    ################################################
    #Create output dir
    ################################################
    outdir =  os.path.join(proj_dir + 'intermediate_data/5_combined_data_cluster',
                           args.train_cohort, 
                           "TRAIN_OL" + str(args.train_overlap) + "_TEST_OL" + str(args.test_overlap),
                           'feature_' + args.fe_method,
                           'TFT' + str(args.tumor_frac))
    create_dir_if_not_exists(outdir)


    ##################
    #Select GPU
    ##################
    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    print(device)
    
    ############################################################################################################
    #Get Train, test, val IDs
    ############################################################################################################      
    id_col = "PATIENT_ID"
    train_test_val_id_df = pd.read_csv(os.path.join(train_val_test_id_path, "train_test_split.csv"))
    train_ids = list(train_test_val_id_df.loc[train_test_val_id_df['TRAIN_OR_TEST'] == 'TRAIN', id_col].unique())
    test_ids = list(train_test_val_id_df.loc[train_test_val_id_df['TRAIN_OR_TEST'] == 'TEST', id_col].unique())

    
    ############################################################################################################
    #Load all tile info df
    #This file contains all tiles before cancer fraction exclusion 
    #and  has tissue membership > 0.9, white space < 0.9 (non white space > 0.1)
    ############################################################################################################
    all_tile_info_df_train = pd.read_csv(os.path.join(info_path_train, "all_tile_info.csv"))
    all_tile_info_df_train = all_tile_info_df_train.loc[all_tile_info_df_train[id_col].isin(train_ids)]
    
    all_tile_info_df_test = pd.read_csv(os.path.join(info_path_test, "all_tile_info.csv")) #5958125
    all_tile_info_df_test = all_tile_info_df_test.loc[all_tile_info_df_test[id_col].isin(test_ids)]
        
    all_tile_info_df_val = pd.read_csv(os.path.join(info_path_val, "all_tile_info.csv"))
    

    

    ############################################################################################################
    #Get label feature and info comb df
    #Get all tile , do not exclude <0.9 tumor fraction
    ############################################################################################################
    if args.train_cohort == "TCGA_PRAD":
        folder_col = "TCGA_FOLDER_ID"
    
    if args.valid_cohort == "OPX":
        folder_col2 = "SAMPLE_ID"
    
    #update ID  for folder IDs = image IDs
    train_ids = list(all_tile_info_df_train[folder_col].unique()) #326
    test_ids = list(all_tile_info_df_test[folder_col].unique()) #120
    val_ids = list(all_tile_info_df_val[folder_col2].unique()) #268
    
    comb_df_train = get_label_feature_info_comb(train_ids,all_tile_info_df_train, feature_path_train, args.fe_method, folder_col)
    comb_df_test = get_label_feature_info_comb(test_ids,all_tile_info_df_test, feature_path_test, args.fe_method,folder_col)
    comb_df_val = get_label_feature_info_comb(val_ids,all_tile_info_df_val, feature_path_val, args.fe_method, folder_col2)

    ############################################################################################################
    #Clustering
    ############################################################################################################
    #Run K mean on Train
    comb_df_train_cancer = comb_df_train.loc[comb_df_train['TUMOR_PIXEL_PERC'] > 0.9].copy() #2200337
    start_time = time.time()
    model, scaler, k = run_clustering_on_training(comb_df_train_cancer, SELECTED_FEATURE, outdir, method = 'kmean', 
                                          rs = 42, use_bestk = False, use_pca = True, k = 10)
    joblib.dump(scaler, outdir + '/scaler.pkl')
    joblib.dump(model, outdir + '/kmean_model_10.pkl')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    
    #Get cluster labels
    comb_df_train['CLUSTER_10'] = get_cluster_labels(comb_df_train,SELECTED_FEATURE, model,scaler, use_pca = True, rs = 42)
    comb_df_test['CLUSTER_10'] = get_cluster_labels(comb_df_test,SELECTED_FEATURE, model,scaler, use_pca = True, rs = 42)
    comb_df_val['CLUSTER_10'] = get_cluster_labels(comb_df_val,SELECTED_FEATURE, model,scaler, use_pca = True, rs = 42)
    
    
    #Run K mean on Train
    comb_df_train_cancer = comb_df_train.loc[comb_df_train['TUMOR_PIXEL_PERC'] > 0.9].copy() #2200337
    start_time = time.time()
    model, _, k = run_clustering_on_training(comb_df_train_cancer, SELECTED_FEATURE, outdir, method = 'kmean', 
                                          rs = 42, use_bestk = False, use_pca = True, k = 5)
    joblib.dump(model, outdir + '/kmean_model_5.pkl')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    
    #Cluster 5
    comb_df_train['CLUSTER_5'] = get_cluster_labels(comb_df_train,SELECTED_FEATURE, model,scaler, use_pca = True, rs = 42)
    comb_df_test['CLUSTER_5'] = get_cluster_labels(comb_df_test,SELECTED_FEATURE, model,scaler, use_pca = True, rs = 42)
    comb_df_val['CLUSTER_5'] = get_cluster_labels(comb_df_val,SELECTED_FEATURE, model,scaler, use_pca = True, rs = 42)
    
    ############################################################################################################
    #This section for mekias input, for all TCGA cases including train and test
    ############################################################################################################
    #Combine all TCGA data
    comb_df_TCGA = pd.concat([comb_df_train,comb_df_test], axis = 0)
    kept_cols1 = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP',
                  'LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE',
                  'TISSUE_COVERAGE', 'pred_map_location', 'TUMOR_PIXEL_PERC',
                   'TCGA_FOLDER_ID', 'PATIENT_ID', 'SLIDE_ID', 'SITE_LOCAL','CLUSTER_10',"CLUSTER_5"]
    comb_df_TCGA  = comb_df_TCGA[kept_cols1]
    comb_df_TCGA.loc[comb_df_TCGA['TUMOR_PIXEL_PERC'] <= 0.9,"CLUSTER_5"] = -1 #For non-cancer assign -1
    comb_df_TCGA.loc[comb_df_TCGA['TUMOR_PIXEL_PERC'] <= 0.9,"CLUSTER_10"] = -1 #For non-cancer assign -1
    comb_df_TCGA.to_csv(os.path.join(outdir, args.train_cohort + '_OL0_cluster_data.csv'), index = False)
    

    #TODO TO CHECK
    # ############################################################################################################
    # #Get image cluster data
    # ############################################################################################################
    # #update ID for folder ID if TCGA
    # train_ids = list(all_tile_info_df_train['TCGA_FOLDER_ID'].unique())
    # test_ids = list(all_tile_info_df_test['TCGA_FOLDER_ID'].unique())
    # val_ids = list(all_tile_info_df_val['SAMPLE_ID'].unique())
    
    # #Train data
    # matrix_list, label_list, sp_id_list, pt_id_list = get_list_for_modelreadydata(comb_df_train, train_ids, SELECTED_LABEL, args.tumor_frac)
    # train_data_cluster = ModelReadyData_clustering(matrix_list, label_list, sp_id_list, pt_id_list)
    # torch.save(train_data_cluster, os.path.join(outdir, args.train_cohort + '_train_data.pth'))

    # #Test data
    # matrix_list, label_list, sp_id_list, pt_id_list = get_list_for_modelreadydata(comb_df_test, test_ids, SELECTED_LABEL, args.tumor_frac)
    # test_data_cluster = ModelReadyData_clustering(matrix_list, label_list, sp_id_list, pt_id_list)
    # torch.save(test_data_cluster, os.path.join(outdir, args.train_cohort + '_test_data.pth'))
    
    # #val
    # matrix_list, label_list, sp_id_list, pt_id_list = get_list_for_modelreadydata(comb_df_val, list(comb_df_val['SAMPLE_ID'].unique()), SELECTED_LABEL, args.tumor_frac)
    # val_data_cluster = ModelReadyData_clustering(matrix_list, label_list, sp_id_list, pt_id_list)
    # torch.save(val_data_cluster, os.path.join(outdir, args.valid_cohort + '_val_data.pth'))

    # #Plot cluster image
   
    # plot_cluster_image(matrix_list[0])
    




    

    
        




    
