#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import os
import torch
import argparse
import time
sys.path.insert(0, '../../Utils/') 
from histoTME_util import create_dir_if_not_exists, save_hdf5
import ast
import numpy as np
from data_loader import get_sample_feature,get_feature_idexes


'''
For get data
# source ~/.bashrc
# conda activate histoTME
python3 0_reformat_data.py --fe_method uni2 --cohort_name Pluvicto_Pretreatment_bx --tumor_frac 0.0
'''

'''
#for infenrece
#cd /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/HistoTME/HistoTME_regression
python3 predict_bulk.py  --cohort OPX --h5_folder /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/model_data/TF0.0/OPX/IMSIZE250_OL0/uni2 --chkpts_dir /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/HistoTME/local_dir/checkpoints  --save_loc /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/TME/TF0.0/ --num_workers 10 --embed uni2 
python3 predict_bulk.py  --cohort TCGA_PRAD --h5_folder /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/model_data/TF0.0/TCGA_PRAD/IMSIZE250_OL0/uni2 --chkpts_dir /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/HistoTME/local_dir/checkpoints  --save_loc /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/TME/TF0.0/ --num_workers 10 --embed uni2 
python3 predict_bulk.py  --cohort Neptune --h5_folder /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/model_data/TF0.0/Neptune/IMSIZE250_OL0/uni2 --chkpts_dir /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/HistoTME/local_dir/checkpoints  --save_loc /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/TME/TF0.0/ --num_workers 10 --embed uni2 
python3 /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/HistoTME/HistoTME_regression/predict_bulk.py  --cohort PrECOG --h5_folder /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/model_data/TF0.0/PrECOG/IMSIZE250_OL0/uni2 --chkpts_dir /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/HistoTME/local_dir/checkpoints  --save_loc /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/TME/TF0.0/ --num_workers 10 --embed uni2 


#Spatial inference
python3 predict_spatial.py  --h5_path /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/model_data/TF0.0/OPX/IMSIZE250_OL0/uni2/OPX_007_features.hdf5 --chkpts_dir /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/HistoTME/local_dir/checkpoints  --save_loc /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/TME_Spatial/TF0.0/ --num_workers 10 --embed uni2 


Note IMPORTANT:
I added the following code to "data.py" in "HistoTME_regression folder" to make it easier to match all embedding model names and the names in the arguments for python predict_spatial.py [-h] [--h5_path H5_PATH] [--chkpts_dir CHKPTS_DIR] [--num_workers NUM_WORKERS]
[--embed EMBED] [--save_loc SAVE_LOC]

elif 'uni1' in embedding_paths[0]:
    embedding_dim = 1024
elif 'uni2' in embedding_paths[0]:
    embedding_dim = 1536
  
  
'''


  


############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("reformat to h5")
parser.add_argument('--save_image_size', default='250', type=int, help='the size of extracted tiles')
parser.add_argument('--pixel_overlap', default='0', type=int, help='specify the level of pixel overlap in your saved tiles, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath, virchow2')
parser.add_argument('--tumor_frac', default= 0.0, type=float, help='tile tumor fraction threshold')
parser.add_argument('--cohort_name', default='PrECOG', type=str, help='data set name: TAN_TMA_Cores, OPX, TCGA_PRAD, Neptune')


            
if __name__ == '__main__':
    
    args = parser.parse_args()


    ##################
    ###### DIR  ######
    ##################
    folder_name = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.pixel_overlap)
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    feature_dir = os.path.join(proj_dir, "intermediate_data", "4_tile_feature")
    cancer_info_dir = os.path.join(proj_dir, "intermediate_data", "2_cancer_detection")
    
    
    out_location = os.path.join(proj_dir,'intermediate_data','0_HistoTME', "model_data" , "TF" + str(args.tumor_frac), args.cohort_name, folder_name,args.fe_method)
    create_dir_if_not_exists(out_location)
    
    ###################################
    #Load
    ###################################
    start_time = time.time()
    
    if args.cohort_name not in ['PrECOG', 'Pluvicto_Pretreatment_bx', 'Pluvicto_TMA_Cores']:
        base_path = os.path.join(data_dir, 
                                 args.cohort_name, 
                                 folder_name, 
                                 f'feature_{args.fe_method}', 
                                 f"TFT{args.tumor_frac}", 
                                 f'{args.cohort_name}_data.pth')
    
        comb_data = torch.load(base_path.format("OL" + str(args.pixel_overlap)),weights_only = False)
    else:
        feature_path = os.path.join(feature_dir, args.cohort_name,folder_name)
        sp_ids = os.listdir(os.path.join(feature_dir, args.cohort_name,folder_name))
        if ".DS_Store" in sp_ids:
            sp_ids.remove(".DS_Store")
        
        comb_data = list()
        for sp in sp_ids:
            feautre_and_tileinfo = get_sample_feature(sp, feature_path, args.fe_method, cancer_info_dir)
            feature_idxes  = get_feature_idexes(args.fe_method, include_tumor_fraction = False)
            
            x = torch.tensor(feautre_and_tileinfo[feature_idxes].to_numpy())
            tile_info = feautre_and_tileinfo[['SAMPLE_ID','MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 
                                              'PIXEL_OVERLAP', 'LIMIT_BOUNDS','TILE_XY_INDEXES', 
                                              'TILE_COOR_ATLV0', 'WHITE_SPACE', 'TISSUE_COVERAGE',
                                              'pred_map_location', 'TUMOR_PIXEL_PERC']]            
            comb_data.append({'x': x, 
                              'tile_info': tile_info,
                              'sample_id': sp})

    elapsed_time = time.time() - start_time
    print(f"Time taken for {args.cohort_name}: {elapsed_time/60:.2f} minutes")
    
    ct = 0
    for d in comb_data: #iter through each WSI
        if ct%50 ==0 : print(ct)
        #Get tile embeddings and tiel info
        tile_embeddings = d['x'].numpy()
        tile_info = d['tile_info']
        s_name = d['sample_id']
        #tiel coords in tuple
        tuples = tile_info["TILE_XY_INDEXES"].apply(ast.literal_eval)  #tile coordiantes row, col
        tile_coord_np = np.stack(tuples.to_numpy()).astype(np.int32)

        # #lvl0 coordinates
        # tile_loc = tile_info["TILE_COOR_ATLV0"].str.split(r'[-_]', expand=True)
        # tile_loc.columns = ['start_x','start_y','rise_x','rise_y']
        # tile_loc = tile_loc.apply(pd.to_numeric)
        # tile_loc['end_x'] = tile_loc['start_x'] + tile_loc['rise_x']
        # tile_loc['end_y'] = tile_loc['start_y'] + tile_loc['rise_y']
        
        attr_dict = {'features': {'wsi_name': s_name , 'label': ''}}
        asset_dict = {'coords': tile_coord_np, 'features': tile_embeddings}
        
        save_hdf5(os.path.join(out_location, s_name + '_features.hdf5'), asset_dict=asset_dict, attr_dict=attr_dict)
        ct = ct + 1
