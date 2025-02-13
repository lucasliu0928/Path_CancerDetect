#!/usr/bin/env python
# coding: utf-8
#NOTE: use paimg9 env

import sys
import os
import numpy as np
import openslide
from fastai.vision.all import *
matplotlib.use('Agg')
import pandas as pd
import warnings
sys.path.insert(0, '../Utils/')
from Preprocessing import preprocess_mutation_data, preprocess_site_data, get_tile_representation, get_tile_representation_tma
from Utils import generate_deepzoom_tiles
from Utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")

import ResNet as ResNet
import time
import PIL 
import timm
import argparse

#Run: python3 -u 3B_get_feature.py --select_idx_start 80 --select_idx_end 120 --cuda_device 'cuda' --pixel_overlap 100 --save_image_size 250 --cohort_name OPX --feature_extraction_method uni1 


############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--pixel_overlap', default='0', type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--save_image_size', default='250', type=int, help='the size of extracted tiles')
parser.add_argument('--cohort_name', default='OPX', type=str, help='data set name: TAN_TMA_Cores or OPX')
parser.add_argument('--feature_extraction_method', default='uni1', type=str, help='feature extraction model: retccl, uni1, uni2')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--select_idx_start', type=int)
parser.add_argument('--select_idx_end', type=int)

if __name__ == '__main__':
    
    args = parser.parse_args()
    ############################################################################################################
    #USER INPUT 
    ############################################################################################################
    pixel_overlap = args.pixel_overlap      
    save_image_size = args.save_image_size   
    cohort_name = args.cohort_name   
    feature_extraction_method = args.feature_extraction_method

    folder_name = cohort_name + "/" + "IMSIZE" + str(save_image_size) + "_OL" + str(pixel_overlap) + "/" 
    if feature_extraction_method == 'retccl':
        resize_transform = False
    else:
        resize_transform = True
        
    ############################################################################################################
    #DIR
    ############################################################################################################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    model_path = proj_dir + 'models/feature_extraction_models/' + feature_extraction_method + '/'

    wsi_location_opx = proj_dir + '/data/OPX/'
    wsi_location_tan = proj_dir + 'data/TAN_TMA_Cores/'
    wsi_location_ccola = proj_dir + '/data/CCola/all_slides/'
    tile_info_path = proj_dir + 'intermediate_data/3_updated_tile_info/'+ folder_name

    out_location = proj_dir + 'intermediate_data/4_tile_feature/'+ folder_name
    create_dir_if_not_exists(out_location)

    ############################################################################################################
    #Select GPU
    ############################################################################################################
    device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')
    print(device)

    ############################################################################################################
    #Select IDS
    ############################################################################################################
    #Get IDs that are in FT train or already processed to exclude 
    fine_tune_ids_df = pd.read_csv(proj_dir + 'intermediate_data/0_cd_finetune/cancer_detection_training/all_tumor_fraction_info.csv')
    ft_train_ids = list(fine_tune_ids_df.loc[fine_tune_ids_df['Train_OR_Test'] == 'Train','sample_id']) #24, 7 from OPX, 17 from ccola
    toexclude_ids = ft_train_ids 

    #All available IDs
    opx_ids = [x.replace('.tif','') for x in os.listdir(wsi_location_opx)] #217
    ccola_ids = [x.replace('.svs','') for x in os.listdir(wsi_location_ccola) if '(2017-0133)' in x] #234
    tan_ids =  [x.replace('.tif','') for x in os.listdir(wsi_location_tan)] #677

    if cohort_name == "OPX":
        all_ids = opx_ids
    elif cohort_name == "ccola":
        all_ids = ccola_ids
    elif cohort_name == "TAN_TMA_Cores":
        all_ids = tan_ids
    elif cohort_name == "all":
        all_ids = opx_ids + ccola_ids + tan_ids

    #Exclude ids in ft_train or processed
    selected_ids = [x for x in all_ids if x not in toexclude_ids] #209 for 
    selected_ids.sort()


    ################################################################################################
    #Load tile info 
    ################################################################################################
    tile_info_df = pd.read_csv(tile_info_path + "all_tile_info.csv")


    ############################################################################################################
    # Load Pretrained representation model
    ############################################################################################################
    if feature_extraction_method == 'retccl':
        model = ResNet.resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
        pretext_model = torch.load(model_path + 'best_ckpt.pth',map_location=torch.device(device))
        model.fc = nn.Identity()
        model.load_state_dict(pretext_model, strict=True)
    elif feature_extraction_method == 'uni1':
        model = timm.create_model("vit_large_patch16_224",img_size = 224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True) # img_size=224, patch_size=16, 
        model.load_state_dict(torch.load(os.path.join(model_path, "vit_large_patch16_224.dinov2.uni_mass100k.bin"), map_location="cpu"), strict=True)
    elif feature_extraction_method == 'uni2':
        timm_kwargs = {
           'model_name': 'vit_giant_patch14_224',
           'img_size': 224, 
           'patch_size': 14, 
           'depth': 24,
           'num_heads': 24,
           'init_values': 1e-5, 
           'embed_dim': 1536,
           'mlp_ratio': 2.66667*2,
           'num_classes': 0, 
           'no_embed_class': True,
           'mlp_layer': timm.layers.SwiGLUPacked, 
           'act_layer': torch.nn.SiLU, 
           'reg_tokens': 8, 
           'dynamic_img_size': True
          }
        model = timm.create_model(**timm_kwargs)
        model.load_state_dict(torch.load(os.path.join(model_path, "uni2-h.bin"), map_location="cpu"), strict=True)


    ############################################################################################################
    #For each patient tile, get representation
    ############################################################################################################
    ct = 0 
    for cur_id in selected_ids[args.select_idx_start:args.select_idx_end]:
        print(cur_id)
        if ct % 10 == 0: print(ct)

        save_location = out_location + cur_id + '/features/'
        create_dir_if_not_exists(save_location)
        save_name = save_location + 'features_alltiles_' + feature_extraction_method + '.h5'

        if os.path.exists(save_name) == False: #check if processed
            
            if 'OPX' in cur_id:
                _file = wsi_location_opx + cur_id + ".tif"
            elif '(2017-0133)' in cur_id:
                _file = wsi_location_ccola + cur_id + '.svs'
            elif 'TMA' in cur_id:
                _file = wsi_location_tan + cur_id + '.tif'

                    
            if cohort_name == "OPX":
                #Load slide
                oslide = openslide.OpenSlide(_file)

                #Get tile info
                cur_tile_info_df = tile_info_df.loc[tile_info_df['SAMPLE_ID'] == cur_id]
                
                #Generate tiles
                tiles, tile_lvls, physSize, base_mag = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds=True)  # limit_bounds set True always, do not change it
                #Grab tile 
                tile_img = get_tile_representation(cur_tile_info_df, tiles, tile_lvls, model, device, resize_transform)
                

                
            elif cohort_name == "TAN_TMA_Cores":      
                #Load slide
                tma = PIL.Image.open(_file)
                
                #Get tile info
                cur_tile_info_df = tile_info_df.loc[tile_info_df['SAMPLE_ID'] == cur_id]
                
                #Grab tile 
                tile_img = get_tile_representation_tma(cur_tile_info_df,tma, model, device, resize_transform)



            
            #Get feature
            start_time = time.time()
            feature_list = [tile_img[i][1] for i in range(cur_tile_info_df.shape[0])]
            print("--- %s seconds ---" % (time.time() - start_time))
            feature_df = np.concatenate(feature_list)
            feature_df = pd.DataFrame(feature_df)    
            feature_df.to_hdf(save_name, key='feature', mode='w')
            cur_tile_info_df.to_hdf(save_name, key='tile_info', mode='a')

            ct += 1


