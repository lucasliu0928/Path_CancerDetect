#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import pandas as pd
import warnings
import glob
sys.path.insert(0, '../Utils/')
from Utils import slide_ROIS
from misc_utils import create_dir_if_not_exists, get_ids
from Utils import generating_tiles, generating_tiles_tma
warnings.filterwarnings("ignore")


#RUN
#source ~/.bashrc
#conda activate paimg9
#python3 -u 1_extract_patches_fixed-res.py  --cohort_name PrECOG --pixel_overlap 0
############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Extract patches")
parser.add_argument('--mag_extract', default='20', type=int, help='specify magnification, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--save_image_size', default='250', type=int, help='the size of extracted tiles, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--pixel_overlap', default='0', type=int, help='specify the level of pixel overlap in your saved tiles, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--mag_target_tiss', default='1.25', type=float, help='magnification for tissue detection: e.g., 1.25x')
parser.add_argument('--cohort_name', default='Pluvicto_Pretreatment_bx', type=str, help='Cohort name: OPX, TCGA_PRAD, Neptune, TAN_TMA_Cores,Pluvicto_TMA_Cores, Pluvicto_Pretreatment_bx, PrECOG, "CCola/all_slides/"')
parser.add_argument('--out_dir', default='1_tile_pulling', type=str, help='output directory')

if __name__ == '__main__':
    
    args = parser.parse_args()

    ############################################################################################################
    #USER INPUT 
    ############################################################################################################
    limit_bounds = True     # this is weird, dont change it
    folder_name = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.pixel_overlap)


    #DIR
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    wsi_location = os.path.join(proj_dir, "data", args.cohort_name)
    out_location = os.path.join(proj_dir,'intermediate_data', args.out_dir, args.cohort_name, folder_name)  #1_feature_extraction, cancer_prediction_results110224

    #Create output dir
    create_dir_if_not_exists(out_location)
    
    ############################################################################################################
    #All available IDs
    #Ccola: 234
    #OPX: 360
    #TCGA: 449
    #Neptune: 350
    #TAN_TMA: 677
    #pluvicto: 606
    #PrECOG: 46
    #Pluvicto_Pretreatment_bx: 27
    ############################################################################################################    
    if args.cohort_name == "CCola/all_slides/":
        all_ids = get_ids(wsi_location, include="(2017-0133)")  # 234
    else:
        all_ids = get_ids(wsi_location)
    print(args.cohort_name, ": N of IDs = " ,len(all_ids))
    
    
    ############################################################################################################
    #ID to Exclude
    ############################################################################################################
    #Get IDs that are in FT train or already processed to exclude 
    fine_tune_ids_df = pd.read_csv(proj_dir + 'intermediate_data/0_cd_finetune/cancer_detection_training/all_tumor_fraction_info.csv')
    ft_train_ids = list(fine_tune_ids_df.loc[fine_tune_ids_df['Train_OR_Test'] == 'Train','sample_id'])
    toexclude_ids = ft_train_ids + ['cca3af0c-3e0e-4cfb-bb07-459c979a0bd5'] #The latter one is TCGA issue file
    
    ############################################################################################################
    #Select ID
    ############################################################################################################
    #Exclude ids in ft_train
    selected_ids = [x for x in all_ids if x not in toexclude_ids]
    selected_ids.sort()
    
    ############################################################################################################
    #Start 
    ############################################################################################################
    for cur_id in selected_ids:     
        #create out path
        save_location = out_location + "/" + cur_id + "/"
        create_dir_if_not_exists(save_location)
    
        #check if processed:
        imgout = glob.glob(save_location + "*.png")
        if len(imgout) > 0:
             print(cur_id + ': already processed')
        elif len(imgout) == 0:
            slides_name = cur_id
            if 'OPX' in args.cohort_name:
                _file =  os.path.join(wsi_location, slides_name + ".tif") 
                rad_tissue = 5
            elif 'CCola' in args.cohort_name:
                _file = os.path.join(wsi_location, slides_name + ".svs") 
                rad_tissue = 2
            elif 'TAN_TMA_Cores' in args.cohort_name:
                _file = os.path.join(wsi_location, slides_name + ".tif") 
                rad_tissue = 2
            elif 'Pluvicto_TMA_Cores' in args.cohort_name:
                _file = os.path.join(wsi_location, slides_name + ".tif") 
                rad_tissue = 2
            elif 'PrECOG' in args.cohort_name:
                _file = os.path.join(wsi_location, slides_name + ".svs") 
                rad_tissue = 2
            elif 'Neptune' in args.cohort_name:
                _file = os.path.join(wsi_location, slides_name + ".tif") 
                if cur_id == 'NEP-081PS2-1_HE_MH_03282024' or cur_id == 'NEP-123PS1-1_HE_MH06032024':
                    rad_tissue = 2
                else:
                    rad_tissue = 5
            elif 'TCGA' in args.cohort_name:
                slides_name = [f for f in os.listdir(os.path.join(wsi_location, cur_id)) if '.svs' in f][0].replace('.svs','')
                _file = os.path.join(wsi_location, cur_id, slides_name + '.svs') 
                rad_tissue = 2
            elif 'Pluvicto_Pretreatment_bx' in args.cohort_name:
                _file =  os.path.join(wsi_location, slides_name + ".tif") 
                rad_tissue = 5
            
            #Generating tiles 
            if args.cohort_name in ["OPX", "CCola/all_slides/", "Neptune", "PrECOG", "Pluvicto_Pretreatment_bx"]:                
                mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles(cur_id, _file, args.save_image_size, args.pixel_overlap, limit_bounds, args.mag_target_tiss, rad_tissue, args.mag_extract)
            elif args.cohort_name in ["TAN_TMA_Cores", "Pluvicto_TMA_Cores"]:
                mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles_tma(cur_id, _file, args.save_image_size, args.pixel_overlap, rad_tissue)
            elif  args.cohort_name in ['TCGA_PRAD']:
                mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles(cur_id, _file, args.save_image_size, args.pixel_overlap, limit_bounds, args.mag_target_tiss, rad_tissue, args.mag_extract)
                tile_info_df['SAMPLE_ID'] = slides_name
                
            tile_info_df.to_csv(os.path.join(save_location, slides_name + "_tiles.csv"), index = False)
            lvl_img.save(os.path.join(save_location,  slides_name + '_low-res.png'))
            lvl_mask.save(os.path.join(save_location, slides_name + '_tissue.png'))
            slide_ROIS(polygons=tissue, mpp=float(mpp),savename=os.path.join(save_location, slides_name + '_tissue.json'),labels='tissue', ref=[0, 0], roi_color=-16770432)