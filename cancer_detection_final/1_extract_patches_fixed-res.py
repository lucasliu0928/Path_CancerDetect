#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import argparse
import pandas as pd
import warnings
import glob
sys.path.insert(0, '../Utils/')
from Utils import slide_ROIS
from Utils import create_dir_if_not_exists
from Utils import generating_tiles, generating_tiles_tma
warnings.filterwarnings("ignore")



#Run: python3 -u 1_extract_patches_fixed-res.py  --cohort_name TCGA_PRAD --pixel_overlap 0 


############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--mag_extract', default='20', type=int, help='specify magnification, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--save_image_size', default='250', type=int, help='the size of extracted tiles')
parser.add_argument('--pixel_overlap', default='0', type=int, help='specify the level of pixel overlap in your saved tiles, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--mag_target_tiss', default='1.25', type=float, help='magnification for tissue detection: e.g., 1.25x')
parser.add_argument('--cohort_name', default='OPX', type=str, help='data set name: TAN_TMA_Cores, OPX, TCGA_PRAD')


if __name__ == '__main__':
    
    args = parser.parse_args()

    ############################################################################################################
    #USER INPUT 
    ############################################################################################################
    mag_extract = args.mag_extract # do not change this, model trained at 250x250 at 20x
    save_image_size = args.save_image_size  # do not change this, model trained at 250x250 at 20x
    pixel_overlap = args.pixel_overlap  # specify the level of pixel overlap in your saved images
    limit_bounds = True  # this is weird, dont change it
    mag_target_tiss = args.mag_target_tiss   #1.25x for tissue detection
    cohort_name = args.cohort_name
    folder_name = "IMSIZE" + str(save_image_size) + "_OL" + str(pixel_overlap)


    #DIR
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    wsi_location_ccola = proj_dir + '/data/CCola/all_slides/'
    wsi_location_opx = proj_dir + '/data/OPX/' #N= 353, Now OPX has all the old and newly added samples (Oncoplex_deidentified)
    wsi_location_opx2 = os.path.join(proj_dir,'data','Oncoplex_deidentified')
    wsi_location_tan = proj_dir + 'data/TAN_TMA_Cores/'
    wsi_location_tcga = proj_dir + 'data/TCGA_PRAD/'
    out_location = os.path.join(proj_dir,'intermediate_data','1_tile_pulling', cohort_name, folder_name)  #1_feature_extraction, cancer_prediction_results110224
    
    
    #Create output dir
    create_dir_if_not_exists(out_location)



    ############################################################################################################
    #Select IDS
    ############################################################################################################
    #Get IDs that are in FT train or already processed to exclude 
    fine_tune_ids_df = pd.read_csv(proj_dir + 'intermediate_data/0_cd_finetune/cancer_detection_training/all_tumor_fraction_info.csv')
    ft_train_ids = list(fine_tune_ids_df.loc[fine_tune_ids_df['Train_OR_Test'] == 'Train','sample_id'])
    toexclude_ids = ft_train_ids + ['cca3af0c-3e0e-4cfb-bb07-459c979a0bd5'] #The latter one is TCGA issue file
    
    #All available IDs
    opx_ids = [x.replace('.tif','') for x in os.listdir(wsi_location_opx) if x != '.DS_Store'] #217
    opx_ids2 = [x.replace('.tif','') for x in os.listdir(wsi_location_opx2) if x != '.DS_Store'] #289
    ccola_ids = [x.replace('.svs','') for x in os.listdir(wsi_location_ccola) if '(2017-0133)' in x] #234
    tan_ids =  [x.replace('.tif','') for x in os.listdir(wsi_location_tan)  if x != '.DS_Store'] #677
    tcga_ids = [x.replace('.svs','') for x in os.listdir(wsi_location_tcga) if x != '.DS_Store'] #449
    
    if cohort_name == "OPX":
        all_ids = opx_ids 
    elif cohort_name == "ccola":
        all_ids = ccola_ids
    elif cohort_name == "TAN_TMA_Cores":
        all_ids = tan_ids
    elif cohort_name == 'TCGA_PRAD':
        all_ids = tcga_ids
    elif cohort_name == "all":
        all_ids = opx_ids + ccola_ids + tan_ids + tcga_ids
    
    #Exclude ids in ft_train or processed
    selected_ids = [x for x in all_ids if x not in toexclude_ids]
    selected_ids.sort()



    ############################################################################################################
    #Start 
    ############################################################################################################
    for cur_id in selected_ids:
    
        save_location = out_location + "/" + cur_id + "/" 
        create_dir_if_not_exists(save_location)
    
        #check if processed:
        imgout = glob.glob(save_location + "*.png")
        if len(imgout) > 0:
             print(cur_id + ': already processed')
        elif len(imgout) == 0:
            slides_name = cur_id
            if 'OPX' in cur_id:
                _file = wsi_location_opx + slides_name + ".tif"
                rad_tissue = 5
            elif '(2017-0133)' in cur_id:
                _file = wsi_location_ccola + slides_name + '.svs'
                rad_tissue = 2
            elif 'TMA' in cur_id:
                _file = wsi_location_tan + slides_name + '.tif'
                rad_tissue = 2
            else:
                slides_name = [f for f in os.listdir(wsi_location_tcga + cur_id + '/') if '.svs' in f][0].replace('.svs','')
                _file = wsi_location_tcga + cur_id + '/' + slides_name + '.svs'
                rad_tissue = 2
        
            #Generating tiles 
            if 'OPX' in cur_id or '(2017-0133)' in cur_id:
                mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles(cur_id, _file, save_image_size, pixel_overlap, limit_bounds, mag_target_tiss, rad_tissue, mag_extract)
                
            elif 'TMA' in cur_id:
                mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles_tma(cur_id, _file, save_image_size, pixel_overlap, rad_tissue)
            else:
                mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles(cur_id, _file, save_image_size, pixel_overlap, limit_bounds, mag_target_tiss, rad_tissue, mag_extract)
                tile_info_df['SAMPLE_ID'] = slides_name
                
            tile_info_df.to_csv(os.path.join(save_location, slides_name + "_tiles.csv"), index = False)
            lvl_img.save(os.path.join(save_location,  slides_name + '_low-res.png'))
            lvl_mask.save(os.path.join(save_location, slides_name + '_tissue.png'))
            slide_ROIS(polygons=tissue, mpp=float(mpp),savename=os.path.join(save_location, slides_name + '_tissue.json'),labels='tissue', ref=[0, 0], roi_color=-16770432)