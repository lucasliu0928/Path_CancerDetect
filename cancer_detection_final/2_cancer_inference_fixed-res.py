#!/usr/bin/env python
# coding: utf-8
# ENV: paimg9
import sys
import os
import argparse
import pandas as pd
import warnings
from fastai.vision.all import load_learner
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists
from Utils import cancer_inference_wsi , cancer_inference_tma
from skimage import io
from train_utils import str2bool
warnings.filterwarnings("ignore")


#Run: 
#source ~/.bashrc
#conda activate paimg9
#python3 -u 2_cancer_inference_fixed-res.py  --cohort_name Neptune --pixel_overlap 100 --select_idx_start 0 --select_idx_end 1

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--mag_extract', default='20', type=int, help='specify magnification, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--save_image_size', default='250', type=int, help='the size of extracted tiles')
parser.add_argument('--pixel_overlap', default='0', type=int, help='specify the level of pixel overlap in your saved tiles, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--mag_target_prob', default='2.5', type=float, help='magnification for cancer detection: e.g., 2.5x')
parser.add_argument('--mag_target_tiss', default='1.25', type=float, help='magnification for tissue detection: e.g., 1.25x')
parser.add_argument('--bi_thres', default='0.4', type=float, help='Binary classification threshold for cancer mask')
parser.add_argument('--cohort_name', default='Pluvicto_TMA_Cores', type=str, help='data set name: TAN_TMA_Cores, OPX, TCGA_PRAD, Neptune, Pluvicto_TMA_Cores')
parser.add_argument('--stain_norm', default='norm', type=str, help='norm or no_norm')
parser.add_argument('--fine_tuned_model', type=str2bool, default=False, help='whether or not to use fine-tuned model')

parser.add_argument('--select_idx_start', default = 0,type=int)
parser.add_argument('--select_idx_end', default = 1, type=int)

if __name__ == '__main__':
    
    args = parser.parse_args()


    ############################################################################################################
    #USER INPUT 
    ############################################################################################################
    mag_extract = args.mag_extract        # do not change this, model trained at 250x250 at 20x
    save_image_size = args.save_image_size   # do not change this, model trained at 250x250 at 20x
    pixel_overlap = args.pixel_overlap       # specify the level of pixel overlap in your saved images
    limit_bounds = True     # this is weird, dont change it
    smooth = True           # whether or not to gaussian smooth the output probability map
    mag_target_prob = args.mag_target_prob   # 2.5x for probality maps
    mag_target_tiss = args.mag_target_tiss   #1.25x for tissue detection, this is not used for TMA
    bi_thres = args.bi_thres           #Binary classification threshold for cancer mask
    cohort_name = args.cohort_name
    folder_name = "IMSIZE" + str(save_image_size) + "_OL" + str(pixel_overlap)


    ############################################################################################################
    #DIR
    ############################################################################################################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    wsi_location_ccola = proj_dir + '/data/CCola/all_slides/'
    wsi_location_opx = proj_dir + '/data/OPX/'
    wsi_location_tan = proj_dir + 'data/TAN_TMA_Cores/'
    wsi_location_tcga = proj_dir + 'data/TCGA_PRAD/'
    wsi_location_nep = proj_dir + 'data/Neptune/'
    wsi_location_plu = proj_dir + 'data/Pluvicto_TMA_Cores/'
    feature_location = os.path.join(proj_dir,'intermediate_data','1_tile_pulling', cohort_name, folder_name) #cancer_prediction_results110224
    model_path = os.path.join(proj_dir,'models','cancer_detection_models', 'mets')
    
    if args.stain_norm == "norm":
        out_location = os.path.join(proj_dir,'intermediate_data','2_cancer_detection_stainnormed', cohort_name, folder_name)
    elif args.stain_norm == "no_norm":
        out_location = os.path.join(proj_dir,'intermediate_data','2_cancer_detection_nostainnormed', cohort_name, folder_name)
    create_dir_if_not_exists(out_location)

    ############################################################################################################
    #Select IDS
    ############################################################################################################
    #Get IDs that are in FT train or already processed to exclude 
    fine_tune_ids_df = pd.read_csv(proj_dir + 'intermediate_data/0_cd_finetune/cancer_detection_training/all_tumor_fraction_info.csv')
    ft_train_ids = list(fine_tune_ids_df.loc[fine_tune_ids_df['Train_OR_Test'] == 'Train','sample_id'])
    toexclude_ids = ft_train_ids + ['cca3af0c-3e0e-4cfb-bb07-459c979a0bd5'] #The latter one is TCGA issue file
    
    #All available IDs
    opx_ids = [x.replace('.tif','') for x in os.listdir(wsi_location_opx) if x != '.DS_Store'] #353
    ccola_ids = [x.replace('.svs','') for x in os.listdir(wsi_location_ccola) if '(2017-0133)' in x] #234
    tan_ids =  [x.replace('.tif','') for x in os.listdir(wsi_location_tan)  if x != '.DS_Store'] #677
    tcga_ids = [x.replace('.svs','') for x in os.listdir(wsi_location_tcga) if x != '.DS_Store'] #449
    nep_ids =  [x.replace('.tif','') for x in os.listdir(wsi_location_nep)  if x != '.DS_Store'] #350
    plu_ids =  [x.replace('.tif','') for x in os.listdir(wsi_location_plu)  if x != '.DS_Store'] #100

    if cohort_name == "OPX":
        all_ids = opx_ids 
    elif cohort_name == "ccola":
        all_ids = ccola_ids
    elif cohort_name == "TAN_TMA_Cores":
        all_ids = tan_ids
    elif cohort_name == 'TCGA_PRAD':
        all_ids = tcga_ids
    elif cohort_name == "Neptune":
        all_ids = nep_ids
    elif cohort_name == "Pluvicto_TMA_Cores":
        all_ids = plu_ids
    
    #Exclude ids in ft_train or processed
    selected_ids = [x for x in all_ids if x not in toexclude_ids]
    selected_ids.sort()
    
    
    ############################################################################################################
    #Load normalization norm target image
    ############################################################################################################
    if args.stain_norm == "norm":
        tile_norm_img_path = os.path.join(proj_dir,'intermediate_data/6A_tile_for_stain_norm/')
        norm_target_img = io.imread(os.path.join(tile_norm_img_path, 'SU21-19308_A1-2_HE_40X_MH110821_40_16500-20500_500-500.png'))
    elif args.stain_norm == "no_norm":
        norm_target_img = None
                
    ############################################################################################################
    #START
    ############################################################################################################
    ct = 0
    for cur_id in selected_ids[args.select_idx_start:args.select_idx_end]:
        if (ct % 50 == 0): print(ct)
        ct += 1
    
        save_location = out_location + "/" + cur_id + "/" 
        create_dir_if_not_exists(save_location)
    
        slides_name = cur_id
        if 'OPX' in cohort_name:
            _file = wsi_location_opx + slides_name + ".tif"
            rad_tissue = 5
        elif 'ccola' in cohort_name:
            _file = wsi_location_ccola + slides_name + '.svs'
            rad_tissue = 2
        elif 'TAN_TMA_Cores' in cohort_name:
            _file = wsi_location_tan + slides_name + '.tif'
            rad_tissue = 2
        elif 'Pluvicto_TMA_Cores' in cohort_name:
            _file = wsi_location_plu + slides_name + '.tif'
            rad_tissue = 2
        elif 'NEP' in cohort_name:
            _file = wsi_location_nep + slides_name + ".tif"
            if cur_id == 'NEP-081PS2-1_HE_MH_03282024' or cur_id == 'NEP-123PS1-1_HE_MH06032024':
                rad_tissue = 2
            else:
                rad_tissue = 5
        else:
            slides_name = [f for f in os.listdir(wsi_location_tcga + cur_id + '/') if '.svs' in f][0].replace('.svs','')
            _file = wsi_location_tcga + cur_id + '/' + slides_name + '.svs'
            rad_tissue = 2
    
    
        #Load model   
        if args.fine_tuned_model == True:
            learn = load_learner(os.path.join(model_path,'ft_models','dlv3_2ep_2e4_update-07182023_RT_fine_tuned..pkl'),cpu=False) #all use mets model
            save_location = save_location + "ft_model" + "/"
            create_dir_if_not_exists(save_location)
        else:
            
            learn = load_learner(os.path.join(model_path,'dlv3_2ep_2e4_update-07182023_RT.pkl'),cpu=False) #all use prior mets model
            save_location = save_location + "prior_model" + "/"
            create_dir_if_not_exists(save_location)
    
        #Check if already processed
        out_files = os.listdir(os.path.join(save_location))
        n_out_files = len([f for f in out_files if not f.startswith('.')])
        if  n_out_files == 8:
            print(f'PROCESSED: index: {ct}, ID: {cur_id}, n_files: {n_out_files}')      
        else:
            print(f'NOT PROCESSED: index: {ct}, ID: {cur_id}, n_files: {n_out_files}')  
            #Load tile info 
            if cohort_name == 'TCGA_PRAD':
                tile_info_df = pd.read_csv(os.path.join(feature_location, cur_id, slides_name + "_tiles.csv"))
            else:
                tile_info_df = pd.read_csv(os.path.join(feature_location, cur_id, cur_id + "_tiles.csv"))
            print(tile_info_df.shape)
            

            #Run
            if 'TMA' in cur_id:
                cancer_inference_tma(_file, learn, tile_info_df, save_image_size, 
                                     pixel_overlap, mag_target_prob, rad_tissue, smooth, bi_thres, save_location, save_name = cur_id,
                                     stain_norm_target_img = norm_target_img)
            else:
                cancer_inference_wsi(_file, learn, tile_info_df, mag_extract, save_image_size, pixel_overlap, 
                                     limit_bounds, mag_target_prob, mag_target_tiss, rad_tissue, smooth, bi_thres, save_location, 
                                     save_name = slides_name,
                                     stain_norm_target_img = norm_target_img)