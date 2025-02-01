#!/usr/bin/env python
# coding: utf-8
# ENV: paimg9

import sys
import os
import numpy as np
import cv2
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import xml.etree.ElementTree as ET
from xml.dom import minidom
import geojson
import argparse
import matplotlib.pyplot as plt
import fastai
from fastai.vision.all import *
import PIL
matplotlib.use('Agg')
import pandas as pd
import datetime
from skimage import draw, measure, morphology, filters
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
from shapely.ops import cascaded_union, unary_union
import json
import shapely
import warnings
from scipy import ndimage
sys.path.insert(0, '../Utils/')
from Utils import generate_deepzoom_tiles, extract_tile_start_end_coords
from Utils import do_mask_original,check_tissue,whitespace_check
from Utils import slide_ROIS
from Utils import get_downsample_factor, get_image_at_target_mag
from Utils import create_dir_if_not_exists
from Utils import get_map_startend
from Utils import cancer_mask_fix_res, tile_ROIS, check_any_invalid_poly, make_valid_poly
from Utils import convert_img
from Utils import plot_tiles_with_topK_cancerprob, get_binary_pred_tile
from Utils import cancer_inference_wsi , cancer_inference_tma
warnings.filterwarnings("ignore")


############################################################################################################
#USER INPUT 
############################################################################################################
mag_extract = 20        # do not change this, model trained at 250x250 at 20x
save_image_size = 250   # do not change this, model trained at 250x250 at 20x
pixel_overlap = 0       # specify the level of pixel overlap in your saved images
limit_bounds = True     # this is weird, dont change it
smooth = True           # whether or not to gaussian smooth the output probability map
ft_model = True         # whether or not to use fine-tuned model
mag_target_prob = 2.5   # 2.5x for probality maps, this might need to change to 4x for TMA
mag_target_tiss = 1.25   #1.25x for tissue detection, this is not used for TMA
bi_thres = 0.4           #Binary classification threshold for cancer mask
cohort_name = "TAN_TMA_Cores"

############################################################################################################
#DIR
############################################################################################################
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
wsi_location_ccola = proj_dir + '/data/CCola/all_slides/'
wsi_location_opx = proj_dir + '/data/OPX/'
wsi_location_tan = proj_dir + 'data/TAN_TMA_Cores/'
feature_location = proj_dir + 'intermediate_data/1_tile_pulling/'+ cohort_name + "/" + "IMSIZE" + str(save_image_size) + "_OL" + str(pixel_overlap) + "/" #cancer_prediction_results110224
model_path = proj_dir + 'models/cancer_detection_models/mets/'

out_location = proj_dir + 'intermediate_data/2_cancer_detection/'+ cohort_name + "/" + "IMSIZE" + str(save_image_size) + "_OL" + str(pixel_overlap) + "/"
create_dir_if_not_exists(out_location)


############################################################################################################
#Select IDS
############################################################################################################
#Get IDs that are in FT train or already processed to exclude 
fine_tune_ids_df = pd.read_csv(proj_dir + 'intermediate_data/cd_finetune/cancer_detection_training/all_tumor_fraction_info.csv')
ft_train_ids = list(fine_tune_ids_df.loc[fine_tune_ids_df['Train_OR_Test'] == 'Train','sample_id'])
toexclude_ids = ft_train_ids + ['OPX_182'] #OPX_182 –Exclude Possible Colon AdenoCa 

#All available IDs
opx_ids = [x.replace('.tif','') for x in os.listdir(wsi_location_opx)] #207
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
selected_ids = [x for x in all_ids if x not in toexclude_ids]
selected_ids.sort()


############################################################################################################
#START
############################################################################################################
for cur_id in selected_ids:

    save_location = out_location + cur_id + "/" 
    create_dir_if_not_exists(save_location)

    if 'OPX' in cur_id:
        _file = wsi_location_opx + cur_id + ".tif"
        rad_tissue = 5
    elif '(2017-0133)' in cur_id:
        _file = wsi_location_ccola + cur_id + '.svs'
        rad_tissue = 2
    elif 'TMA' in cur_id:
        _file = wsi_location_tan + cur_id + '.tif'
        rad_tissue = 2

    #Load model   
    if ft_model == True:
        learn = load_learner(model_path + 'ft_models/dlv3_2ep_2e4_update-07182023_RT_fine_tuned..pkl',cpu=False) #all use mets model
        save_location = save_location + "ft_model" + "/"
        create_dir_if_not_exists(save_location)
    else:
        learn = load_learner(model_path + 'dlv3_2ep_2e4_update-07182023_RT.pkl',cpu=False) #all use prior mets model
        save_location = save_location + "prior_model" + "/"
        create_dir_if_not_exists(save_location)

    #Check if already processed
    if os.path.exists(save_location + "ft_model" + "/") == False:
        #Load tile info 
        tile_info_df = pd.read_csv(feature_location + cur_id + "/"  + cur_id + "_tiles.csv")
        print(tile_info_df.shape)
        
        #Run
        if 'OPX' in cur_id or '(2017-0133)' in cur_id:
            cancer_inference_wsi(_file, learn, tile_info_df, mag_extract, save_image_size, pixel_overlap, limit_bounds, mag_target_prob, mag_target_tiss, rad_tissue, smooth, bi_thres, save_location, save_name = cur_id)
        elif 'TMA' in cur_id:
            cancer_inference_tma(_file, learn, tile_info_df, save_image_size, pixel_overlap, mag_target_prob, rad_tissue, smooth, bi_thres, save_location, save_name = cur_id)

