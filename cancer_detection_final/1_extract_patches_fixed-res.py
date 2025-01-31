#!/usr/bin/env python
# coding: utf-8


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
from Utils import slide_ROIS
from Utils import create_dir_if_not_exists
from Utils import generating_tiles, generating_tiles_tma
warnings.filterwarnings("ignore")


############################################################################################################
#USER INPUT 
############################################################################################################
mag_extract = 20 # do not change this, model trained at 250x250 at 20x
save_image_size = 250  # do not change this, model trained at 250x250 at 20x
pixel_overlap = 0  # specify the level of pixel overlap in your saved images
limit_bounds = True  # this is weird, dont change it
mag_target_tiss = 1.25   #1.25x for tissue detection
cohort_name = "TAN_TMA_Cores"

#DIR
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
out_location = proj_dir + 'intermediate_data/1_tile_pulling/' + cohort_name + "/" #1_tile_pulling, cancer_prediction_results110224
wsi_location_ccola = proj_dir + '/data/CCola/all_slides/'
wsi_location_opx = proj_dir + '/data/OPX/'
wsi_location_tan = proj_dir + 'data/TAN_TMA_Cores/'

#Create output dir
create_dir_if_not_exists(out_location)
out_location = out_location  + "IMSIZE" + str(save_image_size) + "_OL" + str(pixel_overlap) + "/"
create_dir_if_not_exists(out_location)s


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
#Start 
############################################################################################################
for cur_id in selected_ids:

    if 'OPX' in cur_id:
        _file = wsi_location_opx + cur_id + ".tif"
        rad_tissue = 5
    elif '(2017-0133)' in cur_id:
        _file = wsi_location_ccola + cur_id + '.svs'
        rad_tissue = 2
    elif 'TMA' in cur_id:
        _file = wsi_location_tan + cur_id + '.tif'
        rad_tissue = 2

    save_name = str(Path(os.path.basename(_file)).with_suffix(''))
    save_location = out_location + "/" + save_name + "/" 
    create_dir_if_not_exists(save_location)

    #Generating tiles 
    if 'OPX' in cur_id or '(2017-0133)' in cur_id:
        mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles(cur_id, _file, save_image_size, pixel_overlap, limit_bounds, mag_target_tiss, rad_tissue, mag_extract)
        
    elif 'TMA' in cur_id:
        mpp, lvl_img, lvl_mask, tissue, tile_info_df = generating_tiles_tma(cur_id, _file, save_image_size, pixel_overlap, rad_tissue)
        

    
    tile_info_df.to_csv(save_location + save_name + "_tiles.csv", index = False)
    lvl_img.save(os.path.join(save_location + save_name + '_low-res.png'))
    lvl_mask.save(os.path.join(save_location, save_name + '_tissue.png'))
    slide_ROIS(polygons=tissue, mpp=float(mpp),savename=os.path.join(save_location, save_name + '_tissue.json'),labels='tissue', ref=[0, 0], roi_color=-16770432)
    

