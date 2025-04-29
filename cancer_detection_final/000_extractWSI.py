#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 20:29:57 2025

@author: jliu6
"""

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
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator


cohort_name = 'OPX'

proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
wsi_location_ccola = proj_dir + '/data/CCola/all_slides/'
wsi_location_opx = proj_dir + '/data/OPX/' #N= 353, Now OPX has all the old and newly added samples (Oncoplex_deidentified)
wsi_location_opx2 = os.path.join(proj_dir,'data','Oncoplex_deidentified')
wsi_location_tan = proj_dir + 'data/TAN_TMA_Cores/'
wsi_location_tcga = proj_dir + 'data/TCGA_PRAD/'
wsi_location_nep = proj_dir + 'data/Neptune/'
out_location = os.path.join(proj_dir,'intermediate_data','000_WSI_extraction', cohort_name)  #1_feature_extraction, cancer_prediction_results110224


create_dir_if_not_exists(out_location)


selected_ids = ['OPX_007']

for cur_id in selected_ids:

    cur_id = selected_ids[0]

    #check if processed:
    slides_name = cur_id
    if 'OPX' in cur_id:
        _file = wsi_location_opx + slides_name + ".tif"

    slide = openslide.OpenSlide(_file)
    
    # Get full dimensions at level 0 (original resolution)
    width, height = slide.dimensions
    print(f"Slide dimensions: {width} x {height}")



    thumbnail = slide.get_thumbnail((1000, 1000))  # Size (width, height)
    thumbnail.save(os.path.join(out_location, cur_id + ".png"))


