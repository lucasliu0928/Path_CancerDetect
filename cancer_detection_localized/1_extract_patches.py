#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import h5py
from Utils import *
from Utils import create_dir_if_not_exists
from Utils import generate_deepzoom_tiles, extract_tile_start_end_coords
warnings.filterwarnings("ignore")


import torchvision
import torch
from platform import python_version
print("Python: " + python_version())
print("fastai: " + fastai.__version__)
print("torch: " + torch.__version__)
print("torchvision: " + torchvision.__version__)



cur_wd = '/fh/scratch/delete90/etzioni_r/lucas_l/michael_project/mutation_pred/'
save_location = cur_wd + 'intermediate_data/cancer_prediction_results102824/'  
save_location2 = cur_wd + 'intermediate_data/cancer_prediction_results102824/tiles/'  
save_location3 = cur_wd + 'intermediate_data/cancer_prediction_results102824/cancer_pred_out/'  
mag_extract = 20 # do not change this, model trained at 250x250 at 20x
save_image_size = 250  # do not change this, model trained at 250x250 at 20x
pixel_overlap = 100  # specify the level of pixel overlap in your saved images
limit_bounds = True  # this is weird, dont change it
tiff_lvl =2 # low res pyramid level to grab
save_location4 = save_location3 + str(pixel_overlap) + 'and' + str(tiff_lvl)  # args.save_location
save_location6 = save_location2 + str(pixel_overlap) + 'and' + str(tiff_lvl)  # args.save_location


create_dir_if_not_exists(save_location)
create_dir_if_not_exists(save_location2)
create_dir_if_not_exists(save_location3)
create_dir_if_not_exists(save_location4)
create_dir_if_not_exists(save_location6)


# cur_id = 'OPX_020'
# _file = cur_wd + "data/OPX/" + cur_id + ".tif"

# cur_id = '(2017-0133) 15-B_A1-2' 
# _file = '/fh/scratch/delete90/haffner_m/user/scan_archives/Prostate/MDAnderson/CCola/all_slides/' + cur_id + '.svs'

selected_ids = ['OPX_007','OPX_010','OPX_033','OPX_049','OPX_077','OPX_090','OPX_182','OPX_185','OPX_186','OPX_194']

for cur_id in selected_ids:
    _file = cur_wd + "data/OPX/" + cur_id + ".tif"
    #Load slides
    oslide = openslide.OpenSlide(_file)
    save_name = str(Path(os.path.basename(_file)).with_suffix(''))
    
    save_location5 = save_location4 + "/" + cur_id + "/" 
    create_dir_if_not_exists(save_location5)
    
    #Generate tiles
    tiles, tile_lvls, physSize = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)
    
    
    #Get low res image,  intermeadiate level for probability map
    slide_dim = oslide.level_dimensions[tiff_lvl] #slide dim at tiff_lvl
    lvl_resize = oslide.level_downsamples[tiff_lvl] #downsample factor
    lvl_img = oslide.read_region((0, 0), tiff_lvl, slide_dim)
    lvl_img.save(os.path.join(save_location5 + save_name + '_low-res.png'))
    
    
    # send to get tissue polygons
    print('detecting tissue')
    tissue, he_mask = do_mask_original(lvl_img,lvl_resize)
    
    #init x_map and x_count at intermeadiate level size
    x_map   = np.zeros((lvl_img.size[1], lvl_img.size[0]), float)
    x_count = np.zeros((lvl_img.size[1], lvl_img.size[0]), float)
    
    
    lvl =  mag_extract
    if lvl in tile_lvls:
        lvl_in_deepzoom = tile_lvls.index(lvl)
        # pull tile info for level
        x_tiles, y_tiles = tiles.level_tiles[lvl_in_deepzoom] #this extract tiles at mag_extract
        print(x_tiles, y_tiles)
        tile_info = []
        for y in range(0, y_tiles):
            if y % 50 == 0: print(y)
            for x in range(0, x_tiles):
                #Grab tile coordinates
                tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords(tiles, lvl_in_deepzoom, x, y) #this returns the coors at level 0 reference original slides
                
                #Check tissue membership
                tile_tiss = check_tissue(tile_starts= tile_starts, tile_ends=tile_ends,roi=tissue)
                if tile_tiss > 0.9: #If the tile has more than 90% tissue coverage
                    #Extract tile
                    tile_pull = tiles.get_tile(lvl_in_deepzoom, (x, y))
                
                    #Check white space
                    ws = whitespace_check(im=tile_pull)
                    
                    if ws < 0.95: #. If the white space is less than 95%
                        tile_info.append(pd.DataFrame({'SAMPLE_ID' : save_name, 
                                                       'MAG_EXTRACT' : lvl,
                                                       'SAVE_IMAGE_SIZE': save_image_size,
                                                       'PIXEL_OVERLAP': pixel_overlap,
                                                       'TIFF_LVL': tiff_lvl,
                                                       'LIMIT_BOUNDS': limit_bounds,
                                                       'TILE_XY_INDEXES' : str((x ,y)),
                                                       'TILE_COOR_ATLV0' : save_coords,
                                                       'WHITE_SPACE' : ws,
                                                       'TISSUE_COVERAGE': tile_tiss}, index = [0]))
    
    tile_info_df = pd.concat(tile_info)
    tile_info_df.to_csv(save_location6 + save_name + ".csv", index = False)

