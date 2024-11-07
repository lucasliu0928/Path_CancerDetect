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
from Utils import generate_deepzoom_tiles, extract_tile_start_end_coords
from Utils import do_mask_original,check_tissue,whitespace_check
from Utils import slide_ROIS
from Utils import get_downsample_factor, get_image_at_target_mag
from Utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")


#USER INPUT 
mag_extract = 20 # do not change this, model trained at 250x250 at 20x
save_image_size = 250  # do not change this, model trained at 250x250 at 20x
pixel_overlap = 100  # specify the level of pixel overlap in your saved images
limit_bounds = True  # this is weird, dont change it
rad_tissue = 2 #radius for tissue detection

proj_dir = '/fh/scratch/delete90/etzioni_r/lucas_l/michael_project/mutation_pred/'
wsi_location = proj_dir + "data/OPX/"
#wsi_location = '/fh/scratch/delete90/haffner_m/user/scan_archives/Prostate/MDAnderson/CCola/all_slides/'
out_location = proj_dir + 'intermediate_data/cancer_prediction_results110224/'

#Create output dir
create_dir_if_not_exists(out_location)
save_location_tiles = out_location + 'tiles/'  
create_dir_if_not_exists(save_location_tiles)
save_location_pred = out_location + 'cancer_pred_out/'  
create_dir_if_not_exists(save_location_pred)
save_location_pred = save_location_pred + "OL" + str(pixel_overlap) + "/"
create_dir_if_not_exists(save_location_pred)
save_location_tiles = save_location_tiles + "OL" + str(pixel_overlap) + "/"
create_dir_if_not_exists(save_location_tiles)


selected_ids = ['OPX_015', 'OPX_017', 'OPX_020']
['OPX_010','OPX_024','OPX_031','OPX_040','OPX_047','OPX_057','OPX_088',
'(2017-0133) 4-2-B_B1-1','(2017-0133) 15-B_A1-2', '(2017-0133) 23-B_A1-8',
'(2017-0133) 25-B_A1-2', '(2017-0133) 28-B_A1-8','(2017-0133) 32-R_A1-2','(2017-0133) 95-3-P_A1-8','(2017-0133) 99-B_A1-8']
for cur_id in selected_ids:

    if 'OPX' in cur_id:
        _file = wsi_location + cur_id + ".tif"
    elif '(2017-0133)' in cur_id:
        _file = wsi_location + cur_id + '.svs'

    #Load slides
    oslide = openslide.OpenSlide(_file)
    save_name = str(Path(os.path.basename(_file)).with_suffix(''))
    save_location = save_location_pred + "/" + cur_id + "/" 
    create_dir_if_not_exists(save_location)

    #Generate tiles
    tiles, tile_lvls, physSize, base_mag = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)

    #get level 0 size in px
    l0_w = oslide.level_dimensions[0][0]
    l0_h = oslide.level_dimensions[0][1]


    #1.25x for low res img 
    mag_target = 1.25
    lvl_resize = get_downsample_factor(base_mag,mag_target) #downsample factor
    lvl_img = get_image_at_target_mag(oslide,l0_w, l0_h,lvl_resize)
    lvl_img.save(os.path.join(save_location + save_name + '_low-res.png'))


    # 1.25X for tissue detection
    print('detecting tissue')
    tissue, he_mask = do_mask_original(lvl_img, lvl_resize, rad = rad_tissue)
    slide_ROIS(polygons=tissue, mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                    savename=os.path.join(save_location, save_name + '_tissue.json'),
                    labels='tissue', ref=[0, 0], roi_color=-16770432)
    lvl_mask = PIL.Image.fromarray(np.uint8(he_mask * 255))
    lvl_mask = lvl_mask.convert('L')
    lvl_mask.save(os.path.join(save_location, save_name + '_tissue.png'))


    print('Start pulling tiles')
    lvl =  mag_extract
    if lvl in tile_lvls:
        #get deep zoom levels
        lvl_in_deepzoom = tile_lvls.index(lvl)
        # pull tile info for level
        x_tiles, y_tiles = tiles.level_tiles[lvl_in_deepzoom] #this extract tiles at mag_extract
        print(y_tiles,x_tiles)
        tile_info = []
        for y in range(0, y_tiles):
            if y % 50 == 0: print(y)
            for x in range(0, x_tiles):
                #Grab tile coordinates
                tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords(tiles, lvl_in_deepzoom, x, y) #this returns the coors at level 0 reference original slides

                #Check tissue membership
                tile_tiss = check_tissue(tile_starts= tile_starts, tile_ends=tile_ends,roi=tissue)
                if tile_tiss > 0.9:
                    #Extract tile
                    tile_pull = tiles.get_tile(lvl_in_deepzoom, (x, y))

                    #Check white space
                    ws = whitespace_check(im=tile_pull)

                    if ws < 0.9: #If the white space is less than 90%
                        tile_info.append(pd.DataFrame({'SAMPLE_ID' : save_name, 
                                                       'MAG_EXTRACT' : lvl,
                                                       'SAVE_IMAGE_SIZE': save_image_size,
                                                       'PIXEL_OVERLAP': pixel_overlap,
                                                       'LIMIT_BOUNDS': limit_bounds,
                                                       'TILE_XY_INDEXES' : str((x ,y)),
                                                       'TILE_COOR_ATLV0' : save_coords,
                                                       'WHITE_SPACE' : ws,
                                                       'TISSUE_COVERAGE': tile_tiss}, index = [0]))

    else:
        print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")

    tile_info_df = pd.concat(tile_info)
    tile_info_df.to_csv(save_location_tiles + save_name + ".csv", index = False)

