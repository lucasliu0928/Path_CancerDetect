#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:54:40 2025

@author: jliu6
"""

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
import datetime
import random
import torch
from PIL import ImageCms, Image
warnings.filterwarnings("ignore")

#Load slides
oslide = openslide.OpenSlide(_file)

#Generate tiles
tiles, tile_lvls, physSize, base_mag = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)

print('starting inference')
#get level 0 size in px
l0_w = oslide.level_dimensions[0][0]
l0_h = oslide.level_dimensions[0][1]

#2.5x for probability maps
lvl_resize = get_downsample_factor(base_mag,target_magnification = mag_target_prob) #downsample factor
x_map = np.zeros((int(np.ceil(l0_h/lvl_resize)),int(np.ceil(l0_w/lvl_resize))), float)
x_count = np.zeros((int(np.ceil(l0_h/lvl_resize)),int(np.ceil(l0_w/lvl_resize))), float)


tile_info_df['pred_map_location'] = pd.NA
for index, row in tile_info_df.iterrows():
    if (index % 500 == 0): print(index)
    cur_xy = row['TILE_XY_INDEXES'].strip("()").split(", ")
    x ,y = int(cur_xy[0]) , int(cur_xy[1])
    #Extract tile for prediction
    lvl_in_deepzoom = tile_lvls.index(mag_extract)
    tile_pull = tiles.get_tile(lvl_in_deepzoom, (x, y))
    tile_pull = tile_pull.resize(size=(save_image_size, save_image_size),resample=PIL.Image.LANCZOS) #resize
    
    import staintools
    img = staintools.read_image(tile_pull)
    img = staintools.LuminosityStandardizer.standardize(img)
    tile_name = os.path.basename(tile)
    img_normalized = normalizer.transform(img)
    img_norm = Image.fromarray(img_normalized)
    print(img_norm)
    img_norm.save(RESULTS_DIR + str.replace(tile_name, '.png', '_norm.png')) #example if you save to new path
    
    tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords(tiles, lvl_in_deepzoom, x, y) #get tile coords
    map_xstart, map_xend, map_ystart, map_yend = get_map_startend(tile_starts,tile_ends,lvl_resize) #Get current tile position in map
    tile_info_df.loc[index,'pred_map_location'] = str(tuple([map_xstart, map_xend, map_ystart, map_yend]))

    #Cancer segmentation
    tile_pull = np.array(tile_pull)
    with model.no_bar():
        inp, pred_class, pred_idx, outputs = model.predict(tile_pull[:, :, 0:3], with_input=True)
    
