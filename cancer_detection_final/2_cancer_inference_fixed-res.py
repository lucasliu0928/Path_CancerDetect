#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from Utils import cancer_mask_fix_res, tile_ROIS
warnings.filterwarnings("ignore")


#USER INPUT 
mag_extract = 20        # do not change this, model trained at 250x250 at 20x
save_image_size = 250   # do not change this, model trained at 250x250 at 20x
pixel_overlap = 100     # specify the level of pixel overlap in your saved images
limit_bounds = True     # this is weird, dont change it
smooth = True           # whether or not to gaussian smooth the output probability map
ft_model = True        # whether or not to use fine-tuned model
mag_target_prob = 2.5   # 2.5x for probality maps
mag_target_tiss = 1.25   #1.25x for tissue detection
bi_thres = 0.4  #Binary classification threshold for cancer mask

#DIR
proj_dir = '/fh/scratch/delete90/etzioni_r/lucas_l/michael_project/mutation_pred/'
wsi_location_ccola = '/fh/scratch/delete90/haffner_m/user/scan_archives/Prostate/MDAnderson/CCola/all_slides/'
wsi_location_opx = proj_dir + '/data/OPX/'
out_location = proj_dir + 'intermediate_data/cancer_prediction_results110224/'+ "IMSIZE" + str(save_image_size) + "_OL" + str(pixel_overlap) + "/"
model_path = proj_dir + 'models/cancer_detection_models/mets/'


selected_ids = ['OPX_015', 'OPX_017', 'OPX_020', 
                '(2017-0133) 4-2-B_B1-1', '(2017-0133) 15-B_A1-2', 
                '(2017-0133) 23-B_A1-8', '(2017-0133) 25-B_A1-2', 
                '(2017-0133) 28-B_A1-8', '(2017-0133) 32-R_A1-2',
                '(2017-0133) 95-3-P_A1-8','(2017-0133) 99-B_A1-8']
for cur_id in selected_ids:

    if 'OPX' in cur_id:
        _file = wsi_location_opx + cur_id + ".tif"
        rad_tissue = 5
    elif '(2017-0133)' in cur_id:
        _file = wsi_location_ccola + cur_id + '.svs'
        rad_tissue = 2

    #Load slides
    oslide = openslide.OpenSlide(_file)
    save_name = str(Path(os.path.basename(_file)).with_suffix(''))

    save_location = out_location + cur_id + "/" 
    create_dir_if_not_exists(save_location)

    #Load model   
    if ft_model == True:
        learn = load_learner(model_path + 'ft_models/dlv3_2ep_2e4_update-07182023_RT_fine_tuned..pkl',cpu=False) #all use mets model
        save_location = save_location + "ft_model" + "/"
        create_dir_if_not_exists(save_location)
    else:
        learn = load_learner(model_path + 'dlv3_2ep_2e4_update-07182023_RT.pkl',cpu=False) #all use prior mets model
        save_location = save_location + "prior_model" + "/"
        create_dir_if_not_exists(save_location)

    #Load tile info 
    tile_info_df = pd.read_csv(out_location + cur_id + "/"  + save_name + "_tiles.csv")
    tile_mag_extract = list(set(tile_info_df['MAG_EXTRACT']))[0]
    tile_save_image_size = list(set(tile_info_df['SAVE_IMAGE_SIZE']))[0]
    tile_pixel_overlap = list(set(tile_info_df['PIXEL_OVERLAP']))[0]
    tile_limit_bounds =   list(set(tile_info_df['LIMIT_BOUNDS']))[0]

    cond1 = (tile_mag_extract == mag_extract)
    cond2 = (tile_save_image_size == save_image_size)
    cond3 = (tile_pixel_overlap == pixel_overlap)
    cond4 = (tile_limit_bounds == limit_bounds)

    if cond1 & cond2 & cond3 & cond4:
        can_proceed = True
        print(can_proceed)

    print(tile_info_df.shape)

    if can_proceed == True:
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
            tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords(tiles, lvl_in_deepzoom, x, y) #get tile coords
            map_xstart, map_xend, map_ystart, map_yend = get_map_startend(tile_starts,tile_ends,lvl_resize) #Get current tile position in map
            tile_info_df.loc[index,'pred_map_location'] = str(tuple([map_xstart, map_xend, map_ystart, map_yend]))
            
            #Cancer segmentation
            tile_pull = np.array(tile_pull)
            with learn.no_bar():
                inp, pred_class, pred_idx, outputs = learn.predict(tile_pull[:, :, 0:3], with_input=True)
            
            #Get predicted output
            #NOTe: updated 11/06, use cv2.resize
            outputs_np = outputs.numpy() #[N_CLASS, IMAGE_SIZE, IMAGE_SIZE]
            output_c1_np = cv2.resize(outputs_np[1], (map_yend - map_ystart,map_xend - map_xstart)) #class1 predicted prob, resize (width (col in np), height(row in np))
            output_c1_np = output_c1_np.round(2)
            
            #Store predicted probabily in map and count
            try: 
                x_count[map_xstart:map_xend,map_ystart:map_yend] += 1
                x_map[map_xstart:map_xend,map_ystart:map_yend] += output_c1_np
            except:
                pass

        print('post-processing')
        print('Cancer Prob generation')
        x_count = np.where(x_count < 1, 1, x_count)
        x_map = x_map / x_count
        x_map[x_map>1]=1

        if smooth == True:
            x_sm = filters.gaussian(x_map, sigma=2)
        if smooth == False:
            x_sm = x_map
        cmap = plt.get_cmap('jet')
        rgba_img = cmap(x_sm)
        rgb_img = np.delete(rgba_img, 3, 2)
        colimg = PIL.Image.fromarray(np.uint8(rgb_img * 255))
        colimg.save(os.path.join(save_location, save_name + '_cancer_prob.jpeg'))


        print('Get cancer mask')
        print('detecting tissue')
        #1.25x tissue detection
        lvl_resize_tissue = get_downsample_factor(base_mag,target_magnification = mag_target_tiss) #downsample factor
        lvl_img = get_image_at_target_mag(oslide,l0_w, l0_h,lvl_resize_tissue)
        tissue, he_mask = do_mask_original(lvl_img, lvl_resize_tissue, rad = rad_tissue)

        #Binary classification
        binary_preds = cancer_mask_fix_res(x_sm,cv2.resize(np.uint8(he_mask),(x_sm.shape[1],x_sm.shape[0])), bi_thres)

        #Output annotation
        print('saving...')
        polygons = tile_ROIS(mask_arr=binary_preds, lvl_resize=lvl_resize)
        slide_ROIS(polygons=polygons, mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                        savename=os.path.join(save_location,save_name+'_cancer.json'), labels='AI_tumor', ref=[0,0], roi_color=-16711936)


        #Get binary prediction for each tile
        #NOTE: prevoiuse do x_map when predition, is not accuate, because the x_map may change as process to the next tile, so need to to this in post processing
        tile_info_df['TUMOR_PIXEL_PERC'] = pd.NA
        for index, row in tile_info_df.iterrows():
            cur_map_loc = row['pred_map_location'].strip("()").split(", ")
            map_xstart, map_xend, map_ystart, map_yend = int(cur_map_loc[0]),int(cur_map_loc[1]), int(cur_map_loc[2]), int(cur_map_loc[3])

            #Get current prediction
            cur_pred = binary_preds[map_xstart:map_xend,map_ystart:map_yend]
            cur_count1 = np.sum(cur_pred == 1) #num pixels that has predicted prob = 1
            cur_perc1  = (cur_count1 / cur_pred.size) #fraction of pixels prob = 1
            tile_info_df.loc[index,'TUMOR_PIXEL_PERC'] = cur_perc1

        tile_info_df.to_csv(save_location + save_name + "_TILE_TUMOR_PERC.csv", index = False)


        #Grab tiles and plot
        tile_info_df_sorted = tile_info_df.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
        for i in range(0,5): #top5
            cur_row = tile_info_df_sorted.iloc[i]
            cur_xy = cur_row['TILE_XY_INDEXES'].strip("()").split(", ")
            x ,y = int(cur_xy[0]) , int(cur_xy[1])
            tile_pull_ex = tiles.get_tile(tile_lvls.index(mag_extract), (x, y))
            tile_pull_ex = tile_pull_ex.resize(size=(save_image_size, save_image_size),resample=PIL.Image.LANCZOS) #resize

            #Save tile
            cur_tf = round(cur_row['TUMOR_PIXEL_PERC'],2)
            cur_mag = cur_row['MAG_EXTRACT']
            tile_save_name = "TILE_@" + str(cur_mag) + "x" + "_X" + str(x) +  "Y" + str(y) +   "_TF" + str(cur_tf) + ".png"
            tile_pull_ex.save(os.path.join(save_location, tile_save_name))

