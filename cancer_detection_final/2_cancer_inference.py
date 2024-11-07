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
import h5py
sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists
from Utils import generate_deepzoom_tiles, extract_tile_start_end_coords
from Utils import get_map_startend
from Utils import do_mask_original,cancer_mask2,tile_ROIS,slide_ROIS
warnings.filterwarnings("ignore")


import torchvision
import torch
from platform import python_version
print("Python: " + python_version())
print("fastai: " + fastai.__version__)
print("torch: " + torch.__version__)
print("torchvision: " + torchvision.__version__)


#USER INPUT 
mag_extract = 20 # do not change this, model trained at 250x250 at 20x
save_image_size = 250  # do not change this, model trained at 250x250 at 20x
pixel_overlap = 100  # specify the level of pixel overlap in your saved images
limit_bounds = True  # this is weird, dont change it
tiff_lvl =2 # low res pyramid level to grab
ft_model = True

proj_dir = '/fh/scratch/delete90/etzioni_r/lucas_l/michael_project/mutation_pred/'
wsi_location = proj_dir + "data/OPX/"
#wsi_location = '/fh/scratch/delete90/haffner_m/user/scan_archives/Prostate/MDAnderson/CCola/all_slides/'
out_location = proj_dir + 'intermediate_data/cancer_prediction_results102824/'
model_path_m = proj_dir + 'models/cancer_detection_models/mets/ft_models/dlv3_2ep_2e4_update-07182023_RT_fine_tuned..pkl' #METS
model_path_m_prior = proj_dir + 'models/cancer_detection_models/mets/dlv3_2ep_2e4_update-07182023_RT.pkl' #METS before fine-tune
#model_path_l = proj_dir + 'models/cancer_detection_models/local/binary_mblntv3_25ep_lr1e5_wAug_MixUpLS_sz500_bs12_10x.pkl' #LOCAL
data_mut_path = proj_dir + 'data/MutationCalls/'


#Create output dir
create_dir_if_not_exists(out_location)
save_location_tiles = out_location + 'tiles/'  
create_dir_if_not_exists(save_location_tiles)
save_location_pred = out_location + 'cancer_pred_out/'  
create_dir_if_not_exists(save_location_pred)
save_location_pred = save_location_pred + str(pixel_overlap) + 'and' + str(tiff_lvl) + "/"
create_dir_if_not_exists(save_location_pred)
save_location_tiles = save_location_tiles + str(pixel_overlap) + 'and' + str(tiff_lvl) + "/"
create_dir_if_not_exists(save_location_tiles)


# #load mutation site
# mut_site_df = pd.read_excel(data_mut_path + 'OPX_anatomic sites.xlsx')
# mut_site_df
# mets_ids = list(mut_site_df.loc[mut_site_df['Anatomic site']!= 'Prostate', 'OPX_Number'])
# local_ids = list(mut_site_df.loc[mut_site_df['Anatomic site']== 'Prostate', 'OPX_Number'])
# len(mets_ids)
# len(local_ids)

selected_ids = ['OPX_001','OPX_002','OPX_011','OPX_014','OPX_016','OPX_042','(2017-0133) 23-B_A1-8' , 
                '(2017-0133) 25-B_A1-2', '(2017-0133) 28-B_A1-8', '(2017-0133) 32-R_A1-2', 
                '(2017-0133) 95-3-P_A1-8','(2017-0133) 99-B_A1-8']

for cur_id in selected_ids:
    print(cur_id)

    if 'OPX' in cur_id:
        _file = wsi_location + cur_id + ".tif"
    elif '(2017-0133)' in cur_id:
        _file = wsi_location + cur_id + '.svs'

    #Laod slides
    oslide = openslide.OpenSlide(_file)
    save_name = str(Path(os.path.basename(_file)).with_suffix(''))
    
    save_location = save_location_pred + "/" + cur_id + "/" 
    create_dir_if_not_exists(save_location)
    
    #load pytorch model   
    if ft_model == True:
        learn = load_learner(model_path_m,cpu=False) #all use mets model
    else:
        learn = load_learner(model_path_m_prior,cpu=False) #all use prior mets model
        save_location = save_location + "/" + "prior_model" + "/"
        create_dir_if_not_exists(save_location)
    

    #Load tile info 
    tile_info_df = pd.read_csv(save_location_tiles + save_name + ".csv")
    tile_mag_extract = list(set(tile_info_df['MAG_EXTRACT']))[0]
    tile_save_image_size = list(set(tile_info_df['SAVE_IMAGE_SIZE']))[0]
    tile_pixel_overlap = list(set(tile_info_df['PIXEL_OVERLAP']))[0]
    tile_limit_bounds =   list(set(tile_info_df['LIMIT_BOUNDS']))[0]
    tile_tiff_lvl =  list(set(tile_info_df['TIFF_LVL']))[0]
    
    cond1 = (tile_mag_extract == mag_extract)
    cond2 = (tile_save_image_size == save_image_size)
    cond3 = (tile_pixel_overlap == pixel_overlap)
    cond4 = (tile_limit_bounds == limit_bounds)
    cond5 = (tile_tiff_lvl == tiff_lvl)
    
    if cond1 & cond2 & cond3 & cond4 & cond5 :
        can_proceed = True
        print(can_proceed)

    print(tile_info_df.shape)
    if can_proceed == True:
        #Generate tiles
        tiles, tile_lvls, physSize, base_mag = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)
        
        #Get low res image,  intermeadiate level for probability map
        slide_dim = oslide.level_dimensions[tiff_lvl] #slide dim at tiff_lvl
        lvl_resize = oslide.level_downsamples[tiff_lvl] #downsample factor
        
        print('starting inference')
        #init x_map and x_count at intermeadiate level size
        x_map   = np.zeros((slide_dim[1], slide_dim[0]), float)
        x_count = np.zeros((slide_dim[1], slide_dim[0]), float)
        
        tile_info_df['pred_map_location'] = pd.NA
        for index, row in tile_info_df.iterrows():
            if (index % 500 == 0): print(index)
            cur_xy = row['TILE_XY_INDEXES'].strip("()").split(", ")
            x ,y = int(cur_xy[0]) , int(cur_xy[1])
            
            #Extract tile for prediction
            tile_pull = tiles.get_tile(tile_lvls.index(mag_extract), (x, y))
            tile_pull = tile_pull.resize(size=(save_image_size, save_image_size),resample=PIL.Image.LANCZOS) #resize
            tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords(tiles, tile_lvls.index(mag_extract), x, y)
            map_xstart, map_xend, map_ystart, map_yend = get_map_startend(tile_starts,tile_ends,lvl_resize) #Get current tile position in map
            tile_info_df.loc[index,'pred_map_location'] = str(tuple([map_xstart, map_xend, map_ystart, map_yend]))
            #Cancer segmentation
            tile_pull = np.array(tile_pull)
            
            with learn.no_bar():
                inp, pred_class, pred_idx, outputs = learn.predict(tile_pull[:, :, 0:3], with_input=True)
            
            #Get predicted output
            outputs_np = outputs.numpy() #[N_CLASS, IMAGE_SIZE, IMAGE_SIZE]
            output_c1 = PIL.Image.fromarray(outputs_np[1]) #Convert predicted class 1 probabliy to image
            output_c1 = output_c1.resize(size=(map_xend - map_xstart, map_yend - map_ystart),resample=PIL.Image.LANCZOS) #resize for low-res
            output_c1_np = np.array(output_c1)
            
            #Store predicted probabily in map and count
            try: 
                x_count[map_xstart:map_xend,map_ystart:map_yend] += 1
                x_map[map_xstart:map_xend,map_ystart:map_yend] += output_c1_np[1]
            except:
                pass
        
        print('post-processing')
        x_count = np.where(x_count < 1, 1, x_count)
        x_map = x_map / x_count
        slideimg = PIL.Image.fromarray(np.uint8(x_map * 255))
        slideimg = slideimg.convert('L')
        
        cmap = plt.get_cmap('jet')
        rgba_img = cmap(x_map)
        rgb_img = np.delete(rgba_img, 3, 2)
        colimg = PIL.Image.fromarray(np.uint8(rgb_img * 255))
        colimg.save(os.path.join(save_location, save_name + '_cancer_prob.jpeg'))
        
        # send to get tissue polygons
        print('detecting tissue')
        lvl_img = oslide.read_region((0, 0), tiff_lvl, slide_dim)
        tissue, he_mask = do_mask_original(lvl_img,lvl_resize)

        #for diff threshold
        binary_pred_thres_list = [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
        for binary_pred_thres in binary_pred_thres_list:
            save_location_th = save_location + 'TH' + str(binary_pred_thres) + '/' 
            create_dir_if_not_exists(save_location_th)
            
            #Binary classification
            binary_preds = cancer_mask2(slideimg,he_mask, binary_pred_thres) 
            
            #Output annotation
            polygons = tile_ROIS(mask_arr=binary_preds, lvl_resize=lvl_resize)
            slide_ROIS(polygons=polygons, mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                            savename=os.path.join(save_location_th,save_name+'_cancer.json'), labels='AI_tumor', ref=[0,0], roi_color=-16711936)
            slide_ROIS(polygons=tissue, mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                            savename=os.path.join(save_location_th,save_name+'_tissue.json'), labels='tissue', ref=[0,0], roi_color=-16770432)
            
            
            #Get binary prediction for each tile
            #NOTE: prevoiuse do x_map when predition, is not accuate, because the x_map may change as process to the next tile, so need to to this in post processing
            tile_info_df['TUMOR_PIXEL_PERC'] = pd.NA
            for index, row in tile_info_df.iterrows():
                cur_map_loc = row['pred_map_location'].strip("()").split(", ")
                map_xstart, map_xend, map_ystart, map_yend = int(cur_map_loc[0]),int(cur_map_loc[1]), int(cur_map_loc[2]), int(cur_map_loc[3])
            
                #Get current prediction
                cur_pred = binary_preds[map_xstart:map_xend,map_ystart:map_yend]
                cur_count1 = np.sum(cur_pred == 1)
                cur_perc1  = (cur_count1 / cur_pred.size)
                tile_info_df.loc[index,'TUMOR_PIXEL_PERC'] = cur_perc1
            
            tile_info_df.to_csv(save_location_th + save_name + "_TILE_TUMOR_PERC.csv", index = False)
            
            
            #Grab tiles and plot
            tile_info_df_gt05 = tile_info_df.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
            for i in range(0,5): #top5
                cur_row = tile_info_df_gt05.iloc[i]
                cur_xy = cur_row['TILE_XY_INDEXES'].strip("()").split(", ")
                x ,y = int(cur_xy[0]) , int(cur_xy[1])
                tile_pull_ex = tiles.get_tile(tile_lvls.index(mag_extract), (x, y))
                tile_pull_ex = tile_pull_ex.resize(size=(save_image_size, save_image_size),resample=PIL.Image.LANCZOS) #resize
                tile_pull_ex.save(os.path.join(save_location_th, "EX_TILE"  + str(i) + ".png"))

