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
from Utils import *
warnings.filterwarnings("ignore")


import torchvision
import torch
from platform import python_version
print("Python: " + python_version())
print("fastai: " + fastai.__version__)
print("torch: " + torch.__version__)
print("torchvision: " + torchvision.__version__)


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")


def generate_deepzoom_tiles(slide, save_image_size, pixel_overlap, limit_bounds):
    # this is physical microns per pixel
    acq_mag = 10.0/float(slide.properties[openslide.PROPERTY_NAME_MPP_X])

    # this is nearest multiple of 20 for base layer
    base_mag = int(20 * round(float(acq_mag) / 20))

    # this is how much we need to resample our physical patches for uniformity across studies
    physSize = round(save_image_size*acq_mag/base_mag)

    # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
    tiles = DeepZoomGenerator(slide, tile_size=physSize-round(pixel_overlap*acq_mag/base_mag), overlap=round(pixel_overlap*acq_mag/base_mag/2), 
                              limit_bounds=limit_bounds)

    # calculate the effective magnification at each level of tiles, determined from base magnification
    tile_lvls = tuple(base_mag/(tiles._l_z_downsamples[i]*tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in range(0,tiles.level_count))

    return tiles, tile_lvls, physSize


def extract_tile_start_end_coords(all_tile, deepzoom_lvl, x_loc, y_loc):
    r'''
    #This func returns the coordiates in the reference level 0 pixels
    '''
    #Get coords
    tile_coords = all_tile.get_tile_coordinates(deepzoom_lvl, (x_loc, y_loc))

    #Get top left pixel coordinates
    topleft_x = tile_coords[0][0]
    topleft_y = tile_coords[0][1]

    #Get level (original)
    o_lvl = tile_coords[1]

    #Get downsample factor
    ds_factor = all_tile._l0_l_downsamples[o_lvl] #downsample factor

    #Get region size in current level 
    rsize_x = tile_coords[2][0] 
    rsize_y = tile_coords[2][1] 

    #Get tile starts and end   
    start_loc = tile_coords[0] #start
    end_loc = (int(topleft_x + ds_factor * rsize_x), int(topleft_y + ds_factor* rsize_y)) #end

    #Get save coord name (first two is the starting loc, and the last two are the x and y size considering dsfactor)
    coord_name = str(topleft_x) + "-" + str(topleft_y) + "_" + '%.0f' % (ds_factor * rsize_x) + "-" + '%.0f' % (ds_factor * rsize_y)
    
    return start_loc, end_loc, coord_name, tile_coords

def get_map_startend(tile_start_cord, tile_end_cord, level_resize):
    m_xstart = int(np.floor(tile_start_cord[1] / level_resize))
    m_xend = int(np.floor(tile_end_cord[1] / level_resize))
    m_ystart = int(np.floor(tile_start_cord[0] / level_resize))
    m_yend = int(np.floor(tile_end_cord[0] / level_resize))

    return m_xstart, m_xend, m_ystart, m_yend


cur_wd = '/fh/scratch/delete90/etzioni_r/lucas_l/michael_project/mutation_pred/'
save_location = cur_wd + 'intermediate_data/cancer_prediction_results102824/'  
save_location2 = cur_wd + 'intermediate_data/cancer_prediction_results102824/tiles/'  
save_location3 = cur_wd + 'intermediate_data/cancer_prediction_results102824/cancer_pred_out/'  
mag_extract = 20 # do not change this, model trained at 250x250 at 20x
save_image_size = 250  # do not change this, model trained at 250x250 at 20x
pixel_overlap = 100  # specify the level of pixel overlap in your saved images
limit_bounds = True  # this is weird, dont change it
tiff_lvl =2 # low res pyramid level to grab
ft_model = True
model_path_m = cur_wd + 'models/cancer_detection_models/mets/ft_models/dlv3_2ep_2e4_update-07182023_RT_fine_tuned..pkl'
model_path_m_prior = cur_wd + 'models/cancer_detection_models/mets/dlv3_2ep_2e4_update-07182023_RT.pkl'
    
#model_path_l = cur_wd + 'models/cancer_detection_models/local/binary_mblntv3_25ep_lr1e5_wAug_MixUpLS_sz500_bs12_10x.pkl'
data_mut_path = cur_wd + 'data/MutationCalls/'
save_location4 = save_location3 + str(pixel_overlap) + 'and' + str(tiff_lvl)  
save_location6 = save_location2 + str(pixel_overlap) + 'and' + str(tiff_lvl) 


create_dir_if_not_exists(save_location)
create_dir_if_not_exists(save_location2)
create_dir_if_not_exists(save_location3)
create_dir_if_not_exists(save_location4)


# #load mutation site
# mut_site_df = pd.read_excel(data_mut_path + 'OPX_anatomic sites.xlsx')
# mut_site_df
# mets_ids = list(mut_site_df.loc[mut_site_df['Anatomic site']!= 'Prostate', 'OPX_Number'])
# local_ids = list(mut_site_df.loc[mut_site_df['Anatomic site']== 'Prostate', 'OPX_Number'])
# len(mets_ids)
# len(local_ids)


selected_ids = ['OPX_010', 'OPX_015', 'OPX_020',  'OPX_033',  'OPX_049', 'OPX_077', 'OPX_090', 'OPX_182', 'OPX_185', 'OPX_186', 'OPX_194']
for cur_id in selected_ids:
    print(cur_id)
    _file = cur_wd + "data/OPX/" + cur_id + ".tif"
    # cur_id = '(2017-0133) 15-B_A1-2'
    # _file = '/fh/scratch/delete90/haffner_m/user/scan_archives/Prostate/MDAnderson/CCola/all_slides/' + cur_id + '.svs'
    oslide = openslide.OpenSlide(_file)
    save_name = str(Path(os.path.basename(_file)).with_suffix(''))
    
    save_location5 = save_location4 + "/" + cur_id + "/" 
    create_dir_if_not_exists(save_location5)
    
    if ft_model == False:
        save_location5 = save_location5 + "/" + "prior_model" + "/"
        create_dir_if_not_exists(save_location5)
    
    
    #get local or mets
    # site = mut_site_df.loc[mut_site_df['OPX_Number'] == cur_id,'Anatomic site'].item()
    # print(site)
    #load pytorch model
    learn = load_learner(model_path_m,cpu=False) #all use mets model
    
    if ft_model == True:
        learn = load_learner(model_path_m,cpu=False) #all use mets model
    else: 
        learn = load_learner(model_path_m_prior,cpu=False) #all use prior mets model
    
    # if site == "Prostate":
    #     learn = load_learner(model_path_l,cpu=False)
    # else:
    #     learn = load_learner(model_path_m,cpu=False)
    
    
    #Load tile info 
    tile_info_df = pd.read_csv(save_location6 + "/" + save_name + ".csv")
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
        tiles, tile_lvls, physSize = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)
        
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
        colimg.save(os.path.join(save_location5, save_name + '_cancer_prob.jpeg'))
        
        # send to get tissue polygons
        print('detecting tissue')
        lvl_img = oslide.read_region((0, 0), tiff_lvl, slide_dim)
        tissue, he_mask = do_mask_original(lvl_img,lvl_resize)

        #for diff threshold
        binary_pred_thres_list = [0.7,0.6,0.5,0.4,0.3,0.2,0.1]
        for binary_pred_thres in binary_pred_thres_list:
            save_location7 = save_location5 + 'TH' + str(binary_pred_thres) + '/' 
            create_dir_if_not_exists(save_location7)
            
            #Binary classification
            binary_preds = cancer_mask2(slideimg,he_mask, binary_pred_thres) 
            
            #Output annotation
            polygons = tile_ROIS(mask_arr=binary_preds, lvl_resize=lvl_resize)
            slide_ROIS(polygons=polygons, mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                            savename=os.path.join(save_location7,save_name+'_cancer.json'), labels='AI_tumor', ref=[0,0], roi_color=-16711936)
            slide_ROIS(polygons=tissue, mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                            savename=os.path.join(save_location7,save_name+'_tissue.json'), labels='tissue', ref=[0,0], roi_color=-16770432)
            
            
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
            
            tile_info_df.to_csv(save_location7 + save_name + "_TILE_TUMOR_PERC.csv", index = False)
            
            
            #Grab tiles and plot
            tile_info_df_gt05 = tile_info_df.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
            for i in range(0,5): #top5
                cur_row = tile_info_df_gt05.iloc[i]
                cur_xy = cur_row['TILE_XY_INDEXES'].strip("()").split(", ")
                x ,y = int(cur_xy[0]) , int(cur_xy[1])
                tile_pull_ex = tiles.get_tile(tile_lvls.index(mag_extract), (x, y))
                tile_pull_ex = tile_pull_ex.resize(size=(save_image_size, save_image_size),resample=PIL.Image.LANCZOS) #resize
                tile_pull_ex.save(os.path.join(save_location7, "EX_TILE"  + str(i) + ".png"))

