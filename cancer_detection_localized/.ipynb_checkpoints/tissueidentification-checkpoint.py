#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 03:27:29 2024

@author: jliu6
@ENV: paimg1b
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
from scipy import ndimage
from Utils import *
warnings.filterwarnings("ignore")


cur_wd = '/fh/scratch/delete90/etzioni_r/lucas_l/michael_project/mutation_pred/'
save_location = cur_wd + 'intermediate_data/cancer_prediction_results0629/'  # args.save_location
save_location2 = cur_wd + 'intermediate_data/cancer_prediction_results0629/tiles/'  # args.save_location
save_location3 = cur_wd + 'intermediate_data/cancer_prediction_results0629/cancer_pred_out/'  # args.save_location
mag_extract = [20] # do not change this, model trained at 250x250 at 20x
save_image_size = 250  # do not change this, model trained at 250x250 at 20x
run_image_size = 250 # do not change this, model trained at 250x250 at 20x
pixel_overlap = 100  # specify the level of pixel overlap in your saved images
limit_bounds = True  # this is weird, dont change it
tiff_lvl = 2 # low res pyramid level to grab
model_path_m = cur_wd + 'code_s/cancer_detection/model/dlv3_2ep_2e4_update-07182023_RT.pkl'
model_path_l = cur_wd + 'code_s/cancer_detection_localized/model/binary_mblntv3_25ep_lr1e5_wAug_MixUpLS_sz500_bs12_10x.pkl'
data_mut_path = cur_wd + 'data/MutationCalls/'


if not os.path.exists(save_location):
    os.mkdir(save_location)

#load mutation site
mut_site_df = pd.read_excel(data_mut_path + 'OPX_anatomic sites.xlsx')
mets_ids = list(mut_site_df.loc[mut_site_df['Anatomic site']!= 'Prostate', 'OPX_Number'])
local_ids = list(mut_site_df.loc[mut_site_df['Anatomic site']== 'Prostate', 'OPX_Number'])
len(mets_ids)
len(local_ids)

#All files list
flist=sorted(glob.glob(cur_wd + '/data/OPX/*.tif'))

#Select IDs
selected_ids = mets_ids


for cur_id in selected_ids:
    print(cur_id)
    _file = cur_wd + "data/OPX/" + cur_id + ".tif"
    oslide = openslide.OpenSlide(_file)
    savnm = os.path.basename(_file)
    save_name = str(Path(savnm).with_suffix(''))

    save_location4 = save_location3 + cur_id + "/" 
    if not os.path.exists(save_location4):
        os.mkdir(save_location4)


    #get local or mets
    site = mut_site_df.loc[mut_site_df['OPX_Number'] == cur_id,'Anatomic site'].item()
    print(site)
    #load pytorch model
    if site == "Prostate":
        learn = load_learner(model_path_l,cpu=True)
    else:
        learn = load_learner(model_path_m,cpu=True)


    # this is physical microns per pixel
    acq_mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])

    # this is nearest multiple of 20 for base layer
    base_mag = int(20 * round(float(acq_mag) / 20))

    # this is how much we need to resample our physical patches for uniformity across studies
    physSize = round(save_image_size * acq_mag / base_mag)

    # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
    tiles = DeepZoomGenerator(oslide, tile_size=physSize - round(pixel_overlap * acq_mag / base_mag),
                              overlap=round(pixel_overlap * acq_mag / base_mag / 2),
                              limit_bounds=limit_bounds)


    # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
    tiles = DeepZoomGenerator(oslide, tile_size=physSize - round(pixel_overlap * acq_mag / base_mag),
                              overlap=round(pixel_overlap * acq_mag / base_mag / 2),
                              limit_bounds=limit_bounds)

    # calculate the effective magnification at each level of tiles, determined from base magnification
    tile_lvls = tuple(base_mag / (tiles._l_z_downsamples[i] * tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in range(0, tiles.level_count))

    # intermeadiate level for probability map
    lvl_img = oslide.read_region((0, 0), tiff_lvl, oslide.level_dimensions[tiff_lvl])
    lvl_resize = oslide.level_downsamples[tiff_lvl]
    lvl_img.save(os.path.join(save_location4,save_name+'_low-res.png'))



    # send to get tissue polygons
    print('detecting tissue')
    tissue, he_mask = do_mask(lvl_img,lvl_resize)


    x_map = np.zeros((lvl_img.size[1], lvl_img.size[0]), float)
    x_count = np.zeros((lvl_img.size[1], lvl_img.size[0]), float)


    x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(20)]
    x_tiles, y_tiles



    cur_save_loc = save_location2 + cur_id + "/" 
    if not os.path.exists(cur_save_loc):
        os.mkdir(cur_save_loc)

    print('starting inference')
    tile_count = 0
    all_tile_count = 0
    # pull tiles from levels specified by self.mag_extract
    for lvl in mag_extract:
        if lvl in tile_lvls:
            # pull tile info for level
            x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(lvl)]

            tile_df_list = []
            for y in range(0, y_tiles):

                if y % 10 == 0:
                    print(y)

                for x in range(0, x_tiles):
                    cur_outname = "x" + str(x) + "_y" + str(y)

                    # grab tile coordinates
                    tile_coords = tiles.get_tile_coordinates(tile_lvls.index(lvl), (x, y))
                    save_coords = str(tile_coords[0][0]) + "-" + str(tile_coords[0][1]) + "_" + '%.0f' % (tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][0]) + "-" + '%.0f' % (tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][1])
                    tile_ends = (int(tile_coords[0][0] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][0]),int(tile_coords[0][1] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][1]))

                    # check for tissue membership
                    tile_tiss = check_tissue(tile_starts=tile_coords[0], tile_ends=tile_ends,roi=tissue)
                    if tile_tiss > 0.9:
                        tile_pull = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                        tile_copy = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                        ws = whitespace_check(im=tile_pull)
                        if ws < 0.9:
                            all_tile_count += 1
                            tile_pull = tile_pull.resize(size=(save_image_size, save_image_size),resample=PIL.Image.LANCZOS)

                            #segmentation
                            tile_pull = np.array(tile_pull)

                            #Get tile posiition in x_count and x_map
                            map_xstart = int(np.floor(tile_coords[0][1] / lvl_resize))
                            map_xend = int(np.floor(tile_ends[1] / lvl_resize))
                            map_ystart = int(np.floor(tile_coords[0][0] / lvl_resize))
                            map_yend = int(np.floor(tile_ends[0] / lvl_resize))

                            with learn.no_bar():
                                inp, pred_class, pred_idx, outputs = learn.predict(tile_pull[:, :, 0:3], with_input=True)
                            outputs_np = outputs.numpy()
                            output = PIL.Image.fromarray(outputs_np[1])
                            #resize for low-res
                            output = output.resize(size=(map_xend - map_xstart, map_yend - map_ystart),resample=PIL.Image.LANCZOS)
                            output_np = np.array(output)


                            #Add the following because it will give error for images that hard to detect the egdes example: OPX006,007,010...
                            try: 
                                x_count[map_xstart:map_xend,map_ystart:map_yend] += 1
                                x_map[map_xstart:map_xend,map_ystart:map_yend] += output_np[1]
                            except:
                                pass

                            #Save tiles if pred > 0.8
                            cur_xmap = x_map[map_xstart:map_xend,map_ystart:map_yend]
                            cur_x_count = x_count[map_xstart:map_xend,map_ystart:map_yend]
                            cur_x_count = np.where(cur_x_count < 1, 1, cur_x_count)
                            cur_xmap = cur_xmap / cur_x_count

                            #Convert to image
                            tile_pull_RGB = tile_copy.convert('RGB')
                            tile_pull_RGB_np = np.array(tile_pull_RGB.resize((save_image_size, save_image_size), resample=PIL.Image.LANCZOS))

                            if np.any(cur_xmap > 0.8):
                                #print(x,y)
                                tile_count += 1
                                selected = 1
                                plt.imsave(cur_save_loc + cur_outname + "_GT08.png",tile_pull_RGB_np,dpi = 100)
                            else:
                                selected = 0
                                #plt.imsave(cur_save_loc + cur_outname + ".png",img_test,dpi = 100)

                            tile_df_list.append(pd.DataFrame({'SAMPLE_ID' : cur_id, 
                                                              'TILE_COOR' : cur_outname,
                                                              'SELECTED' : selected,
                                                              'Model_Selected': site}, index = [0]))

        else:
            print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")


    tile_df = pd.concat(tile_df_list)
    tile_df.reset_index(drop = True, inplace = True)
    tile_df.to_csv(save_location4 + cur_id + "tile_info.csv")





    print('post-processing')
    x_count = np.where(x_count < 1, 1, x_count)
    x_map = x_map / x_count
    slideimg = PIL.Image.fromarray(np.uint8(x_map * 255))
    slideimg = slideimg.convert('L')





    cmap = plt.get_cmap('jet')
    rgba_img = cmap(x_map)
    rgb_img = np.delete(rgba_img, 3, 2)
    colimg = PIL.Image.fromarray(np.uint8(rgb_img * 255))
    colimg.save(os.path.join(save_location4, save_name + '_cancer_prob.jpeg'))


    binary_preds = cancer_mask2(slideimg,he_mask, 0.0) 
    polygons = tile_ROIS(mask_arr=binary_preds, lvl_resize=lvl_resize)
    slide_ROIS(polygons=polygons, mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                    savename=os.path.join(save_location4,save_name+'_cancer.json'), labels='AI_tumor', ref=[0,0], roi_color=-16711936)
    slide_ROIS(polygons=tissue, mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                    savename=os.path.join(save_location4,save_name+'_tissue.json'), labels='tissue', ref=[0,0], roi_color=-16770432)

