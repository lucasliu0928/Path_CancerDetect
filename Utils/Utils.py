#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 03:27:29 2024

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def do_mask_original(img,lvl_resize, rad = 5):
    ''' create tissue mask '''
    # get he image and find tissue mask
    he = np.array(img)
    he = he[:, :, 0:3]
    heHSV = cv2.cvtColor(he, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(heHSV, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imagem = cv2.bitwise_not(thresh1)
    tissue_mask = morphology.binary_dilation(imagem, morphology.disk(radius=rad))
    tissue_mask = morphology.remove_small_objects(tissue_mask, 1000)
    tissue_mask = ndimage.binary_fill_holes(tissue_mask)

    # create polygons for faster tiling in cancer detection step
    polygons = []
    contours, hier = cv2.findContours(tissue_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cvals = contour.transpose(0, 2, 1)
        cvals = np.reshape(cvals, (cvals.shape[0], 2))
        cvals = cvals.astype('float64')
        for i in range(len(cvals)):
            cvals[i][0] = np.round(cvals[i][0]*lvl_resize,2)
            cvals[i][1] = np.round(cvals[i][1]*lvl_resize,2)
        try:
            poly = Polygon(cvals)
            if poly.length > 0: 
                polygons.append(Polygon(poly.exterior))
        except:
            pass
    tissue = unary_union(polygons)
    while not tissue.is_valid:
        print('pred_union is invalid, buffering...')
        tissue = tissue.buffer(0)

    return tissue, tissue_mask

def check_tissue(tile_starts, tile_ends, roi):
    ''' checks if tile in tissue '''
    tile_box = [tile_starts[0], tile_starts[1]], [tile_starts[0], tile_ends[1]], [tile_ends[0], tile_starts[1]], [tile_ends[0], tile_ends[1]]
    tile_box = list(tile_box)
    tile_box = MultiPoint(tile_box).convex_hull
    ov = 0  # initialize
    if tile_box.intersects(roi):
        ov_reg = tile_box.intersection(roi)
        ov += ov_reg.area / tile_box.area

    return ov

def whitespace_check(im):
    ''' checks if meets whitespace requirement'''
    bw = im.convert('L')
    bw = np.array(bw)
    bw = bw.astype('float')
    bw = bw / 255
    prop_ws = (bw > 0.8).sum() / (bw > 0).sum()
    return prop_ws

def cancer_mask(pred_image,hetissue):
    ''' smooth cancer map and find high probability areas '''
    # get pred image, mask at 50% and find regions
    preds = np.array(pred_image).astype('float64')
    preds = preds / 255
    preds[hetissue < 1] = 0
    preds = filters.gaussian(preds,sigma=10)
    preds_mask = np.zeros(preds.shape)
    preds_mask[preds > 0.8] = 1
    preds_mask = morphology.binary_dilation(preds_mask, morphology.disk(radius=2))
    preds_mask = morphology.binary_erosion(preds_mask, morphology.disk(radius=2))
    preds_mask = ndimage.binary_fill_holes(preds_mask)
    labels = measure.label(preds_mask)
    regions = measure.regionprops(labels,preds)
    for reg in regions:
        if reg.max_intensity<0.6:
            labels[labels==reg.label]=0
    labels[labels>0]=1
    return labels

def cancer_mask2(pred_image,hetissue, thres):
    ''' smooth cancer map and find high probability areas '''
    # get pred image, mask at 50% and find regions
    preds = np.array(pred_image).astype('float64')
    preds = preds / 255
    preds[hetissue < 1] = 0
    preds = filters.gaussian(preds,sigma=10)
    preds_mask = np.zeros(preds.shape)
    preds_mask[preds > thres] = 1
    preds_mask = morphology.binary_dilation(preds_mask, morphology.disk(radius=2))
    preds_mask = morphology.binary_erosion(preds_mask, morphology.disk(radius=2))
    preds_mask = ndimage.binary_fill_holes(preds_mask)
    labels = measure.label(preds_mask)
    regions = measure.regionprops(labels,preds)
    for reg in regions:
        if reg.max_intensity<0.6:
            labels[labels==reg.label]=0
    labels[labels>0]=1
    return labels

def cancer_mask_fix_res(preds,hetissue, thres):
    ''' smooth cancer map and find high probability areas '''
    preds[hetissue < 1] = 0 #If tissue map value < 1, then preds = 0
    preds_mask = np.zeros(preds.shape)
    preds_mask[preds > thres] = 1
    preds_mask = morphology.binary_dilation(preds_mask, morphology.disk(radius=2))
    preds_mask = morphology.binary_erosion(preds_mask, morphology.disk(radius=2))
    preds_mask = ndimage.binary_fill_holes(preds_mask)
    return preds_mask
    
def slide_ROIS(polygons,mpp,savename,labels,ref,roi_color):
    ''' generate geojson from polygons '''
    all_polys = unary_union(polygons)
    final_polys = []
    if all_polys.type == 'Polygon':
        poly = all_polys
        polypoints = poly.exterior.xy
        polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
        polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
        newpoly = Polygon(zip(polyx, polyy))
        if newpoly.area*mpp*mpp > 0.1:
            final_polys.append(newpoly)

    else:
        for poly in all_polys.geoms:
            # print(poly)
            if poly.type == 'Polygon':
                polypoints = poly.exterior.xy
                polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
                polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
                newpoly = Polygon(zip(polyx, polyy))
                if newpoly.area*mpp*mpp > 0.1:
                    final_polys.append(newpoly)
            if poly.type == 'MultiPolygon':
                for roii in poly.geoms:
                    polypoints = roii.exterior.xy
                    polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
                    polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
                    newpoly = Polygon(zip(polyx, polyy))
                    if newpoly.area*mpp*mpp > 0.1:
                        final_polys.append(newpoly)

    final_shape = unary_union(final_polys)
    try:
        trythis = '['
        for i in range(0, len(final_shape)):
            trythis += json.dumps(
                {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape[i]),
                "properties": {"classification": {"name": labels, "colorRGB": roi_color}, "isLocked": False,
                                "measurements": []}}, indent=4)
            if i < len(final_shape) - 1:
                trythis += ','
        trythis += ']'
    except:
        trythis = '['
        trythis += json.dumps(
            {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape),
            "properties": {"classification": {"name": labels, "colorRGB": roi_color}, "isLocked": False,
                            "measurements": []}}, indent=4)
        trythis += ']'

    with open(savename, 'w') as outfile:
        outfile.write(trythis)
    return

def tile_ROIS(mask_arr,lvl_resize):
    ''' get cancer polygons '''
    polygons = []
    contours, hier = cv2.findContours(mask_arr.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cvals = contour.transpose(0, 2, 1)
        cvals = np.reshape(cvals, (cvals.shape[0], 2))
        cvals = cvals.astype('float64')
        for i in range(len(cvals)):
            cvals[i][0] = np.round(cvals[i][0]*lvl_resize,2)
            cvals[i][1] = np.round(cvals[i][1]*lvl_resize,2)
        try:
            poly = Polygon(cvals)
            if poly.length > 0:
                polygons.append(Polygon(poly.exterior))
        except:
            pass

    return polygons

def get_y(row): return row['label'].split(" ")


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

    return tiles, tile_lvls, physSize, base_mag


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
    r'''
    Note: The maps has x and y swaped relatived to the original slides
    #Change from floor to ceil 11/06/24
    '''
    m_xstart = int(np.ceil(tile_start_cord[1] / level_resize))
    m_xend = int(np.ceil(tile_end_cord[1] / level_resize))
    m_ystart = int(np.ceil(tile_start_cord[0] / level_resize))
    m_yend = int(np.ceil(tile_end_cord[0] / level_resize))

    return m_xstart, m_xend, m_ystart, m_yend


def get_downsample_factor(base_magnification, target_magnification):
    
    ds_factor =(base_magnification/target_magnification) #downsample factor
    
    return ds_factor


def get_image_at_target_mag(slide_obj,level0_width, level0_height,downsample_factor):
    getLvl = slide_obj.get_best_level_for_downsample(downsample_factor)
    img_pull = slide_obj.read_region((0, 0), getLvl, slide_obj.level_dimensions[getLvl]) #image at level closed to target_magnification
    img_pull = img_pull.resize(size=(int(np.ceil(level0_width/downsample_factor)),int(np.ceil(level0_height/downsample_factor))),resample=PIL.Image.LANCZOS) #resize to actual target_magnification

    return img_pull

def check_any_invalid_poly(input_polygons):
    invalid_poly = [x for x in input_polygons if not x.is_valid]
    return invalid_poly     
    
def make_valid_poly(input_polygons, buff_value = 4):
    r'''
    For invalid polygon error, add buffer to make the polygon valid:
    '''

    smooth_polygon = []
    for poly in input_polygons:
        if not poly.is_valid:
            buffered_poly = poly.buffer(buff_value)
        else:
            buffered_poly = poly
        smooth_polygon.append(buffered_poly)
    return smooth_polygon


# Min-max normalization function
def minmax_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def log_message(message, file_name):
    """Logs a message to the specified file with a timestamp."""
    with open(file_name, 'a') as file:  # 'a' mode appends to the file
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file.write(f'{timestamp} {message}\n')



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def convert_img(in_img):
    srgb_profile = ImageCms.createProfile("sRGB")
    converted_img = ImageCms.profileToProfile(in_img, srgb_profile, srgb_profile)

    return converted_img


def count_label(label_list, selected_label_names, cohort_name):
    
    label_df = pd.DataFrame(np.concatenate(label_list))
    label_df.columns = selected_label_names

    count_list = []
    for l in selected_label_names:
        cur_count = pd.DataFrame(label_df[l].value_counts()).T
        cur_count.index = [l]
        count_list.append(cur_count)
    count_df = pd.concat(count_list)
    count_df.columns = ['N0','N1']
    count_df['Perc0'] = round(count_df['N0']/(count_df['N0'] + count_df['N1'])*100,1)
    count_df['Perc1'] = round(count_df['N1']/(count_df['N0'] + count_df['N1'])*100,1)

    count_df.columns =  cohort_name + '_' + count_df.columns

    return count_df