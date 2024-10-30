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
warnings.filterwarnings("ignore")

def do_mask_original(img,lvl_resize):
    ''' create tissue mask '''
    # get he image and find tissue mask
    he = np.array(img)
    he = he[:, :, 0:3]
    heHSV = cv2.cvtColor(he, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(heHSV, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imagem = cv2.bitwise_not(thresh1)
    tissue_mask = morphology.binary_dilation(imagem, morphology.disk(radius=5))
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
    
def do_mask(img,lvl_resize):
    ''' create tissue mask '''
    # get he image and find tissue mask
    he = np.array(img)
    he = he[:, :, 0:3]
    heHSV = cv2.cvtColor(he, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(heHSV, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imagem = cv2.bitwise_not(thresh1)
    tissue_mask = morphology.binary_dilation(imagem, morphology.disk(radius=5))
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
            if (poly.length > 0) & (poly.is_valid): #Added is_valid to fix GEOSException: TopologyException: unable to assign free hole to a shell at 0 150524
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