# import sys
# import os
# import numpy as np
# import cv2
# import openslide
# from openslide import open_slide
# from openslide.deepzoom import DeepZoomGenerator
# import xml.etree.ElementTree as ET
# from xml.dom import minidom
# import geojson
# import argparse
# import matplotlib.pyplot as plt
import fastai
from fastai.vision.all import *
# import PIL
# matplotlib.use('Agg')
# import pandas as pd
# import datetime
# from skimage import draw, measure, morphology, filters
# from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
# from shapely.ops import cascaded_union, unary_union
# import json
# import shapely
# import warnings
# from scipy import ndimage
# sys.path.insert(0, '../Utils/')
# from Utils import generate_deepzoom_tiles, extract_tile_start_end_coords
# from Utils import do_mask_original,check_tissue,whitespace_check
# from Utils import slide_ROIS
# from Utils import get_downsample_factor, get_image_at_target_mag
# from Utils import create_dir_if_not_exists
# from Utils import get_map_startend
# from Utils import cancer_mask_fix_res, tile_ROIS, check_any_invalid_poly, make_valid_poly
# warnings.filterwarnings("ignore")

import torch
import torchvision
#import semtorch
import sys
print(sys.version)
print(torch.__version__)
print(torchvision.__version__)
#print(semtorch.__version__)
print(fastai.__version__)

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



if ft_model == True:
    learn = load_learner(model_path + 'ft_models/dlv3_2ep_2e4_update-07182023_RT_fine_tuned..pkl',cpu=False) #all use mets model
    save_location = save_location + "ft_model" + "/"
    create_dir_if_not_exists(save_location)
else:
    learn = load_learner(model_path + 'dlv3_2ep_2e4_update-07182023_RT.pkl',cpu=False) #all use prior mets model
    save_location = save_location + "prior_model" + "/"
    create_dir_if_not_exists(save_location)