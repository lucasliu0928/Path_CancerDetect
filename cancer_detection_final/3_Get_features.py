#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#NOTE: use paimg1 env, the retccl one has package issue with torchvision
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
from Preprocessing import preprocess_mutation_data, preprocess_site_data
from Utils import generate_deepzoom_tiles, extract_tile_start_end_coords
from Utils import do_mask_original,check_tissue,whitespace_check
from Utils import slide_ROIS
from Utils import get_downsample_factor, get_image_at_target_mag
from Utils import create_dir_if_not_exists
from Utils import get_map_startend
from Utils import cancer_mask_fix_res, tile_ROIS, check_any_invalid_poly, make_valid_poly
warnings.filterwarnings("ignore")

import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import ResNet as ResNet
from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import time
from sklearn.model_selection import KFold
import numpy as np

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)


class get_tile_representation(Dataset):
    def __init__(self, tile_info, deepzoom_tiles, tile_levels, pretrain_model):
        super().__init__()
        self.transform = trnsfrms_val
        self.tile_info = tile_info
        self.deepzoom_tiles = deepzoom_tiles
        self.tile_levels = tile_levels
        self.mag_extract = list(set(tile_info['MAG_EXTRACT']))[0]
        self.save_image_size = list(set(tile_info['SAVE_IMAGE_SIZE']))[0]
        self.pretrain_model = pretrain_model

    def __getitem__(self, idx):
        #Get x, y index
        tile_ind = self.tile_info['TILE_XY_INDEXES'].iloc[idx].strip("()").split(", ")
        x ,y = int(tile_ind[0]) , int(tile_ind[1])

        #Pull tiles
        tile_pull = self.deepzoom_tiles.get_tile(self.tile_levels.index(self.mag_extract), (x, y))
        tile_pull = tile_pull.resize(size=(self.save_image_size, self.save_image_size),resample=PIL.Image.LANCZOS) #resize

        #Get features
        tile_pull_trns = self.transform(tile_pull)
        tile_pull_trns = tile_pull_trns.unsqueeze(0)  # Adds a dimension at the 0th index

        #use model to get feature
        self.pretrain_model.eval()
        with torch.no_grad():
            features = self.pretrain_model(tile_pull_trns)
            features = features.cpu().numpy()

        return tile_pull,features

##################
###### DIR  ######
##################
proj_dir = '/fh/scratch/delete90/etzioni_r/lucas_l/michael_project/mutation_pred/'
wsi_path = proj_dir + '/data/OPX/'
label_path = proj_dir + 'data/MutationCalls/'
model_path = proj_dir + 'models/feature_extraction_models/'
tile_path = proj_dir + 'intermediate_data/cancer_prediction_results110224/IMSIZE250_OL100/'
ft_ids_path =  proj_dir + 'intermediate_data/cd_finetune/cancer_detection_training/' #the ID used for fine-tuning cancer detection model, needs to be excluded from mutation study
pretrain_model_name = 'retccl'


##################
#Select GPU
##################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





############################################################################################################
#Select IDS
############################################################################################################

#All available IDs
opx_ids = [x.replace('.tif','') for x in os.listdir(wsi_path)] #207
opx_ids.sort()

#Get IDs that are in FT train or already processed to exclude 
ft_ids_df = pd.read_csv(ft_ids_path + 'all_tumor_fraction_info.csv')
ft_train_ids = list(ft_ids_df.loc[ft_ids_df['Train_OR_Test'] == 'Train','sample_id'])

#OPX_182 –Exclude Possible Colon AdenoCa 
toexclude_ids = ft_train_ids + ['OPX_182']  #25


#Exclude ids in ft_train or processed
selected_ids = [x for x in opx_ids if x not in toexclude_ids] #199

################################################
#Load mutation label data
################################################
label_df = pd.read_excel(label_path + "OPX_FH_original.xlsx")
label_df = preprocess_mutation_data(label_df)
label_df = label_df.loc[label_df['SAMPLE_ID'].isin(selected_ids)] #filter IDs


################################################
#Load Site data
################################################
site_df = pd.read_excel(label_path + "OPX_anatomic sites.xlsx")
site_df = preprocess_site_data(site_df)
site_df = site_df.loc[site_df['SAMPLE_ID'].isin(selected_ids)] #filter IDs


############################################################################################################
#Load tile info for selected_ids
############################################################################################################
tile_info_list = []
for cur_id in selected_ids:
    cur_tile_info_df = pd.read_csv(os.path.join(tile_path,cur_id,"ft_model/",cur_id + "_TILE_TUMOR_PERC.csv"))
    tile_info_list.append(cur_tile_info_df)
all_tile_info_df = pd.concat(tile_info_list)
print(len(set(all_tile_info_df['SAMPLE_ID']))) #199

print(all_tile_info_df.shape) #3375102 tiles in total

#Print stats
tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
print("Max # tile/per pt:", tile_counts.max())
print("Min # tile/per pt:", tile_counts.min())
print("Median # tile/per pt:", tile_counts.median())


############################################################################################################
#Combine all info
############################################################################################################
all_comb_df = all_tile_info_df.merge(label_df, on = ['SAMPLE_ID'])
all_comb_df = all_comb_df.merge(site_df, on = ['SAMPLE_ID'])


mag_extract = list(set(all_comb_df['MAG_EXTRACT']))[0]
save_image_size = list(set(all_comb_df['SAVE_IMAGE_SIZE']))[0]
pixel_overlap = list(set(all_comb_df['PIXEL_OVERLAP']))[0]
limit_bounds =   list(set(all_comb_df['LIMIT_BOUNDS']))[0]


############################################################################################################
# Load Pretrained representation model
############################################################################################################
model = ResNet.resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
pretext_model = torch.load(model_path + 'best_ckpt.pth',map_location=torch.device(device))
model.fc = nn.Identity()
model.load_state_dict(pretext_model, strict=True)


############################################################################################################
#Get Train and test IDs, 80% - 20%
############################################################################################################
# Number of folds
n_splits = 5

# Initialize KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Generate the folds
train_ids_folds = []
test_ids_folds = []
for fold, (train_index, test_index) in enumerate(kf.split(selected_ids)):
    train_ids_folds.append([selected_ids[i] for i in train_index])
    test_ids_folds.append([selected_ids[i] for i in test_index])

selected_fold = 0
train_ids = train_ids_folds[selected_fold]
test_ids = test_ids_folds[selected_fold]

############################################################################################################
#For each patient tile, get representation
#TODOSelect tiles, 200 tiles per patients for training IDs, all tiles for testing IDs
############################################################################################################
ct = 0 
for cur_id in selected_ids:

    if ct % 10 == 0: print(ct)

    #Load slide
    _file = wsi_path + cur_id + ".tif"
    oslide = openslide.OpenSlide(_file)
    save_name = str(Path(os.path.basename(_file)).with_suffix(''))
    
    #Generate tiles
    tiles, tile_lvls, physSize, base_mag = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)
    
    #Get tile info
    comb_df = all_comb_df.loc[all_comb_df['SAMPLE_ID'] == cur_id]
    #sort by tumor fraction
    comb_df = comb_df.sort_values(by = ['TUMOR_PIXEL_PERC'], ascending = False) 
    
    
    #Select tiles
    sample_size = 200
    if comb_df.shape[0] > sample_size:
        comb_df = comb_df.iloc[0:sample_size,]
    else: #upsampling
        comb_df = comb_df.sample(n=sample_size, replace=True, random_state=42)
    comb_df = comb_df.reset_index(drop = True)
    
    
    #Grab tile 
    tile_img = get_tile_representation(comb_df, tiles, tile_lvls, model)
    
    #Get feature
    start_time = time.time()
    feature_list = [tile_img[i][1] for i in range(comb_df.shape[0])]
    print("--- %s seconds ---" % (time.time() - start_time))
    
    feature_df = np.concatenate(feature_list)
    feature_df = pd.DataFrame(feature_df)
    
    
    save_location = tile_path + cur_id + '/' + 'features/'
    create_dir_if_not_exists(save_location)
    save_name = save_location + 'train_features_' + pretrain_model_name + '.h5'
    feature_df.to_hdf(save_name, key='feature', mode='w')
    comb_df.to_hdf(save_name, key='tile_info', mode='a')

    ct += 1

