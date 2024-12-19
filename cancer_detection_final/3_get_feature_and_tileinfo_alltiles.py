#!/usr/bin/env python
# coding: utf-8

#NOTE: use paimg9 env
import sys
import os
import numpy as np
import openslide
from fastai.vision.all import *
matplotlib.use('Agg')
import pandas as pd
import warnings
sys.path.insert(0, '../Utils/')
from Preprocessing import preprocess_mutation_data, preprocess_site_data, get_tile_representation
from Utils import generate_deepzoom_tiles
from Utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")

import ResNet as ResNet
import time

##################
#User Input
##################
feature_extraction_method = 'retccl'
PIXEL_OVERLAP = 0

##################
###### DIR  ######
##################
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
wsi_path = proj_dir + '/data/OPX/'
model_path = proj_dir + 'models/feature_extraction_models/' + feature_extraction_method + '/'
tile_path = proj_dir + 'intermediate_data/cancer_prediction_results110224/IMSIZE250_OL' + str(PIXEL_OVERLAP) + '/'
label_path = proj_dir + 'data/MutationCalls/'

##################
#Select GPU
##################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################################################################################
#Select IDS
############################################################################################################
selected_ids = ['OPX_207', 'OPX_208', 'OPX_209', 'OPX_210', 'OPX_211', 'OPX_212', 'OPX_213', 'OPX_214', 'OPX_215', 'OPX_216']
print(selected_ids)


################################################
#Load mutation label data
################################################
#Old Data
label_df1 = pd.read_excel(label_path + "OPX_FH_original.xlsx")

#newly added data
label_df2 = pd.read_excel(label_path + "MMR_OPX_deidentified.xlsx")
label_df2.rename(columns = {'HR/DDR (BRCA1, BRCA2, ATM, CHEK2, PALB2, BAP1, BARD1, RAD51C, RAD51D, FANCA, FANCD2, MRE11A, ATR, NBN, FANCM, FANCG)': 
                         'MMR (MSH2, MSH6, PMS2, MLH1, MSH3, MLH3, EPCAM)2'}, inplace = True)
label_df2 = label_df2.loc[pd.isna(label_df2['OPX_Number']) == False] #remove NA
label_df2 = label_df2[label_df1.columns] #only keep the same columns as old data

#Combined
label_df = pd.concat([label_df1, label_df2])
label_df = preprocess_mutation_data(label_df)
label_df.reset_index(drop=True, inplace=True)


################################################
#Load Site data
################################################
#Old data
site_df1 = pd.read_excel(label_path + "OPX_anatomic sites.xlsx")

#New data
site_df2 =pd.DataFrame({'OPX_Number': selected_ids,
                       'Bx Type': np.nan,
                       'Anatomic site': np.nan,
                       'Notes': np.nan})

#Combined
site_df = pd.concat([site_df1, site_df2])
site_df.reset_index(drop=True, inplace=True)
site_df = preprocess_site_data(site_df)


############################################################################################################
#Load tile info for selected_ids
############################################################################################################
#All available IDs
opx_ids = [x.replace('.tif','') for x in os.listdir(tile_path) if 'OPX' in x] #207
opx_ids.sort()
tile_info_list = []
for cur_id in opx_ids:
    cur_tile_info_df = pd.read_csv(os.path.join(tile_path,cur_id,cur_id + "_tiles.csv"))
    tile_info_list.append(cur_tile_info_df)
all_tile_info_df = pd.concat(tile_info_list)
print(all_tile_info_df.shape) #3375102 tiles in total

#Print stats
tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
print("Total OPX IDs in tile path: ", len(set(all_tile_info_df['SAMPLE_ID']))) #3375102 tiles in total
print("Max # tile/per pt:", tile_counts.max())
print("Min # tile/per pt:", tile_counts.min())
print("Median # tile/per pt:", tile_counts.median())



############################################################################################################
#Combine all info
############################################################################################################
all_comb_df = all_tile_info_df.merge(label_df, on = ['SAMPLE_ID'])
all_comb_df = all_comb_df.merge(site_df, on = ['SAMPLE_ID'])
all_comb_df = all_comb_df.loc[all_comb_df['SAMPLE_ID'].isin(selected_ids)] #filter IDs


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
#For each patient tile, get representation
############################################################################################################
ct = 0 
for cur_id in selected_ids:
    print(cur_id)

    if ct % 10 == 0: print(ct)

    #Load slide
    _file = wsi_path + cur_id + ".tif"
    oslide = openslide.OpenSlide(_file)
    save_name = str(Path(os.path.basename(_file)).with_suffix(''))
    
    #Generate tiles
    tiles, tile_lvls, physSize, base_mag = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)
    
    #Get tile info
    comb_df = all_comb_df.loc[all_comb_df['SAMPLE_ID'] == cur_id]
    
    
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
    save_name = save_location + 'features_alltiles_' + pretrain_model_name + '.h5'
    feature_df.to_hdf(save_name, key='feature', mode='w')
    comb_df.to_hdf(save_name, key='tile_info', mode='a')

    ct += 1


