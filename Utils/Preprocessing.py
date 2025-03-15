#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 03:27:29 2024

@author: jliu6
"""
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import PIL
from Utils import extract_tile_start_end_coords_tma

def preprocess_mutation_data(indata, id_col = 'OPX_Number'):

    #Rename ID col
    indata.rename(columns = {id_col: 'SAMPLE_ID'}, inplace = True)

    #Recode, 1: mutation, 0: no mutation
    indata.iloc[:, 5:] = indata.iloc[:, 5:].notna().astype(int)
    
    #Recode MSI,1: POS, 0: NEG/NA
    indata['MSI_POS'] = pd.NA
    cond = indata['MSI (POS/NEG)'] == 'POS'
    indata.loc[cond,'MSI_POS'] = 1
    indata.loc[~cond,'MSI_POS'] = 0
    
    #Recode TMB:  1: High or Intermediate, 0: LOW
    indata['TMB_HIGHorINTERMEDITATE'] = pd.NA
    cond = indata['TMB (HIGH/LOW/INTERMEDIATE)'].isin(['INTERMEDITATE','HIGH'])
    indata.loc[cond,'TMB_HIGHorINTERMEDITATE'] = 1
    indata.loc[~cond,'TMB_HIGHorINTERMEDITATE'] = 0

    #Drop extra column
    indata = indata.drop(columns = ['Results',
                           'Limited Study (low tumor content/quality), YES/NO',
                           'TMB (HIGH/LOW/INTERMEDIATE)',
                           'MSI (POS/NEG)'])

    return indata


def preprocess_site_data(indata, id_col = 'OPX_Number'):

    #Rename ID col
    indata.rename(columns = {id_col: 'SAMPLE_ID'}, inplace = True)

    #Recode
    indata['SITE_LOCAL'] = pd.NA
    cond = indata['Anatomic site'] == 'Prostate'
    indata.loc[cond,'SITE_LOCAL'] = 1
    indata.loc[~cond,'SITE_LOCAL'] = 0
    
    return indata



def transform_functions(pretrain_model_name = 'uni2'):

    if pretrain_model_name == 'retccl':
        trnsfrms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.485, 0.456, 0.406), std =(0.229, 0.224, 0.225))
            ]
        )
    elif pretrain_model_name == 'uni' or pretrain_model_name == 'uni2':
        trnsfrms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    elif pretrain_model_name == 'prov_gigapath':
        trnsfrms = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    return trnsfrms

class get_tile_representation(Dataset):
    def __init__(self, tile_info, deepzoom_tiles, tile_levels, pretrain_model_name, pretrain_model, device):
        super().__init__()
        self.transform = transform_functions(pretrain_model_name)
        self.tile_info = tile_info
        self.deepzoom_tiles = deepzoom_tiles
        self.tile_levels = tile_levels
        self.mag_extract = list(set(tile_info['MAG_EXTRACT']))[0]
        self.save_image_size = list(set(tile_info['SAVE_IMAGE_SIZE']))[0]
        self.pretrain_model = pretrain_model
        self.device = device

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
            tile_pull_trns = tile_pull_trns.to(self.device)
            self.pretrain_model = self.pretrain_model.to(self.device)
            features = self.pretrain_model(tile_pull_trns)
            features = features.cpu().numpy()

        return tile_pull,features



class get_tile_representation_tma(Dataset):
    def __init__(self, tile_info, tma, pretrain_model_name, pretrain_model, device):
        super().__init__()
        self.transform = transform_functions(pretrain_model_name)
        self.tile_info = tile_info
        self.save_image_size = list(set(tile_info['SAVE_IMAGE_SIZE']))[0]
        self.pixel_overlap = list(set(tile_info['PIXEL_OVERLAP']))[0]
        self.pretrain_model = pretrain_model
        self.tma = tma
        self.device = device

    def __getitem__(self, idx):
        #Get x, y index
        tile_ind = self.tile_info['TILE_XY_INDEXES'].iloc[idx].strip("()").split(", ")
        x ,y = int(tile_ind[0]) , int(tile_ind[1])

        #Pull tiles
        tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords_tma(x, y, tile_size = self.save_image_size, overlap = self.pixel_overlap)
        tile_pull = self.tma.crop(box=(tile_starts[0], tile_starts[1], tile_ends[0], tile_ends[1])) 
        tile_pull = tile_pull.resize(size=(self.save_image_size, self.save_image_size),resample=PIL.Image.LANCZOS) #resize
        tile_pull = tile_pull.convert('RGB')

        #Get features
        tile_pull_trns = self.transform(tile_pull)
        tile_pull_trns = tile_pull_trns.unsqueeze(0)  # Adds a dimension at the 0th index

        #use model to get feature
        self.pretrain_model.eval()
        with torch.no_grad():
            tile_pull_trns = tile_pull_trns.to(self.device)
            self.pretrain_model = self.pretrain_model.to(self.device)
            features = self.pretrain_model(tile_pull_trns)
            features = features.cpu().numpy()

        return tile_pull,features