#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:22:30 2025

@author: jliu6
"""

import random
import torch
import numpy as np
import os
import pandas as pd
from PIL import ImageCms, Image
import sys
import h5py

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")
        
        
        
def save_hdf5(output_path, asset_dict, attr_dict=None, mode='w'):
    '''
    saves data to h5py format
    :param output_path: path to output h5py file
    :param asset_dict: dictionary containing the data
    :param attr_dict:
    :param mode:
    :return:
    '''
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        #print(key)
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape,
                                        dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    print(key)
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
    file.close()
    print("finished writing to hdf5 file...")
    return output_path
