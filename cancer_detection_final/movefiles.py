#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 12:37:53 2025

@author: jliu6
"""

import os
import shutil

# Path to folder1
parent_path = os.path.dirname('/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/pred_out_040725/TCGA_PRAD/uni2/TrainOL100_TestOL0_TFT0.9/FOLD0/MSI_POS/perf/')
folder1_path = os.path.join(parent_path, "GAMMA_4_ALPHA_1.0/")

# Start at folder1 and walk down until the last nested folder
current_path = folder1_path
subfolders = []


while True:
    entries = [entry for entry in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, entry))]
    if len(entries) != 1:
        break  # Stop if there's not exactly one subfolder (end of the chain)
    next_folder = entries[0]
    current_path = os.path.join(current_path, next_folder)
    subfolders.append(current_path)

subfolders[0]
# Move all the subfolders to the same level as folder1
for folder in subfolders:
    shutil.move(folder, parent_path)
    
    
shutil.move(subfolders[0], parent_path)
