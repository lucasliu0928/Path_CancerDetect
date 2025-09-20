#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 20:22:27 2025

@author: jliu6
"""

import os
import shutil

def extract_pngs(src_dir, dest_dir):
    """
    Recursively finds all PNG files inside src_dir and copies them to dest_dir.
    
    Args:
        src_dir (str): Path to the source directory.
        dest_dir (str): Path to the destination directory where PNGs will be copied.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith("_low-res.png"):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)
                
                # Avoid overwriting if duplicate names exist
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(file)
                    i = 1
                    while os.path.exists(os.path.join(dest_dir, f"{base}_{i}{ext}")):
                        i += 1
                    dest_path = os.path.join(dest_dir, f"{base}_{i}{ext}")
                
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {src_path} -> {dest_path}")

# Example usage:
# Replace with your folder paths
proj_dir = "/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/"
source_folder = proj_dir + "1_tile_pulling/TCGA_PRAD/IMSIZE250_OL0/"
destination_folder = proj_dir + "check2/"

extract_pngs(source_folder, destination_folder)