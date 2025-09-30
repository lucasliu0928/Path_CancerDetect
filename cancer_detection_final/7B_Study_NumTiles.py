#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 20:11:34 2025

@author: jliu6
"""

import os
import sys
import argparse
import time
import torch
import pandas as pd
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists
from misc_utils import count_num_tiles, plot_n_tiles_by_labels
 


# source ~/.bashrc
# conda activate mil
############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--tumor_frac', default= 0.0, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath, virchow2')
parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device name: cuda:0,1,2,3')
            
if __name__ == '__main__':
    
    args = parser.parse_args()

    ##################f
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    
    ######################
    #Create output-dir
    ######################
    outdir_0 = os.path.join(proj_dir, "intermediate_data","8_discrip_stats","tile_counts")
    create_dir_if_not_exists(outdir_0)
            
    
    ###################################
    #Load
    ###################################
    cohorts = [
        "z_nostnorm_OPX",
        "z_nostnorm_TCGA_PRAD",
        "z_nostnorm_Neptune",
        "OPX",
        "TCGA_PRAD",
        "Neptune"
    ]
    
    data = {}
    
    for cohort_name in cohorts:
        start_time = time.time()
        base_path = os.path.join(data_dir, 
                                 cohort_name, 
                                 "IMSIZE250_{}", 
                                 f'feature_{args.fe_method}', 
                                 f"TFT{str(args.tumor_frac)}", 
                                 f'{cohort_name}_data.pth')
    
        data[f'{cohort_name}_ol100'] = torch.load(base_path.format("OL100"), weights_only = False)
        data[f'{cohort_name}_ol0'] = torch.load(base_path.format("OL0"),weights_only = False)
        
        elapsed_time = time.time() - start_time
        print(f"Time taken for {cohort_name}: {elapsed_time/60:.2f} minutes")
    
    opx_ol100_nst = data['z_nostnorm_OPX_ol100']
    opx_ol0_nst = data['z_nostnorm_OPX_ol0']
    tcga_ol100_nst = data['z_nostnorm_TCGA_PRAD_ol100']
    tcga_ol0_nst = data['z_nostnorm_TCGA_PRAD_ol0']
    nep_ol100_nst = data['z_nostnorm_Neptune_ol100']
    nep_ol0_nst = data['z_nostnorm_Neptune_ol0']
    
    opx_ol100 = data['OPX_ol100']
    opx_ol0 = data['OPX_ol0']
    tcga_ol100 = data['TCGA_PRAD_ol100']
    tcga_ol0 = data['TCGA_PRAD_ol0']
    nep_ol100 = data['Neptune_ol100']
    nep_ol0= data['Neptune_ol0']
    
        
    #########################################################################################
    #Count tiles
    #########################################################################################
    #Count AVG , MAX, MIN number of tiles
    nep_count_df, nep_sample_count_df = count_num_tiles(nep_ol0,'Nep')
    tcga_count_df, tcga_sample_count_df = count_num_tiles(tcga_ol0,'TCGA')
    opx_count_df, opx_sample_count_df = count_num_tiles(opx_ol0,'OPX')
    all_count_df = pd.concat([opx_count_df, tcga_count_df, nep_count_df])
    all_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0.csv"))
    nep_sample_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0_nep.csv"))
    tcga_sample_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0_tcga.csv"))
    opx_sample_count_df.to_csv(os.path.join(outdir_0,"tile_n_counts_OL0_opx.csv"))
    
    # Plot bar charts
    plot_n_tiles_by_labels(opx_sample_count_df, 
                           label_cols=None, 
                           value_col="N_TILES",
                           agg="mean", 
                           save_path=os.path.join(outdir_0, "tile_bar_OL0_OPX.png"))
    plot_n_tiles_by_labels(nep_sample_count_df, 
                           label_cols=None, 
                           value_col="N_TILES",
                           agg="mean", 
                           save_path=os.path.join(outdir_0, "tile_bar_OL0_NEP.png"))
    plot_n_tiles_by_labels(tcga_sample_count_df, 
                           label_cols=None, 
                           value_col="N_TILES",
                           agg="mean", 
                           save_path=os.path.join(outdir_0, "tile_bar_OL0_TCGA.png"))

    
    #Count AVG , MAX, MIN number of tiles
    nep_count_df, nep_sample_count_df = count_num_tiles(nep_ol100,'Nep')
    tcga_count_df, tcga_sample_count_df = count_num_tiles(tcga_ol100,'TCGA')
    opx_count_df, opx_sample_count_df = count_num_tiles(opx_ol100,'OPX')
    all_count_df = pd.concat([opx_count_df, tcga_count_df, nep_count_df])
    all_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100.csv"))
    nep_sample_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100_nep.csv"))
    tcga_sample_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100_tcga.csv"))
    opx_sample_count_df.to_csv(os.path.join(outdir_0, "tile_n_counts_OL100_opx.csv"))
    plot_n_tiles_by_labels(opx_sample_count_df, 
                           label_cols=None, 
                           value_col="N_TILES",
                           agg="mean", 
                           save_path=os.path.join(outdir_0, "tile_bar_OL100_OPX.png"))
    plot_n_tiles_by_labels(nep_sample_count_df, 
                           label_cols=None, 
                           value_col="N_TILES",
                           agg="mean", 
                           save_path=os.path.join(outdir_0, "tile_bar_OL100_NEP.png"))
    plot_n_tiles_by_labels(tcga_sample_count_df, 
                           label_cols=None, 
                           value_col="N_TILES",
                           agg="mean", 
                           save_path=os.path.join(outdir_0, "tile_bar_OL100_TCGA.png"))
    

