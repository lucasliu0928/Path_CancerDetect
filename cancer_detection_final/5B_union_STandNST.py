#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 23:26:41 2025

@author: jliu6
"""

import os
import sys
import argparse
import time

sys.path.insert(0, '../Utils/')
from data_loader import merge_data_lists_h5, save_merged_samples_to_h5
from misc_utils import create_dir_if_not_exists
from data_loader import H5Cases



# source ~/.bashrc
# conda activate mil


############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Combine ST and NST data")
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath, virchow2')


            
if __name__ == '__main__':
    
    args = parser.parse_args()

    ##################
    ###### DIR  ######
    ##################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
    data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    out_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
    
    
    
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
    
    #updates: load h5
    for cohort_name in cohorts:
        start_time = time.time()
        base_path = os.path.join(data_dir, 
                                 cohort_name, 
                                 "IMSIZE250_{}", 
                                 f'feature_{args.fe_method}', 
                                 "TFT0.0", 
                                 f'{cohort_name}_data.h5')
        
        if cohort_name != "Neptune":
            data[f'{cohort_name}_ol100'] = H5Cases(os.path.join(base_path.format("OL100")))
        data[f'{cohort_name}_ol0']   = H5Cases(os.path.join(base_path.format("OL0")))
        
        elapsed_time = time.time() - start_time
        print(f"Time taken for {cohort_name}: {elapsed_time/60:.2f} minutes")
        
   
    #Non stain normd
    opx_ol100_nst = data['z_nostnorm_OPX_ol100']
    opx_ol0_nst = data['z_nostnorm_OPX_ol0']
    tcga_ol100_nst = data['z_nostnorm_TCGA_PRAD_ol100']
    tcga_ol0_nst = data['z_nostnorm_TCGA_PRAD_ol0']
    nep_ol0_nst = data['z_nostnorm_Neptune_ol0']
    
    #stain normd
    opx_ol100 = data['OPX_ol100']
    opx_ol0 = data['OPX_ol0']
    tcga_ol100 = data['TCGA_PRAD_ol100']
    tcga_ol0 = data['TCGA_PRAD_ol0']
    
    
    nep_ol0= data['Neptune_ol0']
    



    ##########################################################################################
    #Merge st norm and no st-norm
    ##########################################################################################
    #OPX
    outdir1 = os.path.join(out_dir, "Union_OPX","IMSIZE250_OL100","TFT0.0")
    create_dir_if_not_exists(outdir1)
    opx_union_ol100  = merge_data_lists_h5(opx_ol100_nst, opx_ol100, merge_type = 'union')
    save_merged_samples_to_h5(opx_union_ol100,os.path.join(outdir1,f'union_opx_feature_{args.fe_method}_ol100.h5'))
    
    outdir2 = os.path.join(out_dir, "Union_OPX","IMSIZE250_OL0","TFT0.0")
    create_dir_if_not_exists(outdir2)
    opx_union_ol0    = merge_data_lists_h5(opx_ol0_nst, opx_ol0, merge_type = 'union')
    save_merged_samples_to_h5(opx_union_ol0,os.path.join(outdir2,f'union_opx_feature_{args.fe_method}_ol0.h5'))
    
    #TCGA
    outdir1 = os.path.join(out_dir, "Union_TCGA_PRAD","IMSIZE250_OL100","TFT0.0")
    create_dir_if_not_exists(outdir1)
    tcga_union_ol100 = merge_data_lists_h5(tcga_ol100_nst, tcga_ol100, merge_type = 'union')
    save_merged_samples_to_h5(tcga_union_ol100,os.path.join(outdir1,f'union_TCGA_PRAD_feature_{args.fe_method}_ol100.h5'))
    
    outdir2 = os.path.join(out_dir, "Union_TCGA_PRAD","IMSIZE250_OL0","TFT0.0")
    create_dir_if_not_exists(outdir2)
    tcga_union_ol0   = merge_data_lists_h5(tcga_ol0_nst, tcga_ol0, merge_type = 'union')
    save_merged_samples_to_h5(tcga_union_ol0,os.path.join(outdir2,f'union_TCGA_PRAD_feature_{args.fe_method}_ol0.h5'))

    
    #Neptune
    outdir2 = os.path.join(out_dir, "Union_Neptune","IMSIZE250_OL0","TFT0.0")
    create_dir_if_not_exists(outdir2)
    nep_union_ol0   = merge_data_lists_h5(nep_ol0_nst, nep_ol0, merge_type = 'union')
    save_merged_samples_to_h5(nep_union_ol0,os.path.join(outdir2,f'union_Neptune_feature_{args.fe_method}_ol0.h5'))
    
    #Load
    # merged_ds = H5Cases(os.path.join(out_dir,"opx_union_ol100.h5"))
    # len(merged_ds)   # number of merged samples
    # sample0 = merged_ds[1]
