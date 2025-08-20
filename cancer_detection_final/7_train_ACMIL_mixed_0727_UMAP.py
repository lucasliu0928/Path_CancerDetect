#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:45:28 2025

@author: jliu6
"""

#!/usr/bin/env python
# coding: utf-8

#NOTE: use python env acmil in ACMIL folder
import sys
import os
import numpy as np
import pandas as pd
import warnings
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists, set_seed
from train_utils import get_final_model_data, get_cohort_data, combine_data_from_stnorm_and_nostnorm
warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse
import sklearn.datasets
import umap
import umap.plot
import datashader
import bokeh
import holoviews
import matplotlib.pyplot as plt
from Preprocessing import extract_before_second_underscore,extract_before_third_dash
#ml Python/3.11.5-GCCcore-13.2.0 
#source .umap_env2/bin/activate


def get_comb_feature_label(feature_path, cohort_name):
    #Load feature
    feature_df = pd.read_hdf(os.path.join(feature_path, cohort_name + "_feature.h5"), key='feature')
    feature_df.columns = feature_df.columns.astype(str)
    feature_df.reset_index(drop = True, inplace = True)
    id_df = pd.read_hdf(os.path.join(feature_path, cohort_name + "_feature.h5"), key='id')
    feature_df["ID"] = list(id_df[0])
        
    return feature_df

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--s_fold', default=0, type=int, help='select fold')
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='HR2', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_overlap', default=100, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--train_cohort', default= 'union_STNandNSTN_OPX_TCGA_NEP', type=str, help='TCGA or OPX or OPX_TCGA or z_nostnorm_OPX_TCGA or union_STNandNSTN_OPX_TCGA or comb_STNandNSTN_OPX_TCGA')
parser.add_argument('--out_folder', default= 'pred_out_081225_UMAP', type=str, help='out folder name')
parser.add_argument('--sample_training_n', default= 1000, type=int, help='random sample K tiles')


            
            
if __name__ == '__main__':
    
    args = parser.parse_args()
    fold_list = [0]
    
    #fold_list = [0,1,2,3,4]
    for f in fold_list:
        
        args.s_fold = f
        
        ##################
        ###### DIR  ######
        ##################
        proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/' 
        data_dir = os.path.join(proj_dir, "intermediate_data", "5_combined_data")
        id_data_dir = os.path.join(proj_dir, 'intermediate_data', "3B_Train_TEST_IDS")

        
        ####################################
        #Load data
        ####################################   
        def get_data(data_dir, id_data_dir, fe_method, tumor_frac):
            #OPX data
            data_opx_stnorm0 = get_cohort_data(data_dir, "z_nostnorm_OPX", fe_method, tumor_frac)
            data_opx_stnorm1 = get_cohort_data(data_dir, "OPX", fe_method, tumor_frac) #TODO: Check OPX_001 was removed beased no cancer detected in stnormed
        
        
            #TCGA data
            data_tcga_stnorm0 = get_cohort_data(data_dir, "z_nostnorm_TCGA_PRAD", fe_method, tumor_frac)
            data_tcga_stnorm1 = get_cohort_data(data_dir, "TCGA_PRAD", fe_method, tumor_frac)
        
            
            #Neptune
            data_nep_stnorm0 = get_cohort_data(data_dir, "z_nostnorm_Neptune", fe_method, tumor_frac)
            data_nep_stnorm1 = get_cohort_data(data_dir, "Neptune", fe_method, tumor_frac)
            
        
            ################################################
            # Combine stnorm and nostnorm 
            ################################################
            #OPX
            data_ol100_opx_union = combine_data_from_stnorm_and_nostnorm(data_opx_stnorm0['OL100'], data_opx_stnorm1['OL100'], method = 'union')
            data_ol0_opx_union   = combine_data_from_stnorm_and_nostnorm(data_opx_stnorm0['OL0'], data_opx_stnorm1['OL0'], method = 'union')
            data_opx_stnorm10_union = {'OL100': data_ol100_opx_union, 'OL0': data_ol0_opx_union}

            #TCGA
            data_ol100_tcga_union = combine_data_from_stnorm_and_nostnorm(data_tcga_stnorm0['OL100'], data_tcga_stnorm1['OL100'], method = 'union')
            data_ol0_tcga_union = combine_data_from_stnorm_and_nostnorm(data_tcga_stnorm0['OL0'], data_tcga_stnorm0['OL0'], method = 'union')
            data_tcga_stnorm10_union = {'OL100': data_ol100_tcga_union, 'OL0': data_ol0_tcga_union}

            #NEP
            data_ol0_nep_union = combine_data_from_stnorm_and_nostnorm(data_nep_stnorm0['OL0'], data_nep_stnorm1['OL0'], method = 'union')
            data_nep_stnorm10_union = {'OL100': data_ol0_nep_union, 'OL0': data_ol0_nep_union}
            
            return data_opx_stnorm10_union,data_tcga_stnorm10_union,data_nep_stnorm10_union
        


        opx_data, tcga_data, nep_data = get_data(data_dir, id_data_dir, args.fe_method, args.tumor_frac)

        
        
        def get_feature_label(indata,overlap,cohort_name):
            feature_list = []
            label_list = []
            corhort_list = []
            for x in indata[overlap]:
                features = x[0].mean(dim = 0, keepdim = True)
                labels = x[1]
                labels_repeated = labels.expand(features.shape[0], -1)
                cohort_label = pd.DataFrame([cohort_name]*features.shape[0])
                feature_list.append(features)
                label_list.append(labels_repeated)
                corhort_list.append(cohort_label)
                
            all_feature =  torch.concat(feature_list, dim = 0)
            all_labels =  torch.concat(label_list, dim = 0)
            cohort =  pd.concat(corhort_list, axis = 0)
            
            return all_feature, all_labels, list(cohort[0])

        ######################
        #Create output-dir
        ######################
        folder_name1 = args.fe_method + '/TrainOL' + str(args.train_overlap) +  '_TestOL' + str(args.test_overlap) + '_TFT' + str(args.tumor_frac)  + "/" 
        outdir0 = os.path.join(proj_dir + "intermediate_data/" + args.out_folder,
                               'trainCohort_' + args.train_cohort + '_Samples' + str(args.sample_training_n),
                               folder_name1,
                               'FOLD' + str(args.s_fold),
                               args.mutation)

    
        ##################
        #Select GPU
        ##################
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(device)
        
        pendigits = sklearn.datasets.load_digits()
        
        opx_features, opx_labels, opx_clabels = get_feature_label(opx_data,"OL0","OPX")
        tcga_features, tcga_labels, tcga_clabels = get_feature_label(tcga_data,"OL0","TCGA_PRAD")
        nep_features, nep_labels, nep_clabels = get_feature_label(nep_data,"OL0","Neptune")
        
        all_features = torch.concat([opx_features,tcga_features,nep_features]) 
        all_labels = torch.concat([opx_labels,tcga_labels,nep_labels]) 
        all_clabels = np.asarray(opx_clabels + tcga_clabels + nep_clabels)


        color_key = {
                "OPX": "blue",
                "TCGA_PRAD": "green",
                "Neptune": "red"
            }

        #get all features, labels ["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI"] 
        mapper = umap.UMAP().fit(all_features)
        umap.plot.points(mapper, labels=all_clabels, color_key = color_key)


        #after training
        args.s_fold = 0
        args.learning_method = "acmil"
        feature_path =  os.path.join(proj_dir + "intermediate_data/" + "pred_out_081225_V2",
                               'trainCohort_' + args.train_cohort + '_Samples' + str(args.sample_training_n),
                               args.learning_method,
                               folder_name1,
                               'FOLD' + str(args.s_fold),
                               args.mutation, "trained_features")
          
        train_df = get_comb_feature_label(feature_path, "Train")
        val_df = get_comb_feature_label(feature_path, "VAL")
        test_df = get_comb_feature_label(feature_path, "TEST_COMB")
        all_df = pd.concat([train_df,val_df,test_df])
        
        from train_utils import extract_before_third_hyphen
        all_df['PATIENT_ID'] = pd.NA
        all_df['Cohort'] = pd.NA
        cond = all_df['ID'].str.contains('TCGA')
        all_df.loc[cond,"PATIENT_ID"] = all_df.loc[cond,"ID"].apply(extract_before_third_hyphen)
        all_df.loc[cond,"Cohort"] = "TCGA_PRAD"

        cond = all_df['ID'].str.contains('OPX')
        all_df.loc[cond,"Cohort"] = "OPX"
        all_df.loc[cond,"PATIENT_ID"] = all_df.loc[cond,"ID"].str[:7]
        cond = all_df['ID'].str.contains('NEP')
        all_df.loc[cond,"Cohort"] = "OPX"
        all_df.loc[cond,"PATIENT_ID"] = all_df.loc[cond,"ID"].str[:7]

        
        label_data_dir = os.path.join(proj_dir, 'intermediate_data', "3A_otherinfo")
            
        #Label data
        opx_label_df = pd.read_csv(os.path.join(label_data_dir, "OPX","IMSIZE250_OL0", "all_tile_info.csv"))
        opx_label_df = opx_label_df.drop_duplicates(subset=['PATIENT_ID'])
        opx_label_df = opx_label_df[['SAMPLE_ID', 'PATIENT_ID', 'AR', 'HR2', 'PTEN', 'RB1', 'TP53', 'MSI_POS']]

        tcga_label_df = pd.read_csv(os.path.join(label_data_dir, "TCGA_PRAD","IMSIZE250_OL100", "all_tile_info.csv"))
        tcga_label_df = tcga_label_df.drop_duplicates(subset=['PATIENT_ID'])
        tcga_label_df = tcga_label_df[['SAMPLE_ID', 'PATIENT_ID', 'AR', 'HR2', 'PTEN', 'RB1', 'TP53', 'MSI_POS']]
        
        nep_label_df = pd.read_csv(os.path.join(label_data_dir, "Neptune","IMSIZE250_OL0", "all_tile_info.csv"))
        nep_label_df = nep_label_df.drop_duplicates(subset=['PATIENT_ID'])
        nep_label_df = nep_label_df[['SAMPLE_ID', 'PATIENT_ID', 'AR', 'HR2', 'PTEN', 'RB1', 'TP53' , 'MSI_POS']]

        label_df = pd.concat([opx_label_df,tcga_label_df,nep_label_df])
        label_df.rename(columns = {'MSI_POS':'MSI'}, inplace = True)
        
        all_df = all_df.merge(label_df, on = ['PATIENT_ID'])
        
        
        color_key = {
                "OPX": "blue",
                "TCGA_PRAD": "green",
                "Neptune": "red"
            }

        #get all features, labels ["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI"] 
        mapper = umap.UMAP().fit(all_features)
        umap.plot.points(mapper, labels=all_clabels, color_key = color_key)

        # ####################################################
        # #Training with sampling
        # ####################################################
        # if conf.sample_training_n > 0:
        #     #Random Sample 1000 tiles or oriingal N tiles (if total number is < 1000) for training data
        #     random_sample_tiles(train_data, k = conf.sample_training_n, random_seed = 42)

