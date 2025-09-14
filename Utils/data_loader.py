#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 17:15:45 2025

@author: jliu6
"""

import os
import torch
import pandas as pd
from torch.utils.data import Subset
from torch.utils.data import Dataset
import re
import time

def get_feature_idexes(method, include_tumor_fraction = True):
    
    if method == 'retccl':
        selected_feature = [str(i) for i in range(0,2048)] 
    elif method == 'uni1': 
        selected_feature = [str(i) for i in range(0,1024)] 
    elif method == 'uni2' or method == 'prov_gigapath':
        selected_feature = [str(i) for i in range(0,1536)] 
    elif method == 'virchow2':
        selected_feature = [str(i) for i in range(0,2560)] 

    if include_tumor_fraction == True:
        selected_feature = selected_feature + ['TUMOR_PIXEL_PERC'] 
        
    return selected_feature


def load_model_ready_data(
    data_path: str,
    cohort: str,
    pixel_overlap: int,
    fe_method: str,
    tumor_frac: float
 ):
    """
    Load preprocessed model-ready data for a given cohort.

    Parameters
    ----------
    data_path :     data directory
    cohort :        Name of the cohort to load data for
    pixel_overlap : Pixel overlap 
    fe_method :     Feature extraction method
    tumor_frac :    Tumor fraction threshold used during preprocessing.

    Returns
    -------
    dict or None
        Loaded model-ready data if found, otherwise None.
    """

    feature_path = os.path.join(
        data_path,
        cohort,
        f"IMSIZE250_OL{pixel_overlap}",
        f"feature_{fe_method}",
        f"TFT{tumor_frac}"
    )

    data_file = os.path.join(feature_path, f"{cohort}_data.pth")

    try:
        return torch.load(data_file)
    except FileNotFoundError:
        return None
    

    
def get_cohort_data(data_path, cohort_name, fe_method, tumor_frac):
    
    #Load data
    data_ol100 = load_model_ready_data(data_path, cohort_name, 100, fe_method, tumor_frac) #overlap 100
    data_ol0   = load_model_ready_data(data_path, cohort_name, 0, fe_method, tumor_frac) #overlap 0
    
    #Combine
    data_comb = {'OL100': data_ol100, 'OL0': data_ol0}
    
    return data_comb

def combine_cohort_data(data_dir, id_data_dir, cohort_name, fe_method, tumor_frac):
    
    #norm and nonorm data
    stnorm0 = get_cohort_data(data_dir, "z_nostnorm_" + cohort_name, fe_method, tumor_frac)
    stnorm1 = get_cohort_data(data_dir, cohort_name, fe_method, tumor_frac)

    ################################################
    # Combine stnorm and nostnorm 
    ################################################
    ol100_union = combine_data_from_stnorm_and_nostnorm(stnorm0['OL100'], stnorm1['OL100'], method = 'union')
    ol0_union   = combine_data_from_stnorm_and_nostnorm(stnorm0['OL0'], stnorm1['OL0'], method = 'union')


    cdata = {'stnorm0_OL100':     stnorm0['OL100'],
                'stnorm0_OL0':       stnorm0['OL0'],
                'stnorm1_OL100':     stnorm1['OL100'],
                'stnorm1_OL0':       stnorm1['OL0'],
                'Union_OL100': ol100_union, 
                'Union_OL0':  ol0_union}

    
    return cdata


def get_matching_tile_index(indata, common_tiles_xy):
    # Get row indices in df1
    match_mask = indata['TILE_COOR_ATLV0'].isin(common_tiles_xy)
    df_match = indata[match_mask]
    df_nomatch = indata[~match_mask]
    match_index = df_match.index.tolist()
    nomatch_index = df_nomatch.index.tolist()
    
    return  match_index, nomatch_index

def get_larger_tumor_fraction_tile(tf1, tf2, match_index1, match_index2):
    
    tensor1 = tf1[match_index1]
    tensor2 = tf2[match_index2]
    
    
    mask_tensor1 = tensor1 >= tensor2
    
    # Get indices where tensor1 "wins"
    indices_tensor1 = mask_tensor1.nonzero(as_tuple=True)[0]
    
    # Get indices where tensor2 "wins"
    indices_tensor2 = (~mask_tensor1).nonzero(as_tuple=True)[0]
    
    final_idx1  = [match_index1[i] for i in indices_tensor1.tolist()]
    final_idx2  = [match_index2[i] for i in indices_tensor2.tolist()]
    
    return final_idx1, final_idx2


# def combine_data_from_stnorm_and_nostnorm(indata_stnorm, indata_stnorm_no, method = 'union'):
    
#     # indata_stnorm = data_ol100_opx_stnorm0
#     # indata_stnorm_no = data_ol100_opx_stnorm1
    
#     comb_data_list = []
#     #get the data with more IDs
#     if len(indata_stnorm) >= len(indata_stnorm_no):
#         main_data = indata_stnorm
#         nomain_data = indata_stnorm_no
#     else:
#         main_data = indata_stnorm_no
#         nomain_data = indata_stnorm

#     for i in range(len(main_data)):

#         # Unpack the tuple
#         features1, labels1, tf1, other_info1, sample_id1, patient_id1 = main_data[i]
        
#         #find the data in indata2
#         index = next((i for i, entry in enumerate(nomain_data) if entry[-2] == sample_id1), None)
        
    
        
#         if index is not None: #if the ID is in both data, take the union or combine
#             features2, labels2, tf2, other_info2, sample_id2, patient_id2 = nomain_data[index]
#             if method == 'combine_all':
#                 comb_data = (
#                     features1,
#                     labels1,
#                     tf1,
#                     other_info1,
#                     sample_id1,
#                     patient_id1,
                    
#                     features2,
#                     labels2,
#                     tf2,
#                     other_info2,
#                     sample_id2,
#                     patient_id2,
#                 )
#             elif method == 'union':
#                 #Intersect
#                 common_tiles = set(other_info1['TILE_COOR_ATLV0']).intersection(other_info2['TILE_COOR_ATLV0'])
#                 match_index1, nomatch_index1 = get_matching_tile_index(other_info1, common_tiles)
#                 match_index2, nomatch_index2 = get_matching_tile_index(other_info2, common_tiles)
#                 match_index1_tokeep, match_index2_tokeep = get_larger_tumor_fraction_tile(tf1, tf2,match_index1, match_index2)
                
#                 final_idx_tokeep1 = match_index1_tokeep + nomatch_index1
#                 final_idx_tokeep2 = match_index2_tokeep + nomatch_index2
                
                
#                 #Get updated info
#                 feature = torch.concat([features1[final_idx_tokeep1], features2[final_idx_tokeep2]])
#                 tf = torch.concat([tf1[final_idx_tokeep1], tf2[final_idx_tokeep2]])
#                 labels = labels1 
#                 other_info = pd.concat([other_info1.iloc[final_idx_tokeep1], other_info2.iloc[final_idx_tokeep2]])
#                 sample_id = sample_id1
#                 patient_id = patient_id1
                
#                 comb_data = (
#                     feature,
#                     labels,
#                     tf,
#                     other_info,
#                     sample_id,
#                     patient_id
#                 )

#         else: #only take all the things in main data
#             if method == 'combine_all':
#                 comb_data = (
#                     features1,
#                     labels1,
#                     tf1,
#                     other_info1,
#                     sample_id1,
#                     patient_id1,
#                     [],
#                     [],
#                     [],
#                     [],
#                     [],
#                     []
                    
#                 )
#             elif method == 'union':
#                  comb_data = (
#                      features1,
#                      labels1,
#                      tf1,
#                      other_info1,
#                      sample_id1,
#                      patient_id1
#                  )
                 
#         comb_data_list.append(comb_data)
        
#     return comb_data_list   


def get_selected_labels(mutation_name, train_cohort):
    
    all_labels = ["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI"]    
    
    if mutation_name == 'MT':
        if 'TCGA' in train_cohort:
            selected = ["AR" , "HR2", "PTEN","RB1","TP53","MSI"]   #without TMB
        else:
            selected = ["AR",  "HR2", "PTEN","RB1","TP53","TMB","MSI"]   
    else:
            selected = mutation_name.split('_')

    #get label index in all labels
    selected_index = []
    for l in selected:
        selected_index.append(all_labels.index(l))
    
    return selected, selected_index

def update_label(indata, select_label_index):
    r'''
    Input:   Model Ready data
    Returns: Model ready data, List of ids
    -------
    '''
    
    #Update label
    indata_final = [(x[0], x[1][:,select_label_index], x[2], x[3],x[4],x[5]) for x in indata]

    print(f'DS: {len(indata_final)}')

    return indata_final


def get_final_model_data_v2(opx, tcga, nep, id_data_dir, train_cohort, mutation, fe_method, tumor_frac, s_fold):
    
    ################################################
    #Get Train, test, val data
    ################################################    
    
    
    train_cohort_map = {
        # #stain normed
        # 'OPX': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': opx['stnorm1_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'TCGA_PRAD': {
        #     'train_cohort1': 'TCGA_PRAD',
        #     'model_data1': tcga['stnorm1_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'Neptune': {
        #     'train_cohort1': 'Neptune',
        #     'model_data1': nep['stnorm1_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        
        # #no stain normed
        # 'z_nostnorm_OPX': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': opx['stnorm0_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'z_nostnorm_TCGA_PRAD': {
        #     'train_cohort1': 'TCGA_PRAD',
        #     'model_data1': tcga['stnorm0_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'z_nostnorm_Neptune': {
        #     'train_cohort1': 'Neptune',
        #     'model_data1': nep['stnorm0_OL100'],
        #     'train_cohort2': None,
        #     'model_data2': None,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        
        # #two corhot, no st
        # 'z_nostnorm_OPX_TCGA': {
        #     'train_cohort1': 'z_nostnorm_OPX',
        #     'model_data1': opx['stnorm0_OL100'],
        #     'train_cohort2': 'z_nostnorm_TCGA_PRAD',
        #     'model_data2': tcga['stnorm0_OL100'],
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'z_nostnorm_OPX_NEP': {
        #     'train_cohort1': 'z_nostnorm_OPX',
        #     'model_data1': opx['stnorm0_OL100',
        #     'train_cohort2': 'z_nostnorm_Neptune',
        #     'model_data2': data_nep_stnorm0,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        # 'z_nostnorm_TCGA_NEP': {
        #     'train_cohort1': 'z_nostnorm_TCGA_PRAD',
        #     'model_data1': data_tcga_stnorm0,
        #     'train_cohort2': 'z_nostnorm_Neptune',
        #     'model_data2': data_nep_stnorm0,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # },
        
        # #two corhot, st
        # 'OPX_TCGA': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': data_opx_stnorm1,
        #     'train_cohort2': 'TCGA_PRAD',
        #     'model_data2': data_tcga_stnorm1,
        #     'train_cohort3':  None,
        #     'model_data3': None,
        #     'ext_cohort':  'Neptune',
        #     'ext_data_nost':  data_nep_stnorm0['OL0'],
        #     'ext_data_st':    data_nep_stnorm1['OL0'],
        #     'ext_data_union': data_nep_stnorm10_union['OL0']
        # },
        # 'OPX_NEP': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': data_opx_stnorm1,
        #     'train_cohort2': 'Neptune',
        #     'model_data2': data_nep_stnorm1,
        #     'train_cohort3':  None,
        #     'model_data3': None,
        #     'ext_cohort':  'TCGA_PRAD',
        #     'ext_data_nost':  data_tcga_stnorm0['OL0'],
        #     'ext_data_st':    data_tcga_stnorm1['OL0'],
        #     'ext_data_union': data_tcga_stnorm10_union['OL0']
        # },
        # 'TCGA_NEP': {
        #     'train_cohort1': 'TCGA_PRAD',
        #     'model_data1': data_tcga_stnorm1,
        #     'train_cohort2': 'Neptune',
        #     'model_data2': data_nep_stnorm1,
        #     'train_cohort3':  None,
        #     'model_data3': None,
        #     'ext_cohort':  'OPX',
        #     'ext_data_nost':  data_opx_stnorm0['OL0'],
        #     'ext_data_st':    data_opx_stnorm1['OL0'],
        #     'ext_data_union': data_opx_stnorm10_union['OL0']
        # },
        
        
        #Dict
        #'stnorm0_OL100', 'stnorm0_OL0', 'stnorm1_OL100', 'stnorm1_OL0', 'Union_OL100', 'Union_OL0']

        
        #two cohort, union of st and no-st
        'union_STNandNSTN_OPX_TCGA': {
            'train_cohort1': 'OPX',
            'model_data1': {'OL100': opx['Union_OL100'], 'OL0': opx['Union_OL0']},
            'train_cohort2': 'TCGA_PRAD',
            'model_data2': {'OL100': tcga['Union_OL100'], 'OL0': tcga['Union_OL0']},
            'train_cohort3':  None,
            'model_data3': None,
            'ext_cohort':  'Neptune',
            'ext_data_nost':  nep['stnorm0_OL0'],
            'ext_data_st':    nep['stnorm1_OL0'],
            'ext_data_union': nep['Union_OL0'],

        },
        # 'union_STNandNSTN_OPX_NEP': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': data_opx_stnorm10_union,
        #     'train_cohort2': 'Neptune',
        #     'model_data2': data_nep_stnorm10_union,
        #     'train_cohort3':  None,
        #     'model_data3': None,
        #     'ext_cohort':  'TCGA_PRAD',
        #     'ext_data_nost':  data_tcga_stnorm0['OL0'],
        #     'ext_data_st':    data_tcga_stnorm1['OL0'],
        #     'ext_data_union': data_tcga_stnorm10_union['OL0'],
        # },
        # 'union_STNandNSTN_TCGA_NEP': {
        #     'train_cohort1': 'TCGA_PRAD',
        #     'model_data1': data_tcga_stnorm10_union,
        #     'train_cohort2': 'Neptune',
        #     'model_data2': data_nep_stnorm10_union,
        #     'train_cohort3':  None,
        #     'model_data3': None,
        #     'ext_cohort':  'OPX',
        #     'ext_data_nost':  data_opx_stnorm0['OL0'],
        #     'ext_data_st':    data_opx_stnorm1['OL0'],
        #     'ext_data_union': data_opx_stnorm10_union['OL0'],
        # },
        
        # #three cohort, union of st and no-st
        # 'union_STNandNSTN_OPX_TCGA_NEP': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': data_opx_stnorm10_union,
        #     'train_cohort2': 'TCGA_PRAD',
        #     'model_data2': data_tcga_stnorm10_union,
        #     'train_cohort3':  'Neptune',
        #     'model_data3': data_nep_stnorm10_union,
        #     'ext_cohort':  None,
        #     'ext_data_nost':  None,
        #     'ext_data_st':    None,
        #     'ext_data_union': None,
        # },
        
        
        # #two cohort, combine and later sampling
        # 'comb_STNandNSTN_OPX_TCGA': {
        #     'train_cohort1': 'OPX',
        #     'model_data1': data_opx_stnorm10_comb,
        #     'train_cohort2': 'TCGA_PRAD',
        #     'model_data2': data_tcga_stnorm10_comb,
        #     'train_cohort3':  None,
        #     'model_data3': None
        # }
    }
    
    if train_cohort in train_cohort_map:
        
        config = train_cohort_map[train_cohort]
        train_cohort1 = config['train_cohort1']
        model_data1 = config['model_data1']
        train_cohort2 = config['train_cohort2']
        model_data2 = config['model_data2']
        train_cohort3 = config['train_cohort3']
        model_data3 = config['model_data3']
        ext_cohort = config['ext_cohort']
        ext_data_st0 = config['ext_data_nost']
        ext_data_st1 = config['ext_data_st']
        ext_data_union = config['ext_data_union']
        
        
    else:
        raise ValueError(f"Unknown training cohort: {train_cohort}")

    #TODO for three cohort train
    
    #Get Train, validation and test
    split_data = get_final_split_data(id_data_dir,
                                    train_cohort1,
                                    model_data1, 
                                    train_cohort2, 
                                    model_data2, 
                                    train_cohort3,
                                    model_data3,
                                    tumor_frac, 
                                    s_fold)
    
    train_data, _ = split_data["train"]
    val_data, _ = split_data["val"]
    test_data, _ = split_data["test"]
    test_data1, _ = split_data["test1"]
    test_data2, _ = split_data["test2"]
    test_data3, _ = split_data["test3"]
    

    #Get sanple ID:
    train_ids =  [x[-2] for x in train_data]
    val_ids =  [x[-2] for x in val_data]
    test_ids =  [x[-2] for x in test_data]
    test_ids1 =  [x[-2] for x in test_data1]
    test_ids2 =  [x[-2] for x in test_data2]
    test_ids3 =  [x[-2] for x in test_data3]
    
    if train_cohort != "union_STNandNSTN_OPX_TCGA_NEP":
        ext_ids0 =  [x[-2] for x in ext_data_st0]
        ext_ids1 =  [x[-2] for x in ext_data_st1]
        ext_ids =  [x[-2] for x in ext_data_union]
    else:
        ext_ids0 = []
        ext_ids1 = []
        ext_ids = []        
    
    
    #Update labels
    selected_labels, selected_label_index = get_selected_labels(mutation, train_cohort)
    print(selected_labels)
    print(selected_label_index)

    train_data = update_label(train_data, selected_label_index)
    val_data = update_label(val_data, selected_label_index)
    test_data = update_label(test_data, selected_label_index)
    test_data1 = update_label(test_data1, selected_label_index)
    test_data2 = update_label(test_data2, selected_label_index)
    test_data3 = update_label(test_data3, selected_label_index)

    if train_cohort != "union_STNandNSTN_OPX_TCGA_NEP":
        ext_data_st0 = update_label(ext_data_st0, selected_label_index)
        ext_data_st1 = update_label(ext_data_st1, selected_label_index)
        ext_data_union = update_label(ext_data_union, selected_label_index)
    else:
        ext_data_st0 = []
        ext_data_st1 = []
        ext_data_union = []
    
    #update item for model
            
    if train_cohort == 'comb_STNandNSTN_OPX_TCGA': 
        #Keep feature1, label, tf,1 dlabel, feature2, tf2
        train_data = [(item[0], item[1], item[2], item[3], item[7], item[9]) for item in train_data]
        test_data1 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data1]
        test_data2 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data2]
        test_data3 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data3]

        test_data = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data]
        val_data = [(item[0], item[1], item[2],  item[6], item[8]) for item in val_data]
    else:
        #Exclude tile info data, sample ID, patient ID, do not needed it for training
        train_data = [item[:-3] for item in train_data]
        test_data1 = [item[:-3] for item in test_data1]   
        test_data2 = [item[:-3] for item in test_data2]   
        test_data3 = [item[:-3] for item in test_data3]   
        test_data = [item[:-3] for item in test_data]   
        val_data = [item[:-3] for item in val_data]
    
    if train_cohort != "union_STNandNSTN_OPX_TCGA_NEP":
        ext_data_st0 = [item[:-3] for item in ext_data_st0] #no st norm
        ext_data_st1 = [item[:-3] for item in ext_data_st1] #st normed
        ext_data_union = [item[:-3] for item in ext_data_union]


    
    return {'train': (train_data, train_ids, "Train"),
            'val': (val_data, val_ids, "VAL"),
            'test': (test_data, test_ids,"Test"),
            'test1': (test_data1, test_ids1, train_cohort1),
            'test2': (test_data2, test_ids2, train_cohort2),
            'test3': (test_data3, test_ids3, train_cohort3),
            'ext_data_st0': (ext_data_st0, ext_ids0, ext_cohort),
            'ext_data_st1': (ext_data_st1, ext_ids1, ext_cohort),
            'ext_data_union': (ext_data_union, ext_ids, ext_cohort)
        }, selected_labels


# def get_final_model_data(data_dir, id_data_dir, train_cohort, mutation, fe_method, tumor_frac, s_fold):
#     #OPX data
#     data_opx_stnorm0 = get_cohort_data(data_dir, "z_nostnorm_OPX", fe_method, tumor_frac)
#     data_opx_stnorm1 = get_cohort_data(data_dir, "OPX", fe_method, tumor_frac) #TODO: Check OPX_001 was removed beased no cancer detected in stnormed


#     #TCGA data
#     data_tcga_stnorm0 = get_cohort_data(data_dir, "z_nostnorm_TCGA_PRAD", fe_method, tumor_frac)
#     data_tcga_stnorm1 = get_cohort_data(data_dir, "TCGA_PRAD", fe_method, tumor_frac)

    
#     #Neptune
#     data_nep_stnorm0 = get_cohort_data(data_dir, "z_nostnorm_Neptune", fe_method, tumor_frac)
#     data_nep_stnorm1 = get_cohort_data(data_dir, "Neptune", fe_method, tumor_frac)
    

#     ################################################
#     # Combine stnorm and nostnorm 
#     ################################################
#     #OPX
#     data_ol100_opx_union = combine_data_from_stnorm_and_nostnorm(data_opx_stnorm0['OL100'], data_opx_stnorm1['OL100'], method = 'union')
#     data_ol0_opx_union   = combine_data_from_stnorm_and_nostnorm(data_opx_stnorm0['OL0'], data_opx_stnorm1['OL0'], method = 'union')
#     data_opx_stnorm10_union = {'OL100': data_ol100_opx_union, 'OL0': data_ol0_opx_union}

#     data_ol100_opx_comb = combine_data_from_stnorm_and_nostnorm(data_opx_stnorm0['OL100'],  data_opx_stnorm1['OL100'], method = 'combine_all')
#     data_ol0_opx_comb = combine_data_from_stnorm_and_nostnorm(data_opx_stnorm0['OL0'], data_opx_stnorm1['OL0'], method = 'combine_all')
#     data_opx_stnorm10_comb = {'OL100': data_ol100_opx_comb, 'OL0': data_ol0_opx_comb}

#     #TCGA
#     data_ol100_tcga_union = combine_data_from_stnorm_and_nostnorm(data_tcga_stnorm0['OL100'], data_tcga_stnorm1['OL100'], method = 'union')
#     data_ol0_tcga_union = combine_data_from_stnorm_and_nostnorm(data_tcga_stnorm0['OL0'], data_tcga_stnorm0['OL0'], method = 'union')
#     data_tcga_stnorm10_union = {'OL100': data_ol100_tcga_union, 'OL0': data_ol0_tcga_union}

#     data_ol100_tcga_comb = combine_data_from_stnorm_and_nostnorm(data_tcga_stnorm0['OL100'], data_tcga_stnorm1['OL100'], method = 'combine_all')
#     data_ol0_tcga_comb = combine_data_from_stnorm_and_nostnorm(data_tcga_stnorm0['OL0'], data_tcga_stnorm0['OL0'], method = 'combine_all')
#     data_tcga_stnorm10_comb = {'OL100': data_ol100_tcga_comb, 'OL0': data_ol0_tcga_comb}
    
#     #NEP
#     #TODO:Need to update using OL100 later for nep
#     data_ol0_nep_union = combine_data_from_stnorm_and_nostnorm(data_nep_stnorm0['OL0'], data_nep_stnorm1['OL0'], method = 'union')
#     data_nep_stnorm10_union = {'OL100': data_ol0_nep_union, 'OL0': data_ol0_nep_union}
#     data_ol0_nep_comb = combine_data_from_stnorm_and_nostnorm(data_nep_stnorm0['OL0'], data_nep_stnorm1['OL0'], method = 'combine_all')
#     data_nep_stnorm10_comb = {'OL100': data_ol0_nep_comb, 'OL0': data_ol0_nep_comb}
    
#     ################################################
#     #Get Train, test, val data
#     ################################################    
#     train_cohort_map = {
#         #stain normed
#         'OPX': {
#             'train_cohort1': 'OPX',
#             'model_data1': data_opx_stnorm1,
#             'train_cohort2': None,
#             'model_data2': None,
#             'train_cohort3':  None,
#             'model_data3': None
#         },
#         'TCGA_PRAD': {
#             'train_cohort1': 'TCGA_PRAD',
#             'model_data1': data_tcga_stnorm1,
#             'train_cohort2': None,
#             'model_data2': None,
#             'train_cohort3':  None,
#             'model_data3': None
#         },
#         'Neptune': {
#             'train_cohort1': 'Neptune',
#             'model_data1': data_nep_stnorm1,
#             'train_cohort2': None,
#             'model_data2': None,
#             'train_cohort3':  None,
#             'model_data3': None
#         },
        
#         #no stain normed
#         'z_nostnorm_OPX': {
#             'train_cohort1': 'OPX',
#             'model_data1': data_opx_stnorm0,
#             'train_cohort2': None,
#             'model_data2': None,
#             'train_cohort3':  None,
#             'model_data3': None
#         },
#         'z_nostnorm_TCGA_PRAD': {
#             'train_cohort1': 'TCGA_PRAD',
#             'model_data1': data_tcga_stnorm0,
#             'train_cohort2': None,
#             'model_data2': None,
#             'train_cohort3':  None,
#             'model_data3': None
#         },
#         'z_nostnorm_Neptune': {
#             'train_cohort1': 'Neptune',
#             'model_data1': data_nep_stnorm0,
#             'train_cohort2': None,
#             'model_data2': None,
#             'train_cohort3':  None,
#             'model_data3': None
#         },
        
#         #two corhot, no st
#         'z_nostnorm_OPX_TCGA': {
#             'train_cohort1': 'z_nostnorm_OPX',
#             'model_data1': data_opx_stnorm0,
#             'train_cohort2': 'z_nostnorm_TCGA_PRAD',
#             'model_data2': data_tcga_stnorm0,
#             'train_cohort3':  None,
#             'model_data3': None
#         },
#         'z_nostnorm_OPX_NEP': {
#             'train_cohort1': 'z_nostnorm_OPX',
#             'model_data1': data_opx_stnorm0,
#             'train_cohort2': 'z_nostnorm_Neptune',
#             'model_data2': data_nep_stnorm0,
#             'train_cohort3':  None,
#             'model_data3': None
#         },
#         'z_nostnorm_TCGA_NEP': {
#             'train_cohort1': 'z_nostnorm_TCGA_PRAD',
#             'model_data1': data_tcga_stnorm0,
#             'train_cohort2': 'z_nostnorm_Neptune',
#             'model_data2': data_nep_stnorm0,
#             'train_cohort3':  None,
#             'model_data3': None
#         },
        
#         #two corhot, st
#         'OPX_TCGA': {
#             'train_cohort1': 'OPX',
#             'model_data1': data_opx_stnorm1,
#             'train_cohort2': 'TCGA_PRAD',
#             'model_data2': data_tcga_stnorm1,
#             'train_cohort3':  None,
#             'model_data3': None,
#             'ext_cohort':  'Neptune',
#             'ext_data_nost':  data_nep_stnorm0['OL0'],
#             'ext_data_st':    data_nep_stnorm1['OL0'],
#             'ext_data_union': data_nep_stnorm10_union['OL0']
#         },
#         'OPX_NEP': {
#             'train_cohort1': 'OPX',
#             'model_data1': data_opx_stnorm1,
#             'train_cohort2': 'Neptune',
#             'model_data2': data_nep_stnorm1,
#             'train_cohort3':  None,
#             'model_data3': None,
#             'ext_cohort':  'TCGA_PRAD',
#             'ext_data_nost':  data_tcga_stnorm0['OL0'],
#             'ext_data_st':    data_tcga_stnorm1['OL0'],
#             'ext_data_union': data_tcga_stnorm10_union['OL0']
#         },
#         'TCGA_NEP': {
#             'train_cohort1': 'TCGA_PRAD',
#             'model_data1': data_tcga_stnorm1,
#             'train_cohort2': 'Neptune',
#             'model_data2': data_nep_stnorm1,
#             'train_cohort3':  None,
#             'model_data3': None,
#             'ext_cohort':  'OPX',
#             'ext_data_nost':  data_opx_stnorm0['OL0'],
#             'ext_data_st':    data_opx_stnorm1['OL0'],
#             'ext_data_union': data_opx_stnorm10_union['OL0']
#         },
        
#         #two cohort, union of st and no-st
#         'union_STNandNSTN_OPX_TCGA': {
#             'train_cohort1': 'OPX',
#             'model_data1': data_opx_stnorm10_union,
#             'train_cohort2': 'TCGA_PRAD',
#             'model_data2': data_tcga_stnorm10_union,
#             'train_cohort3':  None,
#             'model_data3': None,
#             'ext_cohort':  'Neptune',
#             'ext_data_nost':  data_nep_stnorm0['OL0'],
#             'ext_data_st':    data_nep_stnorm1['OL0'],
#             'ext_data_union': data_nep_stnorm10_union['OL0'],

#         },
#         'union_STNandNSTN_OPX_NEP': {
#             'train_cohort1': 'OPX',
#             'model_data1': data_opx_stnorm10_union,
#             'train_cohort2': 'Neptune',
#             'model_data2': data_nep_stnorm10_union,
#             'train_cohort3':  None,
#             'model_data3': None,
#             'ext_cohort':  'TCGA_PRAD',
#             'ext_data_nost':  data_tcga_stnorm0['OL0'],
#             'ext_data_st':    data_tcga_stnorm1['OL0'],
#             'ext_data_union': data_tcga_stnorm10_union['OL0'],
#         },
#         'union_STNandNSTN_TCGA_NEP': {
#             'train_cohort1': 'TCGA_PRAD',
#             'model_data1': data_tcga_stnorm10_union,
#             'train_cohort2': 'Neptune',
#             'model_data2': data_nep_stnorm10_union,
#             'train_cohort3':  None,
#             'model_data3': None,
#             'ext_cohort':  'OPX',
#             'ext_data_nost':  data_opx_stnorm0['OL0'],
#             'ext_data_st':    data_opx_stnorm1['OL0'],
#             'ext_data_union': data_opx_stnorm10_union['OL0'],
#         },
        
#         #three cohort, union of st and no-st
#         'union_STNandNSTN_OPX_TCGA_NEP': {
#             'train_cohort1': 'OPX',
#             'model_data1': data_opx_stnorm10_union,
#             'train_cohort2': 'TCGA_PRAD',
#             'model_data2': data_tcga_stnorm10_union,
#             'train_cohort3':  'Neptune',
#             'model_data3': data_nep_stnorm10_union,
#             'ext_cohort':  None,
#             'ext_data_nost':  None,
#             'ext_data_st':    None,
#             'ext_data_union': None,
#         },
        
        
#         #two cohort, combine and later sampling
#         'comb_STNandNSTN_OPX_TCGA': {
#             'train_cohort1': 'OPX',
#             'model_data1': data_opx_stnorm10_comb,
#             'train_cohort2': 'TCGA_PRAD',
#             'model_data2': data_tcga_stnorm10_comb,
#             'train_cohort3':  None,
#             'model_data3': None
#         }
#     }
    
#     if train_cohort in train_cohort_map:
        
#         config = train_cohort_map[train_cohort]
#         train_cohort1 = config['train_cohort1']
#         model_data1 = config['model_data1']
#         train_cohort2 = config['train_cohort2']
#         model_data2 = config['model_data2']
#         train_cohort3 = config['train_cohort3']
#         model_data3 = config['model_data3']
#         ext_cohort = config['ext_cohort']
#         ext_data_st0 = config['ext_data_nost']
#         ext_data_st1 = config['ext_data_st']
#         ext_data_union = config['ext_data_union']
        
        
#     else:
#         raise ValueError(f"Unknown training cohort: {train_cohort}")

#     #TODO for three cohort train
    
#     #Get Train, validation and test
#     split_data = get_final_split_data(id_data_dir,
#                                     train_cohort1,
#                                     model_data1, 
#                                     train_cohort2, 
#                                     model_data2, 
#                                     train_cohort3,
#                                     model_data3,
#                                     tumor_frac, 
#                                     s_fold)
    
#     train_data, _ = split_data["train"]
#     val_data, _ = split_data["val"]
#     test_data, _ = split_data["test"]
#     test_data1, _ = split_data["test1"]
#     test_data2, _ = split_data["test2"]
#     test_data3, _ = split_data["test3"]
    

#     #Get sanple ID:
#     train_ids =  [x[-2] for x in train_data]
#     val_ids =  [x[-2] for x in val_data]
#     test_ids =  [x[-2] for x in test_data]
#     test_ids1 =  [x[-2] for x in test_data1]
#     test_ids2 =  [x[-2] for x in test_data2]
#     test_ids3 =  [x[-2] for x in test_data3]
    
#     if train_cohort != "union_STNandNSTN_OPX_TCGA_NEP":
#         ext_ids0 =  [x[-2] for x in ext_data_st0]
#         ext_ids1 =  [x[-2] for x in ext_data_st1]
#         ext_ids =  [x[-2] for x in ext_data_union]
#     else:
#         ext_ids0 = []
#         ext_ids1 = []
#         ext_ids = []        
    
    
#     #Update labels
#     selected_labels, selected_label_index = get_selected_labels(mutation, train_cohort)
#     print(selected_labels)
#     print(selected_label_index)

#     train_data = update_label(train_data, selected_label_index)
#     val_data = update_label(val_data, selected_label_index)
#     test_data = update_label(test_data, selected_label_index)
#     test_data1 = update_label(test_data1, selected_label_index)
#     test_data2 = update_label(test_data2, selected_label_index)
#     test_data3 = update_label(test_data3, selected_label_index)

#     if train_cohort != "union_STNandNSTN_OPX_TCGA_NEP":
#         ext_data_st0 = update_label(ext_data_st0, selected_label_index)
#         ext_data_st1 = update_label(ext_data_st1, selected_label_index)
#         ext_data_union = update_label(ext_data_union, selected_label_index)
#     else:
#         ext_data_st0 = []
#         ext_data_st1 = []
#         ext_data_union = []
    
#     #update item for model
            
#     if train_cohort == 'comb_STNandNSTN_OPX_TCGA': 
#         #Keep feature1, label, tf,1 dlabel, feature2, tf2
#         train_data = [(item[0], item[1], item[2], item[3], item[7], item[9]) for item in train_data]
#         test_data1 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data1]
#         test_data2 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data2]
#         test_data3 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data3]

#         test_data = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data]
#         val_data = [(item[0], item[1], item[2],  item[6], item[8]) for item in val_data]
#     else:
#         #Exclude tile info data, sample ID, patient ID, do not needed it for training
#         train_data = [item[:-3] for item in train_data]
#         test_data1 = [item[:-3] for item in test_data1]   
#         test_data2 = [item[:-3] for item in test_data2]   
#         test_data3 = [item[:-3] for item in test_data3]   
#         test_data = [item[:-3] for item in test_data]   
#         val_data = [item[:-3] for item in val_data]
    
#     if train_cohort != "union_STNandNSTN_OPX_TCGA_NEP":
#         ext_data_st0 = [item[:-3] for item in ext_data_st0] #no st norm
#         ext_data_st1 = [item[:-3] for item in ext_data_st1] #st normed
#         ext_data_union = [item[:-3] for item in ext_data_union]


    
#     return {'train': (train_data, train_ids, "Train"),
#             'val': (val_data, val_ids, "VAL"),
#             'test': (test_data, test_ids,"Test"),
#             'test1': (test_data1, test_ids1, train_cohort1),
#             'test2': (test_data2, test_ids2, train_cohort2),
#             'test3': (test_data3, test_ids3, train_cohort3),
#             'ext_data_st0': (ext_data_st0, ext_ids0, ext_cohort),
#             'ext_data_st1': (ext_data_st1, ext_ids1, ext_cohort),
#             'ext_data_union': (ext_data_union, ext_ids, ext_cohort)
#         }, selected_labels



def get_final_split_data(id_data_dir, train_cohort1, model_data1, train_cohort2, model_data2, train_cohort3, model_data3, tumor_frac, s_fold):
    

    (train_data1, train_ids1), (val_data1, val_ids1), (test_data1, test_ids1) = get_train_test_val_data_singlecohort(id_data_dir, 
                                                                                                               train_cohort1 ,
                                                                                                               model_data = model_data1, 
                                                                                                               tumor_frac = tumor_frac, 
                                                                                                               s_fold = s_fold)
    
    
    if train_cohort2 is not None:
        (train_data2, train_ids2), (val_data2, val_ids2), (test_data2, test_ids2) = get_train_test_val_data_singlecohort(id_data_dir, 
                                                                                                                   train_cohort2 ,
                                                                                                                   model_data = model_data2, 
                                                                                                                   tumor_frac = tumor_frac, 
                                                                                                                   s_fold = s_fold)
    
    else:
        (train_data2, train_ids2), (val_data2, val_ids2), (test_data2, test_ids2) = ([], []), ([], []), ([], [])
        
        
    
    if train_cohort3 is not None:
        (train_data3, train_ids3), (val_data3, val_ids3), (test_data3, test_ids3) = get_train_test_val_data_singlecohort(id_data_dir, 
                                                                                                                   train_cohort3 ,
                                                                                                                   model_data = model_data3, 
                                                                                                                   tumor_frac = tumor_frac, 
                                                                                                                   s_fold = s_fold)
    
    else:
        (train_data3, train_ids3), (val_data3, val_ids3), (test_data3, test_ids3) = ([], []), ([], []), ([], [])
    
    
    train_data = train_data1 + train_data2 + train_data3
    train_ids = train_ids1 + train_ids2 + train_ids3
    
    val_data = val_data1 + val_data2 + val_data3
    val_ids = val_ids1 + val_ids2 + val_ids3
    
    test_data = test_data1 + test_data2  + test_data3
    test_ids = test_ids1 + test_ids2 + test_ids3
    
    return {
            "train": (train_data, train_ids),
            "val": (val_data, val_ids),
            "test": (test_data, test_ids),
            "test1": (test_data1, test_ids1),
            "test2": (test_data2, test_ids2),
            "test3": (test_data3, test_ids3)
            }


def get_partial_data(indata, selected_ids):
    r'''
    Input:   Model Ready data pool, List of selected_ids
    Returns: Model ready data for selected Ids
    -------
    '''
    #Get ordered patient IDs in indata
    ids_pool  = [x[-1] for x in indata] #The 2nd to the last one is sample ID, the last one is patient ID

    #Find all index of the selected ids
    inc_idx = [i for x in selected_ids for i, val in enumerate(ids_pool) if val == x]

    #Subsets
    indata_subset = Subset(indata, inc_idx)
    
    #Get final 
    sp_ids_order =  [x[-2] for x in indata_subset] #sample IDs
    pt_ids_order =  [x[-1] for x in indata_subset] #patient IDs

    return indata_subset,sp_ids_order, pt_ids_order


def get_train_test_val_data(data_pool_train, data_pool_test, id_df, fold):

    #Get train, test IDs
    train_ids_pt = list(id_df.loc[id_df['FOLD' + str(fold)] == 'TRAIN', 'PATIENT_ID'])
    test_ids_pt  = list(id_df.loc[id_df['FOLD' + str(fold)] == 'TEST', 'PATIENT_ID'])
    val_ids_pt   = list(id_df.loc[id_df['FOLD' + str(fold)] == 'VALID', 'PATIENT_ID'])
   
    train_data_final, train_sp_ids_final, train_pt_ids_final = get_partial_data(data_pool_train, train_ids_pt)
    test_data_final, test_sp_ids_final, test_pt_ids_final = get_partial_data(data_pool_test, test_ids_pt)
    val_data_final, val_sp_ids_final, val_pt_ids_final = get_partial_data(data_pool_train, val_ids_pt)
    
    print(f'Sample N: Train: {len(set(train_sp_ids_final))}; Test: {len(set(test_sp_ids_final))}; Val: {len(set(val_sp_ids_final))}')
    print(f'Patient N: Train: {len(set(train_pt_ids_final))}; Test: {len(set(test_pt_ids_final))}; Val: {len(set(val_pt_ids_final))}')
    print(f'dataset N: Train: {len(train_data_final)}; Test: {len(test_data_final)}; Val: {len(val_data_final)}')

    return (train_data_final, train_sp_ids_final), (val_data_final, val_sp_ids_final), (test_data_final, test_sp_ids_final)



def get_train_test_val_data_singlecohort(data_dir, cohort_name , model_data, tumor_frac, s_fold):
    
    #data_dir = proj_dir + 'intermediate_data/3B_Train_TEST_IDS'
    
    #Load ID split data
    d_path =  os.path.join(data_dir, cohort_name ,'TFT' + str(tumor_frac))
    train_test_val_id_df = pd.read_csv(os.path.join(d_path, "train_test_split.csv"))
    train_test_val_id_df.rename(columns = {'TMB_HIGHorINTERMEDITATE': 'TMB'}, inplace = True)
    
    
    #Load data
    train_model_data = model_data['OL100']
    test_model_data  = model_data['OL0']

    #get train test data
    (train_data, train_ids), (val_data, val_ids), (test_data, test_ids) = get_train_test_val_data(train_model_data, test_model_data, train_test_val_id_df, s_fold)
    
    #add domain label:
    train_data = [item[:3] + (torch.tensor(1.0),) + item[3:] for item in train_data] #1 for OPX, biopsy, #0 for TCGA, surgical
    
    return (train_data, train_ids), (val_data, val_ids), (test_data, test_ids)





class ModelReadyData_diffdim_V2(Dataset):
    def __init__(self,
                 tile_info_list,
                 selected_features,
                 selected_labels,
                ):

        #Get feature
        self.x =  [torch.FloatTensor(df[selected_features].to_numpy()) for df in tile_info_list]
        
        # Get the Y labels
        self.y =  [torch.FloatTensor(df[selected_labels].drop_duplicates().to_numpy()) for df in tile_info_list]

        #Get tumor fraction
        self.tf = [torch.FloatTensor(df['TUMOR_PIXEL_PERC'].to_numpy()) for df in tile_info_list]
        
        #Get site loc
        self.site_loc = [torch.FloatTensor(df['SITE_LOCAL'].to_numpy()) for df in tile_info_list]

        #Get other info
        self.other_info = [df.drop(columns = selected_features + selected_labels + ['TUMOR_PIXEL_PERC']) for df in tile_info_list]
        
        #Train_orTest info
        self.fold0 = [df['FOLD0'].unique().item() for df in tile_info_list]
        self.fold1 = [df['FOLD1'].unique().item() for df in tile_info_list]
        self.fold2 = [df['FOLD2'].unique().item() for df in tile_info_list]
        self.fold3 = [df['FOLD3'].unique().item() for df in tile_info_list]
        self.fold4 = [df['FOLD4'].unique().item() for df in tile_info_list]
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x  = self.x[index]
        y  = self.y[index]
        tf = self.tf[index]
        site_loc = self.site_loc[index]
        of = self.other_info[index]
        sp_id = of['SAMPLE_ID'].unique().item()
        pt_id = of['PATIENT_ID'].unique().item()
        f0 = self.fold0[index]
        f1 = self.fold1[index]
        f2 = self.fold2[index]
        f3 = self.fold3[index]
        f4 = self.fold4[index]
        
        return {
        "x": x,
        "y": y,
        "tumor_fraction": tf,
        "site_location": site_loc,
        "tile_info": of,
        "sample_id": sp_id,
        "patient_id": pt_id,
        "fold0": f0,
        "fold1": f1,
        "fold2": f2,
        "fold3": f3,
        "fold4": f4
        }
                


class ModelReadyData_diffdim_withclusterinfo(Dataset):
    def __init__(self,
                 feature_list,
                 label_list,
                 tumor_info_list,
                 cluster_list,
                ):
        
        self.x =[torch.FloatTensor(feature) for feature in feature_list] 
        
        # Get the Y labels
        self.y = [torch.FloatTensor(label) for label in label_list] 

        self.tf = [torch.FloatTensor(tf) for tf in tumor_info_list] 

        self.c = [torch.FloatTensor(c) for c in cluster_list] 
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x = self.x[index]
        y = self.y[index]
        tf= self.tf[index]
        c= self.c[index]
        
        return x, y, tf,c

class ModelReadyData_Instance_based(Dataset):
    def __init__(self,
                 feature_data,
                 label_data,
                ):
            
        self.x = feature_data
        
        # Get the Y labels
        self.y = label_data
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self,index):
        # Given an index, return a tuple of an X with it's associated Y
        x = self.x[index]
        y = self.y[index]
        
        return x, y

def modify_to_instance_based(indata):
    feature_list = []
    label_list = []
    for data_it, data in enumerate(indata):
        cur_labels = data[1]
        label_list.append(cur_labels.repeat(data[0].shape[0], 1))
        feature_list.append(data[0].squeeze())
    
    feature_data = torch.concat(feature_list, dim = 0) #torch.Size([N_TILES, 1536])
    label_data = torch.concat(label_list, dim = 0) #torch.Size([N_TILES, 7])

    indata_instance_based = ModelReadyData_Instance_based(feature_data, label_data)

    return indata_instance_based
    
class add_tile_xy(Dataset):
    def __init__(self, original_dataset, additional_component):
        self.original_dataset = original_dataset
        self.additional_component = additional_component

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_sample = self.original_dataset[idx]
        tile_xy = torch.tensor(list(self.additional_component[idx]['TILE_XY_INDEXES'].apply(self.str_to_tuple)))

        # Assuming original_sample is a tuple
        return original_sample + (tile_xy,)

    # Function to convert string coordinates to tuples
    def str_to_tuple(self,coord_str):
        coord_str = coord_str.strip('()')
        x, y = map(int, coord_str.split(','))
        return (x, y)
    

def check_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        #print(f"Missing columns: {missing}")
        return True
    return False



def get_sample_feature(folder_name, feature_path, fe_method, cancer_info_path):    
    #Input dir
    input_dir = os.path.join(feature_path, 
                             folder_name, 
                             'features', 
                             f"features_alltiles_{fe_method}.h5")
    
    #feature
    feature_df = pd.read_hdf(input_dir, key='feature')
    feature_df.columns = feature_df.columns.astype(str)
    feature_df.reset_index(drop = True, inplace = True)
    
    #Get all tile info
    id_df = pd.read_hdf(input_dir, key='tile_info') #Tile ID (Do not use the labels in this, only use the tile info, because the label has been updated in all_tile_info_df from 3_otherinfo)
    id_df.reset_index(drop = True, inplace = True)
    
    #Combine
    if id_df.shape[0] != feature_df.shape[0]:
        raise ValueError("id_df and feature_df must have the same number of rows.")
    
    
    combined_df = pd.concat([id_df, feature_df], axis=1)
    
    
    #Check if tile info contains this 
    missing = check_columns(combined_df, ['pred_map_location', 'TUMOR_PIXEL_PERC'])
    
    if missing: #if missing get it from cancer detection
        cancer_info_dir = os.path.join(cancer_info_path, 
                                         folder_name, 
                                         'ft_model')
        file_name = [f for f in os.listdir(cancer_info_dir) if f.endswith("_TILE_TUMOR_PERC.csv")][0]
        cancer_df = pd.read_csv(os.path.join(cancer_info_dir, file_name))
        
        if combined_df.shape[0] != cancer_df.shape[0]:
            raise ValueError("combined_df and cancer_df must have the same number of rows.")
        
        combined_df = combined_df.merge(cancer_df, how = 'left', on = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 
                                                                       'PIXEL_OVERLAP','LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 
                                                                       'WHITE_SPACE','TISSUE_COVERAGE'])
        
  
    cols_to_keep = list(feature_df.columns) + ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 
                                         'PIXEL_OVERLAP','LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 
                                         'WHITE_SPACE','TISSUE_COVERAGE', 'pred_map_location', 'TUMOR_PIXEL_PERC']
    combined_df = combined_df[cols_to_keep] 

        
        
    return combined_df



def combine_features_label_allsamples(selected_ids, feature_path, fe_method, TUMOR_FRAC_THRES, all_label_df, id_col, cancer_info_path):
    
    all_comb = []
    ct = 0
    for pt in selected_ids:
        
        if ct % 10 == 0 : print(ct)
        #Get feature
        feature_df = get_sample_feature(pt, feature_path, fe_method, cancer_info_path)    
        
        #Select tumor fraction > X tiles
        feature_df = feature_df.loc[feature_df['TUMOR_PIXEL_PERC'] >= TUMOR_FRAC_THRES].copy()
        feature_df.reset_index(inplace = True, drop = True)
        
        #Get label df 
        label_df = all_label_df.loc[all_label_df[id_col] == pt]
        
        #Combine label
        comb_df = feature_df.merge(label_df, how = 'left', on = ["SAMPLE_ID"])
        
        
        all_comb.append(comb_df)
        ct += 1
    
    all_comb_df = pd.concat(all_comb)
    
    return all_comb_df, all_comb


def get_model_ready_data(datalist, fold_name = 'fold0', data_type = 'TRAIN', selected_label = 'HR1'):
    
    #Get label
    all_labels = ["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI"]
    label_idx = all_labels.index(selected_label) 
    
    #subset
    subsets = [item for item in datalist if item[fold_name] == data_type]
    
    data_tensor = [(item['x'], 
                    item['y'][:,[label_idx]],
                    item['tumor_fraction'],
                    item['site_location']) for item in subsets]

    sample_ids      = [item['sample_id'] for item in subsets]
    patient_ids     = [item['patient_id'] for item in subsets]
    corhor_names     = [re.split(r"[-_]", item['patient_id'])[0] for item in subsets]
    
    
    
    return data_tensor, sample_ids, patient_ids, corhor_names


def load_dataset_splits(ol100, ol0, fold, label):
    """Load train, validation, and test splits for a dataset."""
    train, train_sp_ids, train_pt_ids, train_cohorts = get_model_ready_data(
        ol100, f'fold{fold}', 'TRAIN', selected_label=label
    )
    val, val_sp_ids, val_pt_ids, val_cohorts = get_model_ready_data(
        ol100, f'fold{fold}', 'VALID', selected_label=label
    )
    test, test_sp_ids, test_pt_ids, test_cohorts = get_model_ready_data(
        ol0, f'fold{fold}', 'TEST', selected_label=label
    )
    
    return {
        "train": (train, train_sp_ids, train_pt_ids, train_cohorts),
        "val":   (val, val_sp_ids, val_pt_ids, val_cohorts),
        "test":  (test, test_sp_ids, test_pt_ids, test_cohorts),
    }



# --- prep helper: unique tile key + remember original row index
def prep(df, tf, dataset_id, key_cols=('SAMPLE_ID','PATIENT_ID','TILE_XY_INDEXES', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP',)):
    out = df.copy().reset_index(drop=True)
    out['original_row_idx'] = out.index
    out['dataset'] = dataset_id
    # tf: torch.Tensor or numpy/Series -> ensure Series for concat
    if isinstance(tf, torch.Tensor):
        tf = tf.detach().cpu().numpy()
    out['tumor_fraction'] = pd.Series(tf).reset_index(drop=True)
    # build a robust string key; adjust key_cols to your schema
    out['__key__'] = out[list(key_cols)].astype(str).agg('§'.join, axis=1)
    return out

def merge_data_lists(list1, list2, merge_type = 'union'):
    """
    For each tile, keep the feature from normed or not normed data, keep the one with the highest tumor fraction
    Keep, for each tile key, the row coming from d1 or d2 with the higher tumor_fraction.
    per-tile tensors (x, tumor_fraction, site_location) are aligned row-wise with tile_info.
    """

    #Convert data list into a dict keyed by sample id
    dict1 = {d['sample_id']: d for d in list1}
    dict2 = {d['sample_id']: d for d in list2}

    merged_list = []
    
    #All sample id
    all_ids = list(set(dict1.keys()) | set(dict2.keys()))
    all_ids.sort()

    for sid in all_ids:
                
        d1 = dict1.get(sid)
        d2 = dict2.get(sid)

        if d1 and d2: #if both exsit                
            
            if merge_type == 'union':
                #add tf and original row idx
                c1 = prep(d1['tile_info'], d1['tumor_fraction'], 1)
                c2 = prep(d2['tile_info'], d2['tumor_fraction'], 2)
                both = pd.concat([c1, c2], ignore_index=True)
                
                #pick the row with max tumor_fraction per tile key
                best_idx = both.groupby('__key__')['tumor_fraction'].idxmax() # idx of the max tumor_fraction within each key group
                chosen = both.loc[best_idx].reset_index(drop=True)
                
                # --- split chosen by source to slice tensors efficiently and in order
                chosen1 = chosen[chosen['dataset'] == 1]
                chosen2 = chosen[chosen['dataset'] == 2]
                idx1 = chosen1['original_row_idx'].tolist()
                idx2 = chosen2['original_row_idx'].tolist()
                
                
                # --- slice tensors (handle empty-index cases)
                parts_x = []
                parts_tf = []
                parts_sl = []
                ti_parts = []
            
                if len(idx1):
                    parts_x.append(d1['x'][idx1])
                    parts_tf.append(d1['tumor_fraction'][idx1])
                    parts_sl.append(d1['site_location'][idx1])
                    ti_parts.append(d1['tile_info'].iloc[idx1])
                    
                if len(idx2):
                    parts_x.append(d2['x'][idx2])
                    parts_tf.append(d2['tumor_fraction'][idx2])
                    parts_sl.append(d2['site_location'][idx2])
                    ti_parts.append(d2['tile_info'].iloc[idx2])
                    
                    
                X = torch.cat(parts_x, dim=0) if len(parts_x) > 1 else (parts_x[0] if parts_x else torch.empty(0))
                TF = torch.cat(parts_tf, dim=0) if len(parts_tf) > 1 else (parts_tf[0] if parts_tf else torch.empty(0))
                SL = torch.cat(parts_sl, dim=0) if len(parts_sl) > 1 else (parts_sl[0] if parts_sl else torch.empty(0))
                TI = pd.concat(ti_parts, axis=0).reset_index(drop=True)


        
            # --- final merged sample dict
            merged = {
                'x': X,
                'y': d1.get('y', d2.get('y')),  #keep whichever is available
                'tumor_fraction': TF,
                'site_location': SL,
                'tile_info': TI,                 # does NOT include helper columns
                'sample_id': d1.get('sample_id', d2.get('sample_id')),
                'patient_id': d1.get('patient_id', d2.get('patient_id')),
                'fold0': d1.get('fold0', d2.get('fold0')),
                'fold1': d1.get('fold1', d2.get('fold1')),
                'fold2': d1.get('fold2', d2.get('fold2')),
                'fold3': d1.get('fold3', d2.get('fold3')),
                'fold4': d1.get('fold4', d2.get('fold4')),
            }
            
            merged_list.append(merged)


    return merged_list



class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
