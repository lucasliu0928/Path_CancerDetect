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


def combine_data_from_stnorm_and_nostnorm(indata_stnorm, indata_stnorm_no, method = 'union'):
    
    # indata_stnorm = data_ol100_opx_stnorm0
    # indata_stnorm_no = data_ol100_opx_stnorm1
    
    comb_data_list = []
    #get the data with more IDs
    if len(indata_stnorm) >= len(indata_stnorm_no):
        main_data = indata_stnorm
        nomain_data = indata_stnorm_no
    else:
        main_data = indata_stnorm_no
        nomain_data = indata_stnorm

    for i in range(len(main_data)):

        # Unpack the tuple
        features1, labels1, tf1, other_info1, sample_id1, patient_id1 = main_data[i]
        
        #find the data in indata2
        index = next((i for i, entry in enumerate(nomain_data) if entry[-2] == sample_id1), None)
        
    
        
        if index is not None: #if the ID is in both data, take the union or combine
            features2, labels2, tf2, other_info2, sample_id2, patient_id2 = nomain_data[index]
            if method == 'combine_all':
                comb_data = (
                    features1,
                    labels1,
                    tf1,
                    other_info1,
                    sample_id1,
                    patient_id1,
                    
                    features2,
                    labels2,
                    tf2,
                    other_info2,
                    sample_id2,
                    patient_id2,
                )
            elif method == 'union':
                #Intersect
                common_tiles = set(other_info1['TILE_COOR_ATLV0']).intersection(other_info2['TILE_COOR_ATLV0'])
                match_index1, nomatch_index1 = get_matching_tile_index(other_info1, common_tiles)
                match_index2, nomatch_index2 = get_matching_tile_index(other_info2, common_tiles)
                match_index1_tokeep, match_index2_tokeep = get_larger_tumor_fraction_tile(tf1, tf2,match_index1, match_index2)
                
                final_idx_tokeep1 = match_index1_tokeep + nomatch_index1
                final_idx_tokeep2 = match_index2_tokeep + nomatch_index2
                
                
                #Get updated info
                feature = torch.concat([features1[final_idx_tokeep1], features2[final_idx_tokeep2]])
                tf = torch.concat([tf1[final_idx_tokeep1], tf2[final_idx_tokeep2]])
                labels = labels1 
                other_info = pd.concat([other_info1.iloc[final_idx_tokeep1], other_info2.iloc[final_idx_tokeep2]])
                sample_id = sample_id1
                patient_id = patient_id1
                
                comb_data = (
                    feature,
                    labels,
                    tf,
                    other_info,
                    sample_id,
                    patient_id
                )

        else: #only take all the things in main data
            if method == 'combine_all':
                comb_data = (
                    features1,
                    labels1,
                    tf1,
                    other_info1,
                    sample_id1,
                    patient_id1,
                    [],
                    [],
                    [],
                    [],
                    [],
                    []
                    
                )
            elif method == 'union':
                 comb_data = (
                     features1,
                     labels1,
                     tf1,
                     other_info1,
                     sample_id1,
                     patient_id1
                 )
                 
        comb_data_list.append(comb_data)
        
    return comb_data_list   


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