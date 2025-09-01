#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 03:27:29 2024

@author: jliu6
"""
import pandas as pd
import os
from train_utils import extract_before_third_hyphen


def extract_before_second_underscore(s):
    parts = s.split('_')
    if len(parts) > 2:
        return '_'.join(parts[:2])
    return s

def extract_before_third_dash(s):
    parts = s.split('-')
    if len(parts) > 2:
        return '-'.join(parts[:3])
    return s


def build_slide_patient_df_tcga(id_path):

    folder_ids, slide_ids, patient_ids = [], [], []

    # Iterate over folders, ignoring hidden files like .DS_Store
    for folder in os.listdir(id_path):
        if folder.startswith("."):
            continue

        folder_path = os.path.join(id_path, folder)

        # Look for .svs files inside the folder
        svs_files = [f for f in os.listdir(folder_path) if f.endswith(".svs")]

        # Use the first .svs file as the slide ID source
        slide_id = svs_files[0].replace(".svs", "")
        patient_id = extract_before_third_hyphen(slide_id)

        # Append aligned values
        folder_ids.append(folder)
        slide_ids.append(slide_id)
        patient_ids.append(patient_id)

    # Build DataFrame
    df = pd.DataFrame({
        "FOLDER_ID": folder_ids,
        "SLIDE_ID": slide_ids,
        "PATIENT_ID": patient_ids
    })

    return df


#OPX specific    
def preprocess_mutation_data(indata, hr_gene_list = ['BRCA1','BRCA2','PALB2'], id_col = 'SAMPLE_ID'):    
    
    indata = indata.copy()
    
    #selected_labels
    select_labels = ["AR","HR","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]
    
    
    #Recode HR
    ori_hr_col = 'HR/DDR (BRCA1, BRCA2, ATM, CHEK2, PALB2, BAP1, BARD1, RAD51C, RAD51D, FANCA, FANCD2, MRE11A, ATR, NBN, FANCM, FANCG)'
    new_hr_col = 'HR'
    cond = indata[ori_hr_col].str.contains('|'.join(hr_gene_list)) == True
    indata[new_hr_col] = pd.NA
    indata.loc[cond,new_hr_col] = 1
    indata.loc[~cond,new_hr_col] = 0
    
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
    
    #Recode others:  1: mutation, 0: no mutation
    other_cols = [x for x in select_labels if x not in ['HR','MSI_POS','TMB_HIGHorINTERMEDITATE']]
    indata.loc[:, other_cols] = indata.loc[:, other_cols].notna().astype(int)
    
    #Recode
    indata['SITE_LOCAL'] = pd.NA
    cond = indata['Anatomic site'] == 'Prostate'
    indata.loc[cond,'SITE_LOCAL'] = 1
    indata.loc[~cond,'SITE_LOCAL'] = 0
    
    #Drop extra column
    indata = indata[['PATIENT_ID','SAMPLE_ID','FOLDER_ID','SITE_LOCAL'] + select_labels]
    
    return indata

def concat_hr1_and_hr2_label(all_label_df, hr_gene1, hr_gene2, id_col = 'SAMPLE_ID'):
    label_df1 = preprocess_mutation_data(all_label_df, hr_gene_list = hr_gene1, id_col = id_col)
    label_df1 = label_df1.rename(columns = {'HR': 'HR1'}).copy()
    label_df2 = preprocess_mutation_data(all_label_df, hr_gene_list = hr_gene2, id_col = id_col)
    label_df2 = label_df2.rename(columns = {'HR': 'HR2'}).copy()
    label_df = label_df1.merge(label_df2, on = ['PATIENT_ID', 'SAMPLE_ID', 'FOLDER_ID' ,
                                                'SITE_LOCAL', 'AR', 
                                                'PTEN', 'RB1', 'TP53', 
                                                'TMB_HIGHorINTERMEDITATE', 'MSI_POS'])
    label_df = label_df[['PATIENT_ID','SAMPLE_ID','FOLDER_ID','SITE_LOCAL', 'AR', 'HR1', 'HR2','PTEN', 'RB1',
                         'TP53', 'TMB_HIGHorINTERMEDITATE', 'MSI_POS']]
    
    return label_df

def combine_sampleinfo_and_label(tile_info_path, label_df, folder_id, slide_id, id_col = 'SAMPLE_ID'):
    
    sample_info  = pd.read_csv(os.path.join(tile_info_path, folder_id, 'ft_model',slide_id + "_TILE_TUMOR_PERC.csv"))
    sample_label = label_df.loc[label_df[id_col] == slide_id]
    sample_comb  = sample_info.merge(sample_label, on = [id_col], how = 'left') #add label
        
    return sample_comb

def combine_sampleinfo_and_label_all(tile_info_path, label_df, ids, id_col = 'SAMPLE_ID', cohort_name = ''):
    
    tile_info_list = []
    for cur_id in ids:
        if 'TCGA' in cohort_name:
            cur_slide_id = [f for f in os.listdir(os.path.join(tile_info_path, cur_id,'ft_model') + '/') if '.csv' in f][0].replace('_TILE_TUMOR_PERC.csv','')
        else:
            cur_slide_id = cur_id
            
        cur_comb_df = combine_sampleinfo_and_label(tile_info_path, label_df, folder_id = cur_id, slide_id = cur_slide_id, id_col = id_col)
        tile_info_list.append(cur_comb_df)
    all_tile_info = pd.concat(tile_info_list)
    
    return all_tile_info


def get_cancer_detected_ids(info_dir, tumor_frac_thres, id_col = 'SAMPLE_ID', cohort_name = ''):
    
    cd_aval_ids = [x for x in os.listdir(info_dir) if x != '.DS_Store'] 
    cancer_detect_list = []
    for cur_id in cd_aval_ids:
        
        if 'TCGA' in cohort_name:
            cur_slide_id = [f for f in os.listdir(os.path.join(info_dir, cur_id, 'ft_model') + '/') if '.csv' in f][0].replace('_TILE_TUMOR_PERC.csv','')
        else:
            cur_slide_id = cur_id
            
        cur_info_df = pd.read_csv(os.path.join(info_dir, cur_id, 'ft_model',cur_slide_id + "_TILE_TUMOR_PERC.csv"))
        
        if 'TCGA' in cohort_name:
            cur_info_df['FOLDER_ID'] = cur_id
        
        cancer_detect_list.append(cur_info_df)
    all_cd_df = pd.concat(cancer_detect_list)

    #Filter for Cancer detected tiles > threshod
    all_cd_df = all_cd_df.loc[all_cd_df['TUMOR_PIXEL_PERC'] >= tumor_frac_thres] 
    
    #Cancer detected IDs
    cancer_detected_ids = list(set(all_cd_df[id_col]))
    
    return cancer_detected_ids


#NEPtune specific
def prepross_neptune_label_data(label_dir, label_file_name):
    label_df = pd.read_excel(os.path.join(label_dir, label_file_name)) 
    label_df.rename(columns = {'Slide ID': 'SAMPLE_ID',
                                   'Neptune ID': 'PATIENT_ID',
                                   'MSI..POS.NEG.': 'MSI (POS/NEG)',
                                   'TMB..HIGH.LOW.INTERMEDIATE.': 'TMB (HIGH/LOW/INTERMEDIATE)',
                                   'HR.DDR..BRCA1..BRCA2..ATM..CHEK2..PALB2..BAP1..BARD1..RAD51C..RAD51D..FANCA..FANCD2..MRE11A..ATR..NBN..FANCM..FANCG.':
                                    'HR/DDR (BRCA1, BRCA2, ATM, CHEK2, PALB2, BAP1, BARD1, RAD51C, RAD51D, FANCA, FANCD2, MRE11A, ATR, NBN, FANCM, FANCG)'}, inplace = True)
    label_df['Anatomic site'] = ''
    label_df['SAMPLE_ID'] = label_df['SAMPLE_ID'].str.replace('.tif', '', regex=False)
    label_df['FOLDER_ID'] = label_df['SAMPLE_ID']

    return label_df







#TCGA specific
def preproposs_tcga_label(label_path, id_path, hr_gene1, hr_gene2, msi_genes):    
    #HR and MSI
    HRMSI_df_all = pd.read_csv(os.path.join(label_path, "Firehose Legacy/cleaned_final/Digital_pathology_TCGA_Mutation_HR_MSI_Pritchard_OtherInfoAdded.csv")) 
    HRMSI_df_all = HRMSI_df_all.loc[HRMSI_df_all['Colin Recommends Keep vs. Exclude'] == 'Keep']
    HRMSI_df_all = HRMSI_df_all[['PATIENT_ID','Pathway','Track_name','SAMPLE_TYPE']]
    
    hr_df1 = HRMSI_df_all.loc[HRMSI_df_all['Track_name'].isin(hr_gene1)].copy()
    hr_df1['Pathway'] = 'HR1'
    hr_df2 = HRMSI_df_all.loc[HRMSI_df_all['Track_name'].isin(hr_gene2)].copy()
    hr_df2['Pathway'] = 'HR2'
    hr_df = pd.concat([hr_df1,hr_df2])
    msi_df = HRMSI_df_all.loc[HRMSI_df_all['Track_name'].isin(msi_genes)].copy()
    HRMSI_df = pd.concat([hr_df,msi_df])
    
    #Other 
    other_df = pd.read_csv(os.path.join(label_path, "Firehose Legacy/cleaned_final/Digital_pathology_TCGA_Mutation_AR_PTEN_RB1_TP53_CP_OtherInfoAdded.csv")) 
    other_df = other_df[['PATIENT_ID','Pathway','Track_name','SAMPLE_TYPE']]
    
    #Combine
    all_mutation_df = pd.concat([HRMSI_df,other_df])
    
    
    #Get all pathway names
    mut_pathways = all_mutation_df['Pathway'].unique().tolist() 

    #Initiate label_df with TCGA folder ID, pateint ID and slide ID
    label_df = build_slide_patient_df_tcga(id_path)
    label_df[mut_pathways] = pd.NA

    #Fill in label_df
    for l in mut_pathways:
        mutated_ids = all_mutation_df.loc[all_mutation_df['Pathway'] == l,'PATIENT_ID'].unique().tolist()
        for i, (idx, row) in enumerate(label_df.iterrows()):
            cur_id = row['PATIENT_ID']
            if cur_id in mutated_ids:
                label_df.loc[idx, l] = 1
            else:
                label_df.loc[idx, l] = 0
    label_df.reset_index(drop=True, inplace=True)
      

    #Load clinical
    clinical_df = pd.read_csv(label_path + '/Firehose Legacy/prad_tcga_all_data/data_clinical_sample.txt', sep = '\t', header=4)
    clinical_df = clinical_df.loc[clinical_df['SAMPLE_ID'] != 'TCGA-V1-A9O5-06'] #NOTE: this is a duplicate patient, the patient other slide was included, not this one
    clinical_df = clinical_df[['PATIENT_ID','SAMPLE_TYPE','OTHER_SAMPLE_ID',
                               'PATHOLOGY_REPORT_FILE_NAME','PATHOLOGY_REPORT_UUID','TMB_NONSYNONYMOUS']]
    #Combine 
    label_df = label_df.merge(clinical_df, on = ['PATIENT_ID'], how = 'left')
    
    #Recode SITE info
    label_df['SITE_LOCAL'] = pd.NA
    cond = label_df['SAMPLE_TYPE'] == 'Primary'
    label_df.loc[cond,'SITE_LOCAL'] = 1
    label_df.loc[~cond,'SITE_LOCAL'] = 0
    
    #ADD TMB
    #TODO: this need to confirm maybe use label_df['TMB_NONSYNONYMOUS']
    label_df['TMB_HIGHorINTERMEDITATE'] = pd.NA
    
    #rename
    label_df.rename(columns = {'MSI-H':'MSI_POS'}, inplace = True)
    label_df.rename(columns = {'SLIDE_ID': 'SAMPLE_ID'},inplace = True)
    
    #reorder
    label_df = label_df[['PATIENT_ID','SAMPLE_ID','FOLDER_ID','SITE_LOCAL', 'AR', 'HR1', 'HR2','PTEN', 'RB1',
                         'TP53', 'TMB_HIGHorINTERMEDITATE', 'MSI_POS']]

    
    return label_df




