#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 03:27:29 2024

@author: jliu6
"""
import pandas as pd

def extract_before_second_underscore(s):
    parts = s.split('_')
    if len(parts) > 2:
        return '_'.join(parts[:2])
    return s
    
def preprocess_mutation_data(indata, select_labels, hr_gene_list = ['BRCA1','BRCA2','PALB2'], id_col = 'OPX_Number'):    
    #Rename ID col
    indata.rename(columns = {id_col: 'SAMPLE_ID'}, inplace = True)
    
    #Add Patient ID
    indata['PATIENT_ID'] = indata['SAMPLE_ID'].apply(extract_before_second_underscore)
    
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
    indata.loc[:, other_cols] = indata.loc[:, other_cols] .notna().astype(int)
    
    #Recode
    indata['SITE_LOCAL'] = pd.NA
    cond = indata['Anatomic site'] == 'Prostate'
    indata.loc[cond,'SITE_LOCAL'] = 1
    indata.loc[~cond,'SITE_LOCAL'] = 0
    
    #Drop extra column
    indata = indata[['PATIENT_ID','SAMPLE_ID','SITE_LOCAL'] + select_labels]
    
    return indata
