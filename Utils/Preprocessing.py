#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 03:27:29 2024

@author: jliu6
"""
import pandas as pd


def preprocess_mutation_data(indata):

    #Rename ID col
    indata.rename(columns = {'OPX_Number': 'SAMPLE_ID'}, inplace = True)

    #Recode, 1: mutation, 0: no mutation
    indata.iloc[:, 5:] = indata.iloc[:, 5:].notna().astype(int)
    
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

    #Drop extra column
    indata = indata.drop(columns = ['Results',
                           'Limited Study (low tumor content/quality), YES/NO',
                           'TMB (HIGH/LOW/INTERMEDIATE)',
                           'MSI (POS/NEG)'])

    return indata


def preprocess_site_data(indata):

    #Rename ID col
    indata.rename(columns = {'OPX_Number': 'SAMPLE_ID'}, inplace = True)

    #Recode MSI,1: POS, 0: NEG/NA
    indata['SITE_LOCAL'] = pd.NA
    cond = indata['Anatomic site'] == 'Prostate'
    indata.loc[cond,'SITE_LOCAL'] = 1
    indata.loc[~cond,'SITE_LOCAL'] = 0
    
    return indata