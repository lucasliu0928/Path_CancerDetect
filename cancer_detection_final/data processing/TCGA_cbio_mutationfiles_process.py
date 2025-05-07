#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:40:57 2025

@author: jliu6
"""

import pandas as pd



def extract_before_third_hyphen(id_string):
    parts = id_string.split('-')
    return '-'.join(parts[:3])

def clean_alteration_one_gene(alteration_df,gene_name, id_type = 'PATIENT_ID'):
    #Drop type of alteration if non ID has that type
    alteration_df = alteration_df.dropna(subset=alteration_df.columns[2:], how ='all')
    
    
    #Get alteration df for one gene
    df = alteration_df.loc[alteration_df['track_name'] == gene_name]
    
    #drop ID if no alteration
    df = df.dropna(axis=1, how='all')
    
    #Get track name (gene name) and types (CNA or mutation)
    track_name = df['track_name'].unique().item()
    track_type = df['track_type'].tolist()
    df.drop(columns = ['track_name','track_type'], inplace = True)
    
    
    #Tranpose the dataframe, rows for samples or patients
    df = df.T
    df.columns = track_type
    df.reset_index(inplace = True)
    df.rename(columns = {'index': id_type},inplace = True)
    
    
    #If more than two type ofs alteration, comeine them
    if len(track_type) == 1:
        df.rename(columns = {track_type[0]: 'Mutation_Type'},inplace = True)
    else:
        df["Mutation_Type"] = df.apply(
            lambda row: 
                f"{row['CNA']}, {row['MUTATIONS']}" if pd.notna(row["CNA"]) and pd.notna(row["MUTATIONS"])
                else row["CNA"] if pd.notna(row["CNA"])
                else row["MUTATIONS"] if pd.notna(row["MUTATIONS"])
                else pd.NA,
            axis=1
        )
        df.drop(columns = track_type, inplace = True)
        
    

    df['Track_name'] = track_name
    
    
    return df

# =============================================================================
# DIR
# =============================================================================
data_dir = "../../data/MutationCalls/TCGA_PRAD/Firehose Legacy/"


###################################################################################################
#Load clinical data
###################################################################################################
clinical_df = pd.read_csv(data_dir + 'prad_tcga_all_data/data_clinical_sample.txt', sep = '\t', header=4)


###################################################################################################
# #HR and MSI
###################################################################################################
# For Patinet DATA 
onco_df_patient = pd.read_csv(data_dir + 'raw_oncoprint/PATIENT_DATA_oncoprint_HR_MSI.tsv', sep = '\t')

HR_genes = ['BRCA2', 'BRCA1', 'PALB2', 'ATM', 'BARD1', 'BRIP1', 'CHEK2', 'NBN','RAD51C', 'RAD51D']
df_list = []
for gene in HR_genes:
    df = clean_alteration_one_gene(onco_df_patient, gene, id_type = 'PATIENT_ID')
    df_list.append(df)
df_HR = pd.concat(df_list)
df_HR['Pathway'] = 'HR'

msi_genes = ['MSH2', 'MSH6', 'PMS2', 'MLH1']
df_list = []
for gene in msi_genes:
    df = clean_alteration_one_gene(onco_df_patient, gene, id_type = 'PATIENT_ID')
    df_list.append(df)

df_msi = pd.concat(df_list)
df_msi['Pathway'] = 'MSI-H'


df_hr_msi_patient = pd.concat([df_HR, df_msi])


# For SAMPLE DATA
onco_df_sample = pd.read_csv(data_dir + 'raw_oncoprint/SAMPLE_DATA_oncoprint_HR_MSI.tsv', sep = '\t')


HR_genes = ['BRCA2', 'BRCA1', 'PALB2', 'ATM', 'BARD1', 'BRIP1', 'CHEK2', 'NBN','RAD51C', 'RAD51D']
df_list = []
for gene in HR_genes:
    df = clean_alteration_one_gene(onco_df_sample, gene, id_type = 'SAMPLE_ID')
    df_list.append(df)
df_HR = pd.concat(df_list)
df_HR['Pathway'] = 'HR'


msi_genes = ['MSH2', 'MSH6', 'PMS2', 'MLH1']
df_list = []
for gene in msi_genes:
    df = clean_alteration_one_gene(onco_df_sample, gene, id_type = 'SAMPLE_ID')
    df_list.append(df)
df_msi = pd.concat(df_list)
df_msi['Pathway'] = 'MSI-H'


df_hr_msi_sample = pd.concat([df_HR, df_msi])
df_hr_msi_sample['PATIENT_ID'] = df_hr_msi_sample['SAMPLE_ID'].apply(extract_before_third_hyphen)



# Merge Sample and Patient
df_merged = pd.merge(df_hr_msi_patient, df_hr_msi_sample, on= ['PATIENT_ID', 'Mutation_Type', 'Track_name', 'Pathway'], how='outer')
df_merged = df_merged[['PATIENT_ID', 'SAMPLE_ID', 'Pathway', 'Track_name' ,'Mutation_Type']]


# #Add SAMPLE_ID to reviewed HR and MSI
# Note in the previouse file， some smpales ATM, or PALB2 track name was copied wrong, now fixed
reviewed_df = pd.read_excel(data_dir + 'Digital_pathology_TCGA_Mutation_Check_Pritchard.xlsx')
reviewed_df.rename(columns = {'Patient ID': 'PATIENT_ID'}, inplace = True)
reviewed_df['Track_name'] = reviewed_df['Track_name'].str.strip()
reviewed_df_updated = reviewed_df.merge(df_merged, on = ['PATIENT_ID', 'Mutation_Type', 'Track_name', 'Pathway'])

#Add other clinical data
comb_df = reviewed_df_updated.merge(clinical_df, on = ['PATIENT_ID','SAMPLE_ID'], how = 'left')


comb_df.to_csv(data_dir + 'cleaned_final/Digital_pathology_TCGA_Mutation_HR_MSI_Pritchard_OtherInfoAdded.csv', index = False)





###################################################################################################
# AR, PTEN, RB1, TP53
###################################################################################################
# For Patinet DATA 
onco_df_patient = pd.read_csv(data_dir + 'raw_oncoprint/PATIENT_DATA_oncoprint_AR, PTEN, RB1, TP53.tsv', sep = '\t')

other_genes = ['AR', 'PTEN', 'RB1', 'TP53']
df_list = []
for gene in other_genes:
    df = clean_alteration_one_gene(onco_df_patient, gene, id_type = 'PATIENT_ID')
    df['Pathway'] = gene
    df_list.append(df)
df_other = pd.concat(df_list)


# For SAMPLE DATA
onco_df_sample = pd.read_csv(data_dir + 'raw_oncoprint/SAMPLE_DATA_oncoprint_AR, PTEN, RB1, TP53.tsv', sep = '\t')
other_genes = ['AR', 'PTEN', 'RB1', 'TP53']
df_list = []
for gene in other_genes:
    df = clean_alteration_one_gene(onco_df_sample, gene, id_type = 'SAMPLE_ID')
    df['Pathway'] = gene
    df_list.append(df)
df_other_sample = pd.concat(df_list)
df_other_sample['PATIENT_ID'] = df_other_sample['SAMPLE_ID'].apply(extract_before_third_hyphen)

# Merge Sample and Patient
df_merged = pd.merge(df_other, df_other_sample, on= ['PATIENT_ID', 'Mutation_Type', 'Track_name', 'Pathway'], how='outer')
df_merged = df_merged[['PATIENT_ID', 'SAMPLE_ID', 'Pathway', 'Track_name' ,'Mutation_Type']]


#Add other info to reviewed other mutation
reviewed_df = pd.read_excel(data_dir + 'Digital_pathology_TCGA_AR_PTEN_RB1_TP53_CP.xlsx')
reviewed_df['Track_name'] = reviewed_df['Track_name'].str.strip()
reviewed_df['Pathway'] = reviewed_df['Track_name']
reviewed_df_updated = reviewed_df.merge(df_merged, on = ['PATIENT_ID','SAMPLE_ID','Mutation_Type', 'Track_name', 'Pathway'])

#Add other clinical data
comb_df = reviewed_df_updated.merge(clinical_df, on = ['PATIENT_ID','SAMPLE_ID'], how = 'left')
comb_df.to_csv(data_dir + 'cleaned_final/Digital_pathology_TCGA_Mutation_AR_PTEN_RB1_TP53_CP_OtherInfoAdded.csv', index = False)

