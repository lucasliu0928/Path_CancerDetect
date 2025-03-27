#!/usr/bin/env python
# coding: utf-8
#NOTE: use paimg9 env

import sys
import os
#from fastai.vision.all import *
import torch
import pandas as pd
import warnings
sys.path.insert(0, '../Utils/')
from Preprocessing import preprocess_mutation_data
from Utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")


############################################################################################################
#USER INPUT 
############################################################################################################
pixel_overlap = 100      # specify the level of pixel overlap in your saved images
save_image_size = 250
TUMOR_FRAC_THRES = 0.9
cohort_name = "OPX"  #TAN_TMA_Cores, OPX
folder_name = "IMSIZE" + str(save_image_size) + "_OL" + str(pixel_overlap)
select_labels = ["AR",
                 "HR",
                 "PTEN",
                 "RB1",
                 "TP53",
                 "TMB_HIGHorINTERMEDITATE",
                 "MSI_POS"]

############################################################################################################
#DIR
############################################################################################################
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
wsi_location_opx = proj_dir + '/data/OPX/'
wsi_location_tan = proj_dir + 'data/TAN_TMA_Cores/'
wsi_location_ccola = proj_dir + '/data/CCola/all_slides/'
wsi_location_tcga = proj_dir + 'data/TCGA_PRAD/'
info_path  = os.path.join(proj_dir,'intermediate_data','2_cancer_detection', cohort_name, folder_name) #Old in cancer_prediction_results110224
label_path = os.path.join(proj_dir,'data','MutationCalls', cohort_name)
out_location = os.path.join(proj_dir,'intermediate_data','3_otherinfo', cohort_name, folder_name)
create_dir_if_not_exists(out_location)



##################
#Select GPU
##################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


############################################################################################################
#Load IDs that are used for finetune
############################################################################################################
#Get IDs that are in FT train or already processed to exclude 
fine_tune_ids_df = pd.read_csv(proj_dir + 'intermediate_data/0_cd_finetune/cancer_detection_training/all_tumor_fraction_info.csv')
ft_train_ids = list(fine_tune_ids_df.loc[fine_tune_ids_df['Train_OR_Test'] == 'Train','sample_id']) #24, 7 from OPX, 17 from ccola


################################################
#Get OPX IDs 
################################################
#All Aval IDs
opx_ids = [x.replace('.tif','') for x in os.listdir(wsi_location_opx)] #360

#Only Include IDs are high quality
label_df = pd.read_excel(os.path.join(label_path, "UWMC_OPX_Master Spreadsheet_Lucas.xlsx")) #274 Samples, 272 patient, #New data (there are some ids in old data exclude due to bad quality)
opx_ids_high_quality = list(label_df['OPX_Number'].unique()) 
selected_ids = opx_ids_high_quality #274

#Exclude Fine tuning opx id
opx_ids_ft_all = [x for x in ft_train_ids if 'OPX' in x] #7 (NOTE: 'OPX_031' is not in opx_ids_high_quality)
opx_ids_ft = [x for x in opx_ids_ft_all if x in opx_ids_high_quality] #Only get the IDs that is in opx_ids_high_quality
print("High Quality IDs Used for Fine Tune (n = ", len(opx_ids_ft), "):" ,opx_ids_ft) #6 #['OPX_010', 'OPX_024', 'OPX_040', 'OPX_047', 'OPX_057', 'OPX_088']
selected_ids = [x for x in selected_ids if x not in opx_ids_ft] #268

#Exclude IDs has no cancer detected
cd_aval_ids = [x for x in os.listdir(info_path) if x != '.DS_Store'] #353
cancer_detect_list = []
for cur_id in cd_aval_ids:
    cur_info_df = pd.read_csv(os.path.join(info_path, cur_id, 'ft_model',cur_id + "_TILE_TUMOR_PERC.csv"))
    cancer_detect_list.append(cur_info_df)
all_cd_df = pd.concat(cancer_detect_list)

#Filter for Cancer detected tiles > threshod
all_cd_df = all_cd_df.loc[all_cd_df['TUMOR_PIXEL_PERC'] >= TUMOR_FRAC_THRES] #555,533

#No Cancer IDs from high quality:
cancer_ids = list(set(all_cd_df['SAMPLE_ID']))
nocancer_ids = [x for x in cd_aval_ids if x not in cancer_ids]
print("Original IDs (without ft) has No Cancer detected (n = ", len(nocancer_ids), "):" ,nocancer_ids) #4 #['OPX_145', 'OPX_005', 'OPX_203', 'OPX_059']
selected_ids = [x for x in selected_ids if x not in nocancer_ids] #268 (No sample is excluded here, because the no cancer detected id is already removed from high quality)
selected_ids.sort()
print("Final OPX IDs (n = ", len(selected_ids), ")")


# toexclude_ids = ft_train_ids + ['cca3af0c-3e0e-4cfb-bb07-459c979a0bd5'] #The latter one is TCGA issue file

# #All available IDs
# ccola_ids = [x.replace('.svs','') for x in os.listdir(wsi_location_ccola) if '(2017-0133)' in x] #234
# tan_ids =  [x.replace('.tif','') for x in os.listdir(wsi_location_tan)] #677
# tcga_ids = [x.replace('.svs','') for x in os.listdir(wsi_location_tcga) if x != '.DS_Store'] #449

# elif cohort_name == "ccola":
#     all_ids = ccola_ids
# elif cohort_name == "TAN_TMA_Cores":
#     all_ids = tan_ids
# elif cohort_name == 'TCGA_PRAD':
#     all_ids = tcga_ids
# elif cohort_name == "all":
#     all_ids = opx_ids + ccola_ids + tan_ids + tcga_ids



if cohort_name == "OPX":
    ################################################
    #Preprocess label, site info and tile info
    ################################################
    label_df = preprocess_mutation_data(label_df, select_labels, hr_gene_list = ['BRCA1','BRCA2','PALB2'], id_col = 'OPX_Number')
    
    #TODO
    # label_df['HR/DDR (BRCA1, BRCA2, ATM, CHEK2, PALB2, BAP1, BARD1, RAD51C, RAD51D, FANCA, FANCD2, MRE11A, ATR, NBN, FANCM, FANCG)'].unique()
    # 'MRE11A', 'BRCA2', 'NBN', 'ATM', 'CHEK2', 'FANCA',
    #        'FANCA, ATM', 'BAP1', 'BRCA1', 'ATR', 'ATM ', 'FANCM',
    #        'PALB2, ATM', 'PALB2'
           
    # check = pd.read_excel("/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/data/MutationCalls/TCGA_PRAD/Digital_pathology_TCGA_Mutation_Check_Pritchard.xlsx")
    # check = check.loc[check['Colin Recommends Keep vs. Exclude'] =='Keep']
    # check = check.loc[check['Pathway'] == 'HR']
    # check['Track_name'].unique()
    #Intersect with OPX: 'BRCA2', 'BRCA1', 'PALB2', 'ATM', 'BARD1','CHEK2', 'NBN', 'RAD51C', 'RAD51D'
    #TCAGA has 'BRCA2', 'BRCA1', 'PALB2', 'ATM', 'BARD1','CHEK2', 'NBN', 'RAD51C', 'RAD51D', 'BRIP1' 
           
    ############################################################################################################
    #Combine site and label info and tile info
    ############################################################################################################
    tile_info_list = []
    for cur_id in selected_ids:
        cur_info_df = pd.read_csv(os.path.join(info_path, cur_id, 'ft_model',cur_id + "_TILE_TUMOR_PERC.csv"))
        cur_label_df = label_df.loc[label_df['SAMPLE_ID'] == cur_id]
        cur_comb_df = cur_info_df.merge(label_df, on = ['SAMPLE_ID'],how = 'left') #add label
        tile_info_list.append(cur_comb_df)
    all_tile_info_df = pd.concat(tile_info_list)
    print(all_tile_info_df.shape) #1743458 tiles overlap0, 4843073 tiles overlap100
    
    #Print stats
    tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
    print("Total OPX IDs in tile path: ", len(set(all_tile_info_df['SAMPLE_ID']))) #3375102 tiles in total
    print("Max # tile/per pt:", tile_counts.max()) #34689
    print("Min # tile/per pt:", tile_counts.min()) #98
    print("Median # tile/per pt:", tile_counts.median()) #1809.5

elif cohort_name == 'TAN_TMA_Cores':
    ################################################
    #Load TAN_TMA mutation label data
    ################################################
    label_df1 = pd.read_excel(label_path + "TAN97_core_mappings.xlsx") #These Ids not in label_df2: ['18-018', '18-087', '18-064', '18-077', '08-016', '06-131']
    label_df1.rename(columns = {'AR': 'AR_inMappingFile'}, inplace = True)
    label_df1.loc[pd.isna(label_df1['AR pos']),'AR pos'] = 0
    label_df1.loc[pd.isna(label_df1['NE pos']),'NE pos'] = 0
    
    label_df2 = pd.read_excel(label_path + "TAN_coded mutation_for Roman.xlsx") 
    #Rename as OPX annotation 
    label_df2.rename(columns = {'AR coded': 'AR',
                               'CHD1 coded': 'CHD1',
                               'PTEN coded': 'PTEN',
                               'RB1 coded': 'RB1',
                               'TP53 coded': 'TP53', 
                               'BRCA2 coded':'BRCA2'}, inplace = True)
    
    
    #Combine
    #Only keep the ids in TAN_coded mutation_for Roman.xlsx, because no mutation labels are aviabale , cannot say it is negative
    label_df = label_df1.merge(label_df2, left_on = ['ptid'], right_on = ['Sample'], how = 'right')
    label_df.reset_index(drop=True, inplace=True)
    
    #There 40 sample IDs does not have matched AR status
    checkAR = label_df.loc[label_df['AR pos'] != label_df['AR'],]
    print(len(set(checkAR['Sample'])))
    checkAR.to_csv(out_location + "AR_notmatch.csv", index = False)
    
    
    #Recode SITE info
    label_df['SITE_LOCAL'] = pd.NA
    cond = label_df['ORGAN SITE'] == 'PROSTATE'
    label_df.loc[cond,'SITE_LOCAL'] = 1
    label_df.loc[~cond,'SITE_LOCAL'] = 0
    
    label_df.rename(columns = {'TMA-row-col': 'SAMPLE_ID'}, inplace= True)
    
    ############################################################################################################
    #Add site and label info into tile info
    ############################################################################################################
    tile_info_list = []
    for cur_id in selected_ids:
        cur_tile_info_df = pd.read_csv(os.path.join(tile_info_path, cur_id, cur_id + "_tiles.csv"))
        cur_comb_df = cur_tile_info_df.merge(label_df, on = ['SAMPLE_ID'],how = 'left') #add label
        tile_info_list.append(cur_comb_df)
    all_tile_info_df = pd.concat(tile_info_list)
    print(all_tile_info_df.shape) #146888 tiles overlap0
    
    #Print stats
    tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
    print("Total IDs in tile path: ", len(set(all_tile_info_df['SAMPLE_ID']))) #3375102 tiles in total
    print("Max # tile/per pt:", tile_counts.max()) #311
    print("Min # tile/per pt:", tile_counts.min()) #5
    print("Median # tile/per pt:", tile_counts.median()) #233.0

elif cohort_name == "TCGA_PRAD":
    ################################################
    #Load mutation label data
    #TODO: This need to be replaced with actual data
    ################################################
    label_df = pd.DataFrame({'TCGA_ID': tcga_ids})
    label_df[['Results',
           'Limited Study (low tumor content/quality), YES/NO', 'MSI (POS/NEG)',
           'TMB (HIGH/LOW/INTERMEDIATE)',
           'MMR (MSH2, MSH6, PMS2, MLH1, MSH3, MLH3, EPCAM)',
           'MMR (MSH2, MSH6, PMS2, MLH1, MSH3, MLH3, EPCAM)2', 'CDK12', 'SPOP',
           'CHD1',
           'Chromatin remodeling other (KDM6A, KMT2A, KMT2C, KMT2D, PBRM1, ASXL1, ASXL2, SMARCB1, SMARCA4)',
           'PTEN', 'PI3K other (PIK3CA, PIK3CB, AKT1, PIK3R1, MAPK1, MAP2K2)',
           'ETS fusion', 'TP53', 'RB1',
           'Cell Cycle Other (CCND1, CCNE1, CDKN2A/B, CDK4)',
           'RAS/RAF (BRAF, KRAS, HRAS, NRAS)', 'AR',
           'AR pathway other (ZBTB16, FOXA1)', 'WNT (APC, CTNNB1, RNF43, RSPO3)',
           'TGF-beta (SMAD2, SMAD4, TGFRBR2, ACVR1)', 'GATA2', 'MYC', 'MED12',
           'MTOR pathway (MTOR, RPTOR, TSC1, TSC2)', 'IDH (IDH1, IDH2)', 'Other']] = pd.NA
    
    #Combined
    label_df = preprocess_mutation_data(label_df, id_col = 'TCGA_ID')
    label_df.reset_index(drop=True, inplace=True)
    
    ################################################
    #Load Site data
    #TODO: need replace with actual data
    ################################################
    site_df = pd.DataFrame({'TCGA_ID': tcga_ids})
    site_df[['Bx Type', 'Anatomic site', 'Notes']] = pd.NA
    site_df.reset_index(drop=True, inplace=True)
    site_df = preprocess_site_data(site_df, id_col = 'TCGA_ID')

    ############################################################################################################
    #Add site and label info into tile info
    ############################################################################################################
    tile_info_list = []
    for cur_id in selected_ids:
        cur_slides_name = [f for f in os.listdir(tile_info_path + cur_id + '/') if '.csv' in f][0].replace('_tiles.csv','')
        cur_tile_info_df = pd.read_csv(os.path.join(tile_info_path, cur_id, cur_slides_name + "_tiles.csv"))
        cur_tile_info_df['SAMPLE_ID'] = cur_slides_name
        cur_comb_df = cur_tile_info_df.merge(label_df, on = ['SAMPLE_ID'],how = 'left') #add label
        cur_comb_df = cur_comb_df.merge(site_df, on = ['SAMPLE_ID'], how = 'left') #add site
        tile_info_list.append(cur_comb_df)
    all_tile_info_df = pd.concat(tile_info_list)
    print(all_tile_info_df.shape) #1308050 tiles overlap0, 3633199 tiles overlap100
    
    #Print stats
    tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
    print("Total IDs in tile path: ", len(set(all_tile_info_df['SAMPLE_ID']))) #3375102 tiles in total
    print("Max # tile/per pt:", tile_counts.max()) #34689
    print("Min # tile/per pt:", tile_counts.min()) #43
    print("Median # tile/per pt:", tile_counts.median()) #1570.5



all_tile_info_df[all_tile_info_df['TUMOR_PIXEL_PERC']>=0.9].shape #(1389408, 20) #for overlap100, (555533, 20) for overlap0


#Output
#This file contains all tiles without cancer fraction exclusion and  has tissue membership > 0.9, white space < 0.9 (non white space > 0.1)
all_tile_info_df.to_csv(os.path.join(out_location, "all_tile_info.csv"), index = False)

