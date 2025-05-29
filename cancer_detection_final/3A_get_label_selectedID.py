#!/usr/bin/env python
# coding: utf-8
#NOTE: use paimg9 env

import sys
import os
import torch
import pandas as pd
import warnings
sys.path.insert(0, '../Utils/')
from Preprocessing import preprocess_mutation_data
from train_utils import extract_before_third_hyphen
from Utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")
import argparse

############################################################################################################
#USER INPUT 
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--pixel_overlap', default='100', type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--save_image_size', default='250', type=int, help='the size of extracted tiles')
parser.add_argument('--cohort_name', default='TCGA_PRAD', type=str, help='data set name: TAN_TMA_Cores or OPX or TCGA_PRAD or Neptune')
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--out_folder', default= '3A_otherinfo', type=str, help='out folder name')

args = parser.parse_args()

############################################################################################################
#USER INPUT 
############################################################################################################
folder_name = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.pixel_overlap)
select_labels = ["AR","HR","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]
selected_hr_genes1 = ['BRCA2', 'BRCA1', 'PALB2', 'ATM', 'BARD1','CHEK2', 'NBN', 'RAD51C', 'RAD51D'] #Intersection TCGA and OPX: 'BRCA2', 'BRCA1', 'PALB2', 'ATM', 'BARD1','CHEK2', 'NBN', 'RAD51C', 'RAD51D'
selected_hr_genes2 = ['BRCA2', 'BRCA1', 'PALB2']
selected_msi_genes = ['MSH2', 'MSH6', 'PMS2', 'MLH1']

############################################################################################################
#DIR
############################################################################################################
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
wsi_location = proj_dir +  'data/' + args.cohort_name + "/"
info_path  = os.path.join(proj_dir,'intermediate_data','2_cancer_detection', args.cohort_name, folder_name) #Old in cancer_prediction_results110224
label_path = os.path.join(proj_dir,'data','MutationCalls', args.cohort_name)
out_location = os.path.join(proj_dir,'intermediate_data',args.out_folder, args.cohort_name, folder_name)
create_dir_if_not_exists(out_location)



##################
#Select GPU
##################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if args.cohort_name == "OPX":
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
    opx_ids = [x.replace('.tif','') for x in os.listdir(wsi_location)] #360

    #Only Include IDs are high quality
    label_df_all = pd.read_excel(os.path.join(label_path, "UWMC_OPX_Master Spreadsheet_Lucas.xlsx")) #274 Samples, 272 patient, #New data (there are some ids in old data exclude due to bad quality)
    opx_ids_high_quality = list(label_df_all['OPX_Number'].unique()) 
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
    all_cd_df = all_cd_df.loc[all_cd_df['TUMOR_PIXEL_PERC'] >= args.TUMOR_FRAC_THRES] #555,533

    #No Cancer IDs from high quality:
    cancer_ids = list(set(all_cd_df['SAMPLE_ID']))
    nocancer_ids = [x for x in cd_aval_ids if x not in cancer_ids]
    print("Original IDs (without ft) has No Cancer detected (n = ", len(nocancer_ids), "):" ,nocancer_ids) #4 #['OPX_145', 'OPX_005', 'OPX_203', 'OPX_059']
    selected_ids = [x for x in selected_ids if x not in nocancer_ids] #268 (No sample is excluded here, because the no cancer detected id is already removed from high quality)
    selected_ids.sort()
    print("Final OPX IDs (n = ", len(selected_ids), ")")
    
    ################################################
    #Preprocess label, site info and tile info
    ################################################
    label_df1 = preprocess_mutation_data(label_df_all, select_labels, hr_gene_list = selected_hr_genes1, id_col = 'OPX_Number')
    label_df1 = label_df1.rename(columns = {'HR': 'HR1'}).copy()
    label_df2 = preprocess_mutation_data(label_df_all, select_labels, hr_gene_list = selected_hr_genes2, id_col = 'OPX_Number')
    label_df2 = label_df2.rename(columns = {'HR': 'HR2'}).copy()
    label_df = label_df1.merge(label_df2, on = ['PATIENT_ID', 'SAMPLE_ID', 
                                     'SITE_LOCAL', 'AR', 
                                     'PTEN', 'RB1', 'TP53', 
                                     'TMB_HIGHorINTERMEDITATE', 'MSI_POS'])
    label_df = label_df[['PATIENT_ID','SAMPLE_ID','SITE_LOCAL', 'AR', 'HR1', 'HR2','PTEN', 'RB1',
                         'TP53', 'TMB_HIGHorINTERMEDITATE', 'MSI_POS']]


    ############################################################################################################
    #Combine site and label info and tile info
    ############################################################################################################     
    tile_info_list = []
    for cur_id in selected_ids:
        cur_info_df = pd.read_csv(os.path.join(info_path, cur_id, 'ft_model',cur_id + "_TILE_TUMOR_PERC.csv"))
        cur_label_df = label_df.loc[label_df['SAMPLE_ID'] == cur_id]
        cur_comb_df = cur_info_df.merge(cur_label_df, on = ['SAMPLE_ID'],how = 'left') #add label
        tile_info_list.append(cur_comb_df)
    all_tile_info_df = pd.concat(tile_info_list)
    print(all_tile_info_df.shape) #1743458 tiles overlap0, 4843073 tiles overlap100
    
    #Print stats
    tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
    print("Total OPX IDs in tile path: ", len(set(all_tile_info_df['SAMPLE_ID']))) #268
    print("Max # tile/per pt:", tile_counts.max()) #34689, 96406
    print("Min # tile/per pt:", tile_counts.min()) #98, 285
    print("Median # tile/per pt:", tile_counts.median()) #1809.5,5011

elif args.cohort_name == 'TAN_TMA_Cores':
    #All Aval IDs
    tan_ids =  [x.replace('.tif','') for x in os.listdir(wsi_location)] #677
    
    #Load TAN_TMA mutation label data
    label_df1 = pd.read_excel(os.path.join(label_path, "TAN97_core_mappings.xlsx")) #These Ids not in label_df2: ['18-018', '18-087', '18-064', '18-077', '08-016', '06-131']
    label_df1.rename(columns = {'AR': 'AR_inMappingFile'}, inplace = True)
    label_df1.loc[pd.isna(label_df1['AR pos']),'AR pos'] = 0
    label_df1.loc[pd.isna(label_df1['NE pos']),'NE pos'] = 0
    label_df2 = pd.read_excel(os.path.join(label_path, "TAN_coded mutation_for Roman.xlsx"))
    label_df2.rename(columns = {'AR coded': 'AR',
                               'CHD1 coded': 'CHD1',
                               'PTEN coded': 'PTEN',
                               'RB1 coded': 'RB1',
                               'TP53 coded': 'TP53', 
                               'BRCA2 coded':'BRCA2'}, inplace = True)
    #Combine to get core ids
    #Only keep the ids in TAN_coded mutation_for Roman.xlsx, because no mutation labels are aviabale , cannot say it is negative
    label_df = label_df1.merge(label_df2, left_on = ['ptid'], right_on = ['Sample'], how = 'right')
    label_df.reset_index(drop=True, inplace=True)
        
    # #There 40 sample IDs does not have matched AR status
    # checkAR = label_df.loc[label_df['AR pos'] != label_df['AR'],]
    # print(len(set(checkAR['Sample'])))
    # checkAR.to_csv(out_location + "AR_notmatch.csv", index = False)
    
    
    #Recode SITE info
    label_df['SITE_LOCAL'] = pd.NA
    cond = label_df['ORGAN SITE'] == 'PROSTATE'
    label_df.loc[cond,'SITE_LOCAL'] = 1
    label_df.loc[~cond,'SITE_LOCAL'] = 0
    
    #Rename sample id column
    label_df.rename(columns = {'TMA-row-col': 'SAMPLE_ID'}, inplace= True)
    
    #Only select ID that is in label file
    selected_ids = [x for x in tan_ids if x in list(label_df['SAMPLE_ID'].unique())] #596
    
    #Exclude IDs has no cancer detected
    cd_aval_ids = [x for x in os.listdir(info_path) if x != '.DS_Store'] #677
    cancer_detect_list = []
    for cur_id in cd_aval_ids:
        cur_info_df = pd.read_csv(os.path.join(info_path, cur_id, 'ft_model',cur_id + "_TILE_TUMOR_PERC.csv"))
        cancer_detect_list.append(cur_info_df)
    all_cd_df = pd.concat(cancer_detect_list) #146888
    
    
    #Filter for Cancer detected tiles > threshod
    all_cd_df = all_cd_df.loc[all_cd_df['TUMOR_PIXEL_PERC'] >= args.TUMOR_FRAC_THRES] #19098

    #No Cancer IDs 
    cancer_ids = list(set(all_cd_df['SAMPLE_ID']))
    nocancer_ids = [x for x in cd_aval_ids if x not in cancer_ids] #299
    print("No Cancer detected (n = ", len(nocancer_ids), ")") 
    selected_ids = [x for x in selected_ids if x not in  nocancer_ids] 
    selected_ids.sort()
    print("Final TCGA IDs (n = ", len(selected_ids), ")") #355
    

    ############################################################################################################
    #Combine site and label info and tile info
    ############################################################################################################     
    tile_info_list = []
    for cur_id in selected_ids:
        cur_info_df = pd.read_csv(os.path.join(info_path, cur_id, 'ft_model',cur_id + "_TILE_TUMOR_PERC.csv"))
        cur_label_df = label_df.loc[label_df['SAMPLE_ID'] == cur_id]
        cur_comb_df = cur_info_df.merge(cur_label_df, on = ['SAMPLE_ID'],how = 'left') #add label
        tile_info_list.append(cur_comb_df)
    all_tile_info_df = pd.concat(tile_info_list)
    print(all_tile_info_df.shape) #80630 tiles overlap0
    
    #Print stats
    tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
    print("Total OPX IDs in tile path: ", len(set(all_tile_info_df['SAMPLE_ID']))) #355
    print("Max # tile/per pt:", tile_counts.max()) #279
    print("Min # tile/per pt:", tile_counts.min()) #54
    print("Median # tile/per pt:", tile_counts.median()) #236


elif args.cohort_name == "TCGA_PRAD":
    ################################################
    #Get TCGA IDs 
    ################################################
    selected_ids = [x.replace('.svs','') for x in os.listdir(wsi_location) if x != '.DS_Store'] #449
    
    #Exclude tissue issue
    issue_ids = ['cca3af0c-3e0e-4cfb-bb07-459c979a0bd5']
    
    #Exclude IDs has no cancer detected
    cd_aval_ids = [x for x in os.listdir(info_path) if x != '.DS_Store'] #448 #Tissue ID is excluded here: cca3af0c-3e0e-4cfb-bb07-459c979a0bd5
    cancer_detect_list = []
    for cur_id in cd_aval_ids:
        cur_info_path = os.path.join(info_path, cur_id,'ft_model')
        cur_slides_name = [f for f in os.listdir(cur_info_path + '/') if '.csv' in f][0].replace('_TILE_TUMOR_PERC.csv','')
        cur_info_df = pd.read_csv(os.path.join(cur_info_path, cur_slides_name + "_TILE_TUMOR_PERC.csv"))    
        cur_info_df['TCGA_FOLDER_ID'] = cur_id
        cancer_detect_list.append(cur_info_df)
    all_cd_df = pd.concat(cancer_detect_list)

    #Filter for Cancer detected tiles > threshod
    all_cd_df = all_cd_df.loc[all_cd_df['TUMOR_PIXEL_PERC'] >= args.TUMOR_FRAC_THRES] #1307688, 3335532

    #No Cancer IDs 
    cancer_ids = list(set(all_cd_df['TCGA_FOLDER_ID']))
    nocancer_ids = [x for x in cd_aval_ids if x not in cancer_ids]
    print("Original IDs has No Cancer detected (n = ", len(nocancer_ids), "):" ,nocancer_ids) #1: ['2c6fbdb0-2fbb-4881-aa2e-ad3627665576']
    selected_ids = [x for x in selected_ids if x not in issue_ids + nocancer_ids] #447 (No sample is excluded here, because the no cancer detected id is already removed from high quality)
    selected_ids.sort()
    print("Final TCGA IDs (n = ", len(selected_ids), ")") #447
    
    
    
    ################################################
    #Load mutation label data
    #NOTE: the TCGA folder ID != UUIDs in label file
    ################################################
    HRMSI_df_all = pd.read_csv(os.path.join(label_path, "Firehose Legacy/cleaned_final/Digital_pathology_TCGA_Mutation_HR_MSI_Pritchard_OtherInfoAdded.csv")) 
    HRMSI_df_all = HRMSI_df_all.loc[HRMSI_df_all['Colin Recommends Keep vs. Exclude'] == 'Keep']
    HRMSI_df_all = HRMSI_df_all[['PATIENT_ID','Pathway','Track_name','SAMPLE_TYPE']]
    hr_df1 = HRMSI_df_all.loc[HRMSI_df_all['Track_name'].isin(selected_hr_genes1)].copy()
    hr_df1['Pathway'] = 'HR1'
    hr_df2 = HRMSI_df_all.loc[HRMSI_df_all['Track_name'].isin(selected_hr_genes2)].copy()
    hr_df2['Pathway'] = 'HR2'
    hr_df = pd.concat([hr_df1,hr_df2])
    msi_df = HRMSI_df_all.loc[HRMSI_df_all['Track_name'].isin(selected_msi_genes)].copy()
    HRMSI_df = pd.concat([hr_df,msi_df])
    other_df = pd.read_csv(os.path.join(label_path, "Firehose Legacy/cleaned_final/Digital_pathology_TCGA_Mutation_AR_PTEN_RB1_TP53_CP_OtherInfoAdded.csv")) 
    other_df = other_df[['PATIENT_ID','Pathway','Track_name','SAMPLE_TYPE']]
    all_mutation_df = pd.concat([HRMSI_df,other_df])
    mut_pathways = all_mutation_df['Pathway'].unique().tolist() 

    #Initiate label_df with TCGA folder ID, pateint ID and slide ID
    label_df = pd.DataFrame({'TCGA_FOLDER_ID': selected_ids})
    label_df[mut_pathways] = pd.NA
    slide_ids = []
    labels = []
    for cur_id in selected_ids:
        cur_info_path = os.path.join(info_path, cur_id,'ft_model')
        cur_slides_name = [f for f in os.listdir(cur_info_path + '/') if '.csv' in f][0].replace('_TILE_TUMOR_PERC.csv','')
        slide_ids.append(cur_slides_name)
    label_df['SLIDE_ID'] =    slide_ids
    label_df['PATIENT_ID'] =  label_df['SLIDE_ID'].apply(extract_before_third_hyphen)


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
      
    #Rename
    label_df.rename(columns = {'MSI-H':'MSI_POS'}, inplace = True)
    
    ################################################
    #LCombine label and clinical data
    ################################################
    #Load clinical
    clinical_df = pd.read_csv(label_path + '/Firehose Legacy/prad_tcga_all_data/data_clinical_sample.txt', sep = '\t', header=4)
    clinical_df = clinical_df.loc[clinical_df['SAMPLE_ID'] != 'TCGA-V1-A9O5-06'] #NOTE: this is a duplicate patient, the patient other slide was included, not this one
    clinical_df = clinical_df[['PATIENT_ID','SAMPLE_TYPE','SAMPLE_ID','OTHER_SAMPLE_ID',
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
    
    #reorder
    label_df = label_df[['TCGA_FOLDER_ID','PATIENT_ID','SLIDE_ID','SITE_LOCAL', 'AR', 'HR1', 'HR2','PTEN', 'RB1',
                         'TP53', 'TMB_HIGHorINTERMEDITATE', 'MSI_POS']]
    

    ############################################################################################################
    #Combine site and label info and tile info
    ############################################################################################################
    tile_info_list = []
    for cur_id in selected_ids:
        cur_info_path = os.path.join(info_path, cur_id,'ft_model')
        cur_slides_name = [f for f in os.listdir(cur_info_path + '/') if '.csv' in f][0].replace('_TILE_TUMOR_PERC.csv','')
        cur_info_df = pd.read_csv(os.path.join(cur_info_path, cur_slides_name + "_TILE_TUMOR_PERC.csv"))
        cur_label_df = label_df.loc[label_df['TCGA_FOLDER_ID'] == cur_id]
        cur_comb_df = cur_info_df.merge(cur_label_df, left_on= ['SAMPLE_ID'], right_on = ['SLIDE_ID'], how = 'left') #add label
        tile_info_list.append(cur_comb_df)
    all_tile_info_df = pd.concat(tile_info_list)
        
    print(all_tile_info_df.shape) #5964499 tiles overlap0, 16570195 tiles overlap100
    
    #Print stats
    tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
    print("Total IDs in tile path: ", len(set(all_tile_info_df['SAMPLE_ID']))) #447
    print("Max # tile/per pt:", tile_counts.max()) #45419,126346
    print("Min # tile/per pt:", tile_counts.min()) #294, 826
    print("Median # tile/per pt:", tile_counts.median()) #13683, 38110
    

elif args.cohort_name == "Neptune":
    ################################################
    #Get Neptune IDs 
    ################################################
    #All Aval IDs
    nep_ids = [x.replace('.tif','') for x in os.listdir(wsi_location) if x != '.DS_Store'] #350

    #Only Include IDs are high quality
    label_df_all = pd.read_excel(os.path.join(label_path, "Neptune_ES_ver2_updated_rescan_ids.xlsx")) #274 Samples, 272 patient, #New data (there are some ids in old data exclude due to bad quality)
    label_df_all.rename(columns = {'Slide ID': 'SAMPLE_ID',
                                   'Neptune ID': 'PATIENT_ID',
                                   'MSI..POS.NEG.': 'MSI (POS/NEG)',
                                   'TMB..HIGH.LOW.INTERMEDIATE.': 'TMB (HIGH/LOW/INTERMEDIATE)',
                                   'HR.DDR..BRCA1..BRCA2..ATM..CHEK2..PALB2..BAP1..BARD1..RAD51C..RAD51D..FANCA..FANCD2..MRE11A..ATR..NBN..FANCM..FANCG.':
                                       'HR/DDR (BRCA1, BRCA2, ATM, CHEK2, PALB2, BAP1, BARD1, RAD51C, RAD51D, FANCA, FANCD2, MRE11A, ATR, NBN, FANCM, FANCG)'}, inplace = True)
    label_df_all['Anatomic site'] = ''
    label_df_all['SAMPLE_ID'] = label_df_all['SAMPLE_ID'].str.replace('.tif', '', regex=False)
    ids_high_quality = list(label_df_all['SAMPLE_ID'].unique())  #209
    selected_ids = ids_high_quality #209

    #Exclude IDs has no cancer detected
    cd_aval_ids = [x for x in os.listdir(info_path) if x != '.DS_Store'] #350
    cancer_detect_list = []
    for cur_id in cd_aval_ids:
        cur_info_df = pd.read_csv(os.path.join(info_path, cur_id, 'ft_model',cur_id + "_TILE_TUMOR_PERC.csv"))
        cancer_detect_list.append(cur_info_df)
    all_cd_df = pd.concat(cancer_detect_list)

    #Filter for Cancer detected tiles > threshod
    all_cd_df = all_cd_df.loc[all_cd_df['TUMOR_PIXEL_PERC'] >= args.TUMOR_FRAC_THRES] #555,533

    #No Cancer IDs from high quality:
    cancer_ids = list(set(all_cd_df['SAMPLE_ID']))
    nocancer_ids = [x for x in cd_aval_ids if x not in cancer_ids]
    print("Original IDs has No Cancer detected (n = ", len(nocancer_ids), "):" ,nocancer_ids) #4 #['OPX_145', 'OPX_005', 'OPX_203', 'OPX_059']
    selected_ids = [x for x in selected_ids if x not in nocancer_ids] #268 (No sample is excluded here, because the no cancer detected id is already removed from high quality)
    selected_ids.sort()
    print("Final Nep IDs (n = ", len(selected_ids), ")") #202
    

    ################################################
    #Preprocess label, site info and tile info
    ################################################
    label_df1 = preprocess_mutation_data(label_df_all, select_labels, hr_gene_list = selected_hr_genes1, id_col = 'SAMPLE_ID')
    label_df1 = label_df1.rename(columns = {'HR': 'HR1'}).copy()
    label_df2 = preprocess_mutation_data(label_df_all, select_labels, hr_gene_list = selected_hr_genes2, id_col = 'SAMPLE_ID')
    label_df2 = label_df2.rename(columns = {'HR': 'HR2'}).copy()
    label_df = label_df1.merge(label_df2, on = ['PATIENT_ID', 'SAMPLE_ID', 
                                     'SITE_LOCAL', 'AR', 
                                     'PTEN', 'RB1', 'TP53', 
                                     'TMB_HIGHorINTERMEDITATE', 'MSI_POS'])
    label_df = label_df[['PATIENT_ID','SAMPLE_ID','SITE_LOCAL', 'AR', 'HR1', 'HR2','PTEN', 'RB1',
                         'TP53', 'TMB_HIGHorINTERMEDITATE', 'MSI_POS']]
    
    ############################################################################################################
    #Combine site and label info and tile info
    ############################################################################################################     
    tile_info_list = []
    for cur_id in selected_ids:
        cur_info_df = pd.read_csv(os.path.join(info_path, cur_id, 'ft_model',cur_id + "_TILE_TUMOR_PERC.csv"))
        cur_label_df = label_df.loc[label_df['SAMPLE_ID'] == cur_id]
        cur_comb_df = cur_info_df.merge(cur_label_df, on = ['SAMPLE_ID'],how = 'left') #add label
        tile_info_list.append(cur_comb_df)
    all_tile_info_df = pd.concat(tile_info_list)
    print(all_tile_info_df.shape) #653182 tiles overlap0
    
    #Print stats
    tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
    print("Total Nep IDs in tile path: ", len(set(all_tile_info_df['SAMPLE_ID']))) #202
    print("Max # tile/per pt:", tile_counts.max()) #34689, 96406
    print("Min # tile/per pt:", tile_counts.min()) #98, 285
    print("Median # tile/per pt:", tile_counts.median()) #1809, 5132
    


#Output
#This file contains all tiles without cancer fraction exclusion and  has tissue membership > 0.9, white space < 0.9 (non white space > 0.1)
#OPX:   1743458 for overlap0, 4841982 #for overlap100,   
#TCGA (no stain normed) :  5964499 for overlap0, 16570195 #for overlap100,
#TCGA:  5958125 for overlap0, 16570195 #for overlap100,
#TMA: 80630 for oeverlap0
#Neptune: 653182 for oeverlap0, 1809448 for overlap100
all_tile_info_df.to_csv(os.path.join(out_location, "all_tile_info.csv"), index = False)


#Jsut check tumor tile numbers:
#OPX    524370 for overlap0,   1332241 for overlap100
#TCGA (no stain normed)  1307688 for overlap0,   3335532 for overlap100
#TCGA  1298540 for overlap0,   3317865 for overlap100
#TMA    18328 for overlap0
#Neptune    150806 for overlap0, 399753 for oeverlap100
print(all_tile_info_df[all_tile_info_df['TUMOR_PIXEL_PERC']>=0.9].shape)
