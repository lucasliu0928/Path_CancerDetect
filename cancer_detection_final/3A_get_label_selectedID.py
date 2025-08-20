#!/usr/bin/env python
# coding: utf-8
#NOTE: use paimg9 env

import sys
import os
import torch
import pandas as pd
import warnings
sys.path.insert(0, '../Utils/')
from Preprocessing import combine_sampleinfo_and_label_all, get_cancer_detected_ids, concat_hr1_and_hr2_label
from Preprocessing import prepross_neptune_label_data, extract_before_second_underscore, preproposs_tcga_label
from misc_utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")
import argparse

############################################################################################################
#USER INPUT 
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--pixel_overlap', default='100',   type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--save_image_size', default='250', type=int, help='the size of extracted tiles')
parser.add_argument('--cohort_name', default='z_nostnorm_TCGA_PRAD', type=str, help='data set name: TAN_TMA_Cores or OPX or TCGA_PRAD or Neptune, or z_nostnorm_Neptune')
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--out_folder', default= '3A_otherinfo', type=str, help='out folder name')

args = parser.parse_args()

############################################################################################################
#USER INPUT 
############################################################################################################
folder_name = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.pixel_overlap)
selected_hr_genes1 = ['BRCA2', 'BRCA1', 'PALB2', 'ATM', 'BARD1','CHEK2', 'NBN', 'RAD51C', 'RAD51D'] #Intersection TCGA and OPX: 'BRCA2', 'BRCA1', 'PALB2', 'ATM', 'BARD1','CHEK2', 'NBN', 'RAD51C', 'RAD51D'
selected_hr_genes2 = ['BRCA2', 'BRCA1', 'PALB2']
selected_msi_genes = ['MSH2', 'MSH6', 'PMS2', 'MLH1']

############################################################################################################
#DIR
############################################################################################################
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
wsi_location = proj_dir +  'data/' + args.cohort_name.replace("z_nostnorm_", "") + "/"
info_path  = os.path.join(proj_dir,'intermediate_data','2_cancer_detection', args.cohort_name, folder_name) #Old in cancer_prediction_results110224
label_path = os.path.join(proj_dir,'data','MutationCalls', args.cohort_name.replace("z_nostnorm_", ""))
out_location = os.path.join(proj_dir,'intermediate_data',args.out_folder, args.cohort_name, folder_name)
create_dir_if_not_exists(out_location)

##################
#Select GPU
##################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
if args.cohort_name.replace("z_nostnorm_", "") == "OPX":
    ################################################
    #Get OPX IDs 
    ################################################
    #All Aval IDs
    opx_ids = [x.replace('.tif','') for x in os.listdir(wsi_location)] #360
    selected_ids = opx_ids
    
    #Include IDs are high quality
    label_df = pd.read_excel(os.path.join(label_path, "UWMC_OPX_Master Spreadsheet_Lucas.xlsx")) #274 Samples, 272 patient, #New data (there are some ids in old data exclude due to bad quality)
    ids_high_quality = list(label_df['OPX_Number'].unique()) 
    selected_ids = [x for x in selected_ids if x in ids_high_quality] #274
    

    #Exclude IDs that are in finetune train to exclude 
    fine_tune_ids_df = pd.read_csv(proj_dir + 'intermediate_data/0_cd_finetune/cancer_detection_training/all_tumor_fraction_info.csv')
    ft_train_ids = list(fine_tune_ids_df.loc[fine_tune_ids_df['Train_OR_Test'] == 'Train','sample_id']) #24, 7 from OPX, 17 from ccola
    selected_ids = [x for x in selected_ids if x not in ft_train_ids] #268


    #Include Cancer detected IDs 
    cancer_ids = get_cancer_detected_ids(info_path, args.TUMOR_FRAC_THRES, id_col = 'SAMPLE_ID')
    selected_ids = [x for x in selected_ids if x in cancer_ids] #268
    selected_ids.sort()
    

    ################################################
    #Preprocess label
    ################################################
    #Rename ID col
    label_df.rename(columns = {'OPX_Number': 'SAMPLE_ID'}, inplace = True)
    label_df['PATIENT_ID'] = label_df['SAMPLE_ID'].apply(extract_before_second_underscore)
    label_df['FOLDER_ID'] = label_df['SAMPLE_ID']
    
    #Concatenate HR1 and HR2 label
    label_df = concat_hr1_and_hr2_label(label_df, selected_hr_genes1, selected_hr_genes2, id_col = 'SAMPLE_ID') 

    #filter ids
    label_df = label_df.loc[label_df['SAMPLE_ID'].isin(selected_ids)]

    #output
    label_df.to_csv(os.path.join(out_location, "all_sample_label_df.csv"), index = False)
    #Correlation
    cor_df = label_df[["AR","HR1","HR2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]].corr()
    cor_df.to_csv(os.path.join(out_location, "all_sample_labelcoor_df.csv"))
    
    print("Final OPX SAMPLE IDs (n = ", len(label_df['SAMPLE_ID'].unique()), ")") #268, 204
    print("Final OPX PATIENT IDs (n = ", len(label_df['PATIENT_ID'].unique()), ")") #266, 201

    ############################################################################################################
    #Combine site and label info and tile info
    ############################################################################################################       
    all_tile_info_df = combine_sampleinfo_and_label_all(info_path, label_df, selected_ids, id_col = 'SAMPLE_ID')  
    print(all_tile_info_df.shape) #1743458 tiles overlap0, 4843073 tiles overlap100
    print("Total OPX SP IDs in tile path: ", len(set(all_tile_info_df['SAMPLE_ID']))) #268,204
    print("Total OPX PT IDs in tile path: ", len(set(all_tile_info_df['PATIENT_ID']))) #266,201

    #Print stats
    tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
    print("Max # tile/per sp:", tile_counts.max()) #34689, 96406
    print("Min # tile/per sp:", tile_counts.min()) #98, 285
    print("Median # tile/per sp:", tile_counts.median()) #1809.5,5132
    

elif args.cohort_name.replace("z_nostnorm_", "") == 'TAN_TMA_Cores':
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


elif args.cohort_name.replace("z_nostnorm_", "") == "TCGA_PRAD":    
    ################################################
    #Get TCGA IDs 
    ################################################
    #All Aval IDs
    tcga_ids = [x.replace('.svs','') for x in os.listdir(wsi_location) if x != '.DS_Store'] #449
    selected_ids = tcga_ids
    
    #Exclude tissue issue
    issue_ids = ['cca3af0c-3e0e-4cfb-bb07-459c979a0bd5']
    selected_ids = [x for x in selected_ids if x not in issue_ids]
    
    #include Cancer detected IDs
    cancer_ids = get_cancer_detected_ids(info_path, args.TUMOR_FRAC_THRES, id_col = 'FOLDER_ID', cohort_name = args.cohort_name)
    selected_ids = [x for x in selected_ids if x in cancer_ids] #446
    selected_ids.sort()
       
    ################################################
    #Load mutation label data
    #NOTE: the TCGA folder ID != UUIDs in label file
    ################################################
    label_df = preproposs_tcga_label(label_path, info_path, selected_ids, selected_hr_genes1, selected_hr_genes2, selected_msi_genes)
    
    #output
    label_df.to_csv(os.path.join(out_location, "all_sample_label_df.csv"), index = False)
    #Correlation
    cor_df = label_df[["AR","HR1","HR2","PTEN","RB1","TP53","MSI_POS"]].corr()
    cor_df.to_csv(os.path.join(out_location, "all_sample_labelcoor_df.csv"))
    
    print("Final TCGA SAMPLE IDs (n = ", len(label_df['SAMPLE_ID'].unique()), ")") #207, 204
    print("Final TCGA PATIENT IDs (n = ", len(label_df['PATIENT_ID'].unique()), ")") #204, 201
    


    ############################################################################################################
    #Combine site and label info and tile info
    ############################################################################################################       
    all_tile_info_df = combine_sampleinfo_and_label_all(info_path, label_df, selected_ids, id_col = 'SAMPLE_ID', cohort_name = args.cohort_name)  
    print(all_tile_info_df.shape) #5964499 tiles overlap0, 16570195 tiles overlap100
    print("Total TCGA SP IDs in tile path: ", len(set(all_tile_info_df['SAMPLE_ID']))) #207,204
    print("Total TCGA PT IDs in tile path: ", len(set(all_tile_info_df['PATIENT_ID']))) #204,201

    #Print stats
    tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
    print("Max # tile/per pt:", tile_counts.max()) #45419,126346
    print("Min # tile/per pt:", tile_counts.min()) #294, 826
    print("Median # tile/per pt:", tile_counts.median()) #13726.5, 38110

    

elif args.cohort_name.replace("z_nostnorm_", "") == "Neptune":
    ################################################
    #Get Neptune IDs 
    ################################################
    #All Aval IDs
    nep_ids = [x.replace('.tif','') for x in os.listdir(wsi_location) if x != '.DS_Store'] #350
    selected_ids = nep_ids
    
    #Load label to get high quality ids
    label_df = prepross_neptune_label_data(label_path, "Neptune_ES_ver2_updated_rescan_ids.xlsx")
    #include High quality ids
    ids_high_quality = list(label_df['SAMPLE_ID'].unique()) 
    selected_ids = [x for x in selected_ids if x in ids_high_quality] #209

    #include Cancer detected IDs
    cancer_ids = get_cancer_detected_ids(info_path, args.TUMOR_FRAC_THRES, id_col = 'SAMPLE_ID')
    selected_ids = [x for x in selected_ids if x in cancer_ids] #207
    selected_ids.sort()
    
    
    ################################################
    #Preprocess label
    ################################################    
    #Concatenate HR1 and HR2 label
    label_df = concat_hr1_and_hr2_label(label_df, selected_hr_genes1, selected_hr_genes2, id_col = 'SAMPLE_ID') 
    #filter ids
    label_df = label_df.loc[label_df['SAMPLE_ID'].isin(selected_ids)]
    #output
    label_df.to_csv(os.path.join(out_location, "all_sample_label_df.csv"), index = False)
    #Correlation
    cor_df = label_df[["AR","HR1","HR2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]].corr()
    cor_df.to_csv(os.path.join(out_location, "all_sample_labelcoor_df.csv"))
    
    print("Final Nep SAMPLE IDs (n = ", len(label_df['SAMPLE_ID'].unique()), ")") #207, 204
    print("Final Nep PATIENT IDs (n = ", len(label_df['PATIENT_ID'].unique()), ")") #204, 201
    

    ############################################################################################################
    #Combine site and label info and tile info
    ############################################################################################################       
    all_tile_info_df = combine_sampleinfo_and_label_all(info_path, label_df, selected_ids, id_col = 'SAMPLE_ID')  
    print(all_tile_info_df.shape) #658112 tiles overlap0, #1818673 tiles overlap100
    print("Total Nep SP IDs in tile path: ", len(set(all_tile_info_df['SAMPLE_ID']))) #207,204
    print("Total Nep PT IDs in tile path: ", len(set(all_tile_info_df['PATIENT_ID']))) #204,201

    #Print stats
    tile_counts = all_tile_info_df['SAMPLE_ID'].value_counts()
    print("Max # tile/per sp:", tile_counts.max()) #35004, 97269
    print("Min # tile/per sp:", tile_counts.min()) #54, 148
    print("Median # tile/per sp:", tile_counts.median()) #1394, 3864
    

############################################################################################################
#Output Tile info
############################################################################################################
#This file contains all tiles without cancer fraction exclusion and  has tissue membership > 0.9, white space < 0.9 (non white space > 0.1)
#OPX:   1743458 for overlap0, 4841982 #for overlap100,   
#TCGA (no stain normed) :  5964499 for overlap0, 16570195 #for overlap100,
#TCGA:  5958125 for overlap0, 16570195 #for overlap100,
#TMA: 80630 for oeverlap0
#Neptune: 653182 for oeverlap0, 1818673 for overlap100
all_tile_info_df.to_csv(os.path.join(out_location, "all_tile_info.csv"), index = False)
print(all_tile_info_df.shape)

#Jsut check tumor tile numbers:
#OPX    524370 for overlap0,   1332241 for overlap100
#OPX  (no stain normed)   555533 for overlap0,    4843073 for overlap100
#TCGA  1298540 for overlap0,   3317865 for overlap100
#TCGA (no stain normed)  1307688 for overlap0,   3335532 for overlap100
#TMA    18328 for overlap0
#Neptune    150806 for overlap0, 346566 for oeverlap100
print(all_tile_info_df[all_tile_info_df['TUMOR_PIXEL_PERC']>=0.9].shape)

