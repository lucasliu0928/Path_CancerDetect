#!/usr/bin/env python
# coding: utf-8
#NOTE: use paimg9 env

import sys
import os
import torch
import pandas as pd
import warnings
sys.path.insert(0, '../Utils/')
from Preprocessing import get_cancer_detected_ids, concat_hr1_and_hr2_label
from Preprocessing import prepross_neptune_label_data, extract_before_second_underscore, preproposs_tcga_label
from Preprocessing import build_slide_patient_df_tcga
from misc_utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")
import argparse

############################################################################################################
#USER INPUT 
############################################################################################################
parser = argparse.ArgumentParser("Preprocessing Label")
parser.add_argument('--cohort_name', default='Neptune', type=str, help='data set name: TAN_TMA_Cores or OPX or TCGA_PRAD or Neptune, or z_nostnorm_Neptune')
parser.add_argument('--out_folder', default= '3_labels', type=str, help='out folder name')

args = parser.parse_args()


############################################################################################################
#USER INPUT 
############################################################################################################
#selected_hr_genes1 = ['BRCA2', 'BRCA1', 'PALB2', 'ATM', 'BARD1','CHEK2', 'NBN', 'RAD51C', 'RAD51D'] #Intersection TCGA and OPX: 'BRCA2', 'BRCA1', 'PALB2', 'ATM', 'BARD1','CHEK2', 'NBN', 'RAD51C', 'RAD51D'
selected_hr_genes1 = ['BRCA1', 'BRCA2', 'ATM', 'CHEK2', 'PALB2', 'BAP1', 'BARD1', 'RAD51C', 'RAD51D', 'FANCA', 'FANCD2', 'MRE11A', 'ATR', 'NBN', 'FANCM', 'FANCG'] #oriingal sets in OPX
selected_hr_genes2 = ['BRCA2', 'BRCA1', 'PALB2']
selected_msi_genes = ['MSH2', 'MSH6', 'PMS2', 'MLH1']


############################################################################################################
#DIR
############################################################################################################
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
label_path = os.path.join(proj_dir,'data','MutationCalls', args.cohort_name)
id_path    = os.path.join(proj_dir,'data', args.cohort_name) 
out_location = os.path.join(proj_dir,'intermediate_data',
                            args.out_folder, 
                            args.cohort_name)
create_dir_if_not_exists(out_location)

##################
#Select GPU
##################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
if "OPX" in args.cohort_name:
    
    #Load label df
    label_df = pd.read_excel(os.path.join(label_path, "UWMC_OPX_Master Spreadsheet_Lucas.xlsx")) #274 Samples, 272 patient, #New data (there are some ids in old data exclude due to bad quality)

    
    #Preprocess label
    label_df.rename(columns = {'OPX_Number': 'SAMPLE_ID'}, inplace = True)
    label_df['PATIENT_ID'] = label_df['SAMPLE_ID'].apply(extract_before_second_underscore)
    label_df['FOLDER_ID'] = label_df['SAMPLE_ID']
    
    #Concatenate HR1 and HR2 label
    label_df = concat_hr1_and_hr2_label(label_df, selected_hr_genes1, selected_hr_genes2, id_col = 'SAMPLE_ID') 


    #output
    label_df.to_csv(os.path.join(out_location, "label_df_allslides.csv"), index = False)
    
    #Correlation
    cor_df = label_df[["AR","HR1","HR2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]].corr()
    cor_df.to_csv(os.path.join(out_location, "label_coor_df_allslides.csv"))



elif "TCGA_PRAD" in args.cohort_name:    
    #Load  label df and preproess
    label_df = preproposs_tcga_label(label_path, id_path, selected_hr_genes1, selected_hr_genes2, selected_msi_genes)
    label_df.to_csv(os.path.join(out_location, "label_df_allslides.csv"), index = False)

    #Correlation
    cor_df = label_df[["AR","HR1","HR2","PTEN","RB1","TP53","MSI_POS"]].corr()
    cor_df.to_csv(os.path.join(out_location, "label_coor_df_allslides.csv"))


elif "Neptune" in args.cohort_name:

    #Load label df and preprocess
    label_df = prepross_neptune_label_data(label_path, "Neptune_ES_ver2_updated_rescan_ids.xlsx")

    
    #Concatenate HR1 and HR2 label
    label_df = concat_hr1_and_hr2_label(label_df, selected_hr_genes1, selected_hr_genes2, id_col = 'SAMPLE_ID') 
    label_df.to_csv(os.path.join(out_location, "label_df_allslides.csv"), index = False)
    
    #Correlation
    cor_df = label_df[["AR","HR1","HR2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]].corr()
    cor_df.to_csv(os.path.join(out_location, "label_coor_df_allslides.csv"))



# elif args.cohort_name.replace("z_nostnorm_", "") == 'TAN_TMA_Cores':
#     #All Aval IDs
#     tan_ids =  [x.replace('.tif','') for x in os.listdir(wsi_location)] #677
    
#     #Load TAN_TMA mutation label data
#     label_df1 = pd.read_excel(os.path.join(label_path, "TAN97_core_mappings.xlsx")) #These Ids not in label_df2: ['18-018', '18-087', '18-064', '18-077', '08-016', '06-131']
#     label_df1.rename(columns = {'AR': 'AR_inMappingFile'}, inplace = True)
#     label_df1.loc[pd.isna(label_df1['AR pos']),'AR pos'] = 0
#     label_df1.loc[pd.isna(label_df1['NE pos']),'NE pos'] = 0
#     label_df2 = pd.read_excel(os.path.join(label_path, "TAN_coded mutation_for Roman.xlsx"))
#     label_df2.rename(columns = {'AR coded': 'AR',
#                                'CHD1 coded': 'CHD1',
#                                'PTEN coded': 'PTEN',
#                                'RB1 coded': 'RB1',
#                                'TP53 coded': 'TP53', 
#                                'BRCA2 coded':'BRCA2'}, inplace = True)
#     #Combine to get core ids
#     #Only keep the ids in TAN_coded mutation_for Roman.xlsx, because no mutation labels are aviabale , cannot say it is negative
#     label_df = label_df1.merge(label_df2, left_on = ['ptid'], right_on = ['Sample'], how = 'right')
#     label_df.reset_index(drop=True, inplace=True)
        
#     # #There 40 sample IDs does not have matched AR status
#     # checkAR = label_df.loc[label_df['AR pos'] != label_df['AR'],]
#     # print(len(set(checkAR['Sample'])))
#     # checkAR.to_csv(out_location + "AR_notmatch.csv", index = False)
    
    
#     #Recode SITE info
#     label_df['SITE_LOCAL'] = pd.NA
#     cond = label_df['ORGAN SITE'] == 'PROSTATE'
#     label_df.loc[cond,'SITE_LOCAL'] = 1
#     label_df.loc[~cond,'SITE_LOCAL'] = 0
    
#     #Rename sample id column
#     label_df.rename(columns = {'TMA-row-col': 'SAMPLE_ID'}, inplace= True)
    
#     #Only select ID that is in label file
#     selected_ids = [x for x in tan_ids if x in list(label_df['SAMPLE_ID'].unique())] #596
    
#     #Exclude IDs has no cancer detected
#     cd_aval_ids = [x for x in os.listdir(info_path) if x != '.DS_Store'] #677
#     cancer_detect_list = []
#     for cur_id in cd_aval_ids:
#         cur_info_df = pd.read_csv(os.path.join(info_path, cur_id, 'ft_model',cur_id + "_TILE_TUMOR_PERC.csv"))
#         cancer_detect_list.append(cur_info_df)
#     all_cd_df = pd.concat(cancer_detect_list) #146888
    
    
#     #Filter for Cancer detected tiles > threshod
#     all_cd_df = all_cd_df.loc[all_cd_df['TUMOR_PIXEL_PERC'] >= args.TUMOR_FRAC_THRES] #19098

#     #No Cancer IDs 
#     cancer_ids = list(set(all_cd_df['SAMPLE_ID']))
#     nocancer_ids = [x for x in cd_aval_ids if x not in cancer_ids] #299
#     print("No Cancer detected (n = ", len(nocancer_ids), ")") 
#     selected_ids = [x for x in selected_ids if x not in  nocancer_ids] 
#     selected_ids.sort()
#     print("Final TCGA IDs (n = ", len(selected_ids), ")") #355
    

