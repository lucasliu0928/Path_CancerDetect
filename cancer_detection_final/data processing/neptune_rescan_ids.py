#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:46:04 2025

@author: jliu6
"""

import sys
import os
import argparse
import pandas as pd
import warnings
import glob
import pandas as pd


#DIR
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
wsi_location_nep = proj_dir + 'data/Neptune/'
wsi_location_nep2 = wsi_location_nep + 'rescanned/'

label_location = proj_dir + 'data/MutationCalls/Neptune/'

################################################################################
#Load slide info, get rescanned ids in slide info file
################################################################################
scan_info1 = pd.read_excel(label_location + "Neptune Trial - Slide Return.xlsx")
scan_info1.dropna(subset=['Slide name'], inplace = True)
cond = scan_info1['Slide name'].str.contains('tif', na=False)
scan_info1.loc[~cond,'Slide name'] = scan_info1.loc[~cond,'Slide name'] + '.tif'
scan_info1.rename(columns = {'Notes': 'old Notes','Slide name': 'old Slide name'}, inplace = True)

scan_info2 = pd.read_excel(label_location + "Neptune Trial - Slide Return 1 _updated.xlsx")
scan_info2.dropna(subset=['Slide name'], inplace = True)
cond = scan_info2['Slide name'].str.contains('tif', na=False)
scan_info2.loc[~cond,'Slide name'] = scan_info2.loc[~cond,'Slide name'] + '.tif'
scan_info2.rename(columns = {'Notes': 'new Notes','Slide name': 'new Slide name'}, inplace = True)

#Combined the two
scan_info_merged = scan_info1.merge(scan_info2, on = ["Patient_name","Date rec'd", "MRN"])
scan_info_merged['Neptune ID'] = scan_info_merged['Patient_name'].str.extract(r'(NEP),\s*(\d{3})')[0] + '-' + scan_info_merged['Patient_name'].str.extract(r'(NEP),\s*(\d{3})')[1]
len(scan_info_merged['Neptune ID'].unique()) #unique patient ID 316

#rescaned id
cond = (scan_info_merged['new Notes'] != 'RESCAN') & (scan_info_merged['old Notes'] == 'RESCAN') 
ids_rescaned_df = scan_info_merged.loc[cond]
ids_rescaned = list(ids_rescaned_df['new Slide name'])


################################################################################
#After store the rescanned slides in the location
################################################################################
#load rescanned slide names
slide_ids_rescaned = os.listdir(wsi_location_nep2)
slide_id_rescan_df = pd.DataFrame({'slide_id_rescaned': slide_ids_rescaned})
slide_id_rescan_df[['Neptune ID', 'Block']] = slide_id_rescan_df['slide_id_rescaned'].str.extract(
    r'((?:NEP)-\d{3})(?:PS(\d(?:-\d)?))?'
)
slide_id_rescan_df['Block'] = slide_id_rescan_df['Block'].apply(lambda x: f'PS{x}' if pd.notna(x) else pd.NA)


#Coomfirm store the right lisdes
check1 = list(slide_id_rescan_df['slide_id_rescaned'])
check1.sort()
check2 = ids_rescaned
check2.sort()
check1 == check2



################################################################################
#load old slide names
################################################################################
slide_ids_old = [x for x in os.listdir(wsi_location_nep) if x not in ['rescanned', '.DS_Store']]
slide_id_old_df = pd.DataFrame({'slide_id_old': slide_ids_old})
slide_id_old_df[['Neptune ID', 'Block']] = slide_id_old_df['slide_id_old'].str.extract(
    r'((?:NEP)-\d{3})(?:PS(\d(?:-\d)?))?'
)
slide_id_old_df['Block'] = slide_id_old_df['Block'].apply(lambda x: f'PS{x}' if pd.notna(x) else pd.NA)

################################################################################
#Merge old and rescanned slides
################################################################################
slide_ids_df = slide_id_old_df.merge(slide_id_rescan_df, on = ['Neptune ID' , 'Block'], how = 'outer')
#manul correction
cur_id = 'NEP-054'
cond1 = slide_ids_df['slide_id_old'] == 'NEP-054PS1_HE_MH_03252024.tif'
cond2 = slide_ids_df['Neptune ID'] == cur_id
slide_ids_df.loc[cond1 & cond2, 'Block'] = 'PS1 or PS1-1'
slide_ids_df.loc[cond1 & cond2, 'slide_id_rescaned'] = 'NEP-054PS1-1_HE_MH_04142025.tif'
#drop
slide_ids_df = slide_ids_df.loc[~pd.isna(slide_ids_df['slide_id_old'])]

################################################################################
#Load mutation info
################################################################################
label_df = pd.read_excel(label_location + "Neptune_ES_ver2.xlsx")
slide_ids_label = label_df['Slide ID'] #209
len(label_df['Neptune ID'].unique()) #unique patient ID #206


###################################################################################################
#All rescanned ids not in label df
###################################################################################################
len([x for x in slide_ids_label if x in slide_ids_old]) #all rescanned ids in orignal scan folder
len([x for x in slide_ids_label if x in slide_ids_rescaned]) #All rescanned ids not in label df


###################################################################################################
#Mark the ones that rescaned for checking
###################################################################################################
slide_ids_df.rename(columns = {'slide_id_old': 'Slide ID'}, inplace = True)
mark_df = label_df.merge(slide_ids_df, on = ['Neptune ID','Block','Slide ID'], how = 'left')
mark_df.to_excel(label_location + 'Neptune_ES_ver2_check_rescanned.xlsx', index= False)
