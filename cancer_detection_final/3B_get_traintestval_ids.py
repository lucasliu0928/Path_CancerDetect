#!/usr/bin/env python
# coding: utf-8
#NOTE: use paimg9 env
import sys
import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import warnings
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists
warnings.filterwarnings("ignore")
import argparse

def count_mutation_perTrainTest(train_test_df, selected_label):

    # Compute N (count) per TRAIN/TEST group
    train_df = train_test_df[train_test_df['TRAIN_OR_TEST'] == 'TRAIN']
    test_df  = train_test_df[train_test_df['TRAIN_OR_TEST'] == 'TEST']
    
    n_train = len(train_df)
    n_test = len(test_df)
    
    # Initialize result storage
    results = []
    
    for col in selected_label:
        train_n = train_df[col].sum()
        test_n = test_df[col].sum()
        
        train_pct = (train_n / n_train) * 100 if n_train > 0 else 0
        test_pct = (test_n / n_test) * 100 if n_test > 0 else 0
        
        # Format as "N (%)"
        train_formatted = f"{train_n} ({train_pct:.1f}%)"
        test_formatted = f"{test_n} ({test_pct:.1f}%)"
        
        results.append({
            'Outcome': col,
            'Train N (%)': train_formatted,
            'Test N (%)': test_formatted
        })
    # Convert to DataFrame for nice display
    results_df = pd.DataFrame(results)

    
    return results_df


def get_pos_neg_ids(tile_info_pt, label_name):
    #Postive IDs
    pos_ids = list(tile_info_pt.loc[tile_info_pt[label_name] == 1 , 'PATIENT_ID'].unique()) #24
    
    #Neg Ids
    neg_ids = list(tile_info_pt.loc[tile_info_pt[label_name] == 0 , 'PATIENT_ID'].unique()) #242
    
    return pos_ids, neg_ids

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--save_image_size', default=250, type=int, help='the size of extracted tiles')
parser.add_argument('--pixel_overlap', default=100, type=int, help='specify the level of pixel overlap in your saved tiles')
parser.add_argument('--feature_extraction_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--cohort_name', default='OPX', type=str, help='data set name: OPX or TCGA_PRAD or Neptune')
parser.add_argument('--tile_info_path', default= '3A_otherinfo', type=str, help='tile info folder name')
parser.add_argument('--out_folder', default= '3B_Train_TEST_IDS', type=str, help='out folder name')

args = parser.parse_args()

############################################################################################################
#USER INPUT 
############################################################################################################
SELECTED_LABEL = ["AR","HR1","HR2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]


##################
###### DIR  ######
##################
proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
folder_name = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.pixel_overlap)
info_dir = os.path.join(proj_dir,'intermediate_data',args.tile_info_path, args.cohort_name, folder_name)
outdir =  os.path.join(proj_dir + 'intermediate_data', args.out_folder, args.cohort_name, "TFT" + str(args.TUMOR_FRAC_THRES))
create_dir_if_not_exists(outdir)


##################
#Select GPU
##################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


################################################
#    Load labels from tile info
################################################
tile_info_df = pd.read_csv(os.path.join(info_dir,"all_tile_info.csv"))
tile_info_df = tile_info_df[tile_info_df['TUMOR_PIXEL_PERC']>=args.TUMOR_FRAC_THRES]
tile_info_df_pt = tile_info_df.drop_duplicates(subset = ['PATIENT_ID']).copy() #patient-level


################################################
#balance by  mutation < 10%
################################################
if args.cohort_name == 'OPX':
    mutations_to_balance = ['AR','HR2' , 'RB1' ,'MSI_POS','TMB_HIGHorINTERMEDITATE']
elif args.cohort_name == 'Neptune':
    mutations_to_balance = ['AR', 'RB1' ,'MSI_POS','TMB_HIGHorINTERMEDITATE']
else:
    mutations_to_balance = ['AR', 'HR2' ,'MSI_POS']

from sklearn.utils import shuffle

df = tile_info_df_pt.copy()
df = shuffle(df, random_state=42)

# Step 1: Initialize
used_patient_ids = set()
train_parts = []
test_parts = []

# Step 2: Balance specified mutations (equal split)
for mutation in mutations_to_balance:
    pos_df = df[(df[mutation] == 1) & (~df['PATIENT_ID'].isin(used_patient_ids))]
    pos_df = pos_df.drop_duplicates(subset='PATIENT_ID')
    pos_df = shuffle(pos_df, random_state=42)
    
    split_idx = len(pos_df) // 2
    train_mut = pos_df.iloc[:split_idx]
    test_mut = pos_df.iloc[split_idx:]
    
    train_parts.append(train_mut)
    test_parts.append(test_mut)
    used_patient_ids.update(pos_df['PATIENT_ID'])

# Step 3: Handle remaining patients
remaining_df = df[~df['PATIENT_ID'].isin(used_patient_ids)]
remaining_df = remaining_df.drop_duplicates(subset='PATIENT_ID')
remaining_df = shuffle(remaining_df, random_state=42)

# Step 4: Calculate target sizes
total_patient_ids = pd.concat([pd.DataFrame(list(used_patient_ids), columns=['PATIENT_ID']), remaining_df])['PATIENT_ID'].nunique()
target_train = int(total_patient_ids * 0.75)
target_test = total_patient_ids - target_train

current_train = pd.concat(train_parts)['PATIENT_ID'].nunique()
current_test = pd.concat(test_parts)['PATIENT_ID'].nunique()

remaining_needed_train = max(0, target_train - current_train)
remaining_needed_test = max(0, target_test - current_test)

# Step 5: Fill up remaining train/test to meet 75/25 split
remaining_train = remaining_df.iloc[:remaining_needed_train]
remaining_test = remaining_df.iloc[remaining_needed_train:remaining_needed_train + remaining_needed_test]

train_parts.append(remaining_train)
test_parts.append(remaining_test)

# Finalize splits
train_df = pd.concat(train_parts).drop_duplicates(subset='PATIENT_ID')
test_df = pd.concat(test_parts).drop_duplicates(subset='PATIENT_ID')

train_df = shuffle(train_df, random_state=42).reset_index(drop=True)
test_df = shuffle(test_df, random_state=42).reset_index(drop=True)

# Summary Output
print(f"Train patients: {train_df['PATIENT_ID'].nunique()}")
print(f"Test patients: {test_df['PATIENT_ID'].nunique()}")

print("\nFinal mutation counts:")
all_mutations = ['AR', 'HR1', 'HR2', 'PTEN', 'RB1', 'TP53', 'TMB_HIGHorINTERMEDITATE', 'MSI_POS']
for mutation in all_mutations:
    print(f"{mutation} - Train: {train_df[mutation].sum()}, Test: {test_df[mutation].sum()}")
    

################################################
#Get all train and test
################################################
train_ids  = list(train_df['PATIENT_ID'].unique())
test_ids   = list(test_df['PATIENT_ID'].unique())


# #For train_ids_full, then k-fold validation
# n_splits = 5 
# kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) #Initialize KFold
# fold_ids = {}
# for fold, (train_index, val_index) in enumerate(kf.split(train_ids)):
#     train_ids_fold = [train_ids[i] for i in train_index]  # Get train IDs
#     val_ids = [train_ids[i] for i in val_index]  # Get val IDs
#     fold_ids[fold] = {'Train': train_ids_fold, 'Val' :val_ids}  # Store as lists

#stratified by y    , make sure we have pos in validation
n_splits = 5 
y = tile_info_df_pt.loc[tile_info_df_pt['PATIENT_ID'].isin(train_ids), SELECTED_LABEL]
if args.cohort_name == 'TCGA_PRAD':
    y.drop(columns=["TMB_HIGHorINTERMEDITATE"], inplace = True)
mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_ids = {}
for fold, (train_index, val_index) in enumerate(mskf.split(np.zeros(len(y)), y)):
    train_ids_fold = [train_ids[i] for i in train_index]  # Get train IDs
    val_ids = [train_ids[i] for i in val_index]  # Get val IDs
    fold_ids[fold] = {'Train': train_ids_fold, 'Val' :val_ids}  # Store as lists



################################################
#Train and Test and VAl df
################################################
train_test_valid_df = tile_info_df_pt[['SAMPLE_ID', 'PATIENT_ID'] + SELECTED_LABEL].copy()
train_test_valid_df['TRAIN_OR_TEST'] = pd.NA
cond = train_test_valid_df['PATIENT_ID'].isin(train_ids)
train_test_valid_df.loc[cond, 'TRAIN_OR_TEST'] = 'TRAIN'
train_test_valid_df.loc[~cond, 'TRAIN_OR_TEST'] = 'TEST'

for k in range(n_splits):
    #Update dataframe
    cond1 = train_test_valid_df['PATIENT_ID'].isin(fold_ids[k]['Train'])
    train_test_valid_df.loc[cond1, 'FOLD' + str(k)] = 'TRAIN'
    cond2 = train_test_valid_df['PATIENT_ID'].isin(fold_ids[k]['Val'])
    train_test_valid_df.loc[cond2, 'FOLD' + str(k)] = 'VALID'
    cond3 = ~(cond1 | cond2)
    train_test_valid_df.loc[cond3, 'FOLD' + str(k)] = 'TEST'

train_test_valid_df.to_csv(os.path.join(outdir, 'train_test_split.csv'))



################################################
#Count mutation Sample level by Train and Test
################################################
count_pt = count_mutation_perTrainTest(train_test_valid_df,SELECTED_LABEL)
count_pt.to_csv(os.path.join(outdir, 'label_count_patient_level.csv'))