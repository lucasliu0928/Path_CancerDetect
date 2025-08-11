#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 12:55:08 2025

@author: jliu6
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the root directory containing all the GAMMA_* folders
proj_dir = "/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/"



all_tile_dir_opx = proj_dir + '2_cancer_detection/OPX/IMSIZE250_OL0/'
all_tile_dir_tcga = proj_dir + '2_cancer_detection/TCGA_PRAD/IMSIZE250_OL0/'
all_tile_dir_nep = proj_dir + '2_cancer_detection/Neptune/IMSIZE250_OL0/'




# Walk through all subdirectories
root_dir = all_tile_dir_nep
cancer_dfs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith('.csv'):
            file_path = os.path.join(dirpath, file)
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = file_path  # Optional: track file source
                cancer_dfs.append(df)
            except Exception as e:
                print(f"Could not read {file_path}: {e}")

# Combine all DataFrames into one
all_cancer_df = pd.concat(cancer_dfs, ignore_index=True)

#Compute cancer tile percentage for each nep slide

total_counts = all_cancer_df.groupby('SAMPLE_ID').size().reset_index(name='total_tiles_count') # Group total counts per SAMPLE_ID
high_tumor_counts = all_cancer_df[all_cancer_df['TUMOR_PIXEL_PERC'] >= 0.9].groupby('SAMPLE_ID').size().reset_index(name='high_tumor_tile_count')
merged = pd.merge(total_counts, high_tumor_counts, on='SAMPLE_ID', how='left')
merged['high_tumor_tile_count'] = merged['high_tumor_tile_count'].fillna(0)
merged['tumor_tile_ratio'] = (merged['high_tumor_tile_count'] / merged['total_tiles_count'])
merged['tumor_tile_ratio'] = round(merged['tumor_tile_ratio'],10)

#Get pred df for each slide
root_dir = proj_dir  + "pred_out_072925v2/trainCohort_union_STNandNSTN_OPX_TCGA_Samples1000_GRLFalse/acmil/uni2/"
root_dir = root_dir + 'TrainOL100_TestOL0_TFT0.9/FOLD4/MTHR_TYPEHR2/predictions/'
pred_dir = root_dir  + 'GAMMA_0_ALPHA_-1/'

selected_out = 'HR2'
pred_df = pd.read_csv(pred_dir + "n_token3_NEP_pred_df.csv")
pred_df = pred_df.loc[pred_df['OUTCOME'] == selected_out]


root_dir = "/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/pred_out_072925v2/"
root_dir = root_dir + "trainCohort_union_STNandNSTN_OPX_TCGA_Samples1000_GRLFalse/acmil/uni2/TrainOL100_TestOL0_TFT0.9/"
root_dir = root_dir + selected_out + '/perf/GAMMA_0_ALPHA_-1/'

# Rename SAMPLE_IDs in pred_df to match merged_df's column name
pred_df = pred_df.rename(columns={"SAMPLE_IDs": "SAMPLE_ID"})

# Merge on SAMPLE_ID
final_df = pd.merge(pred_df, merged, on="SAMPLE_ID", how="inner")


# Step 1: Create false positive and false negative flags
final_df["false_positive"] = ((final_df["Y_True"] == 0) & (final_df["Pred_Class"] == 1)).astype(int)
final_df["false_negative"] = ((final_df["Y_True"] == 1) & (final_df["Pred_Class"] == 0)).astype(int)
final_df['error'] = (final_df['Y_True'] != final_df['Pred_Class']).astype(int)


# Step 1: Create bins for tumor_tile_ratio
max_val = final_df["tumor_tile_ratio"].max()
bin_edges = np.arange(0, max_val + 0.1, 0.1)
final_df["percent_bin"] = pd.cut(final_df["tumor_tile_ratio"], bins=bin_edges, include_lowest=True)
final_df["percent_bin_str"] = final_df["percent_bin"].astype(str).str.replace(r"\(-?0\.0{0,4}1?,", "[0.0,", regex=True)
final_df.to_csv(pred_dir + "tumor_perc_info.csv")

####PLOT
# Step 2: Compute error rate in each bin
error_rate_by_bin = final_df.groupby("percent_bin")["error"].mean()

# Step 3: Compute rates by bin
error_rate_by_bin = final_df.groupby("percent_bin")["error"].mean()
fp_rate_by_bin = final_df.groupby("percent_bin")["false_positive"].mean()
fn_rate_by_bin = final_df.groupby("percent_bin")["false_negative"].mean()

# Combine into one DataFrame
rate_df = pd.DataFrame({
    "Error Rate": error_rate_by_bin,
    "False Positive Rate": fp_rate_by_bin,
    "False Negative Rate": fn_rate_by_bin
})

# Step 4: Plot side-by-side bars
ax = rate_df.plot(kind='bar', figsize=(12, 6), edgecolor='black', width=0.8)
plt.title( selected_out + ' Overall Error Types by Tumor Tile Ratio')
plt.xlabel('Tumor Tile Ratio (Grouped)')
plt.ylabel('Rate')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.show()

#PLOT2
# Step 4: Group and compute proper rates
grouped = final_df.groupby("percent_bin")

error_rate_by_bin = grouped["error"].sum() / grouped["error"].count()
fp_rate_by_bin = grouped["false_positive"].sum() / grouped["false_positive"].count()
fn_rate_by_bin = grouped["false_negative"].sum() / grouped["false_negative"].count()

# Step 5: Count positives and negatives
pos_counts = grouped["Y_True"].sum()
neg_counts = grouped["Y_True"].apply(lambda x: (x == 0).sum())

# Step 6: Assemble results
rate_df = pd.DataFrame({
    "Error Rate": error_rate_by_bin,
    "False Positive Rate": fp_rate_by_bin,
    "False Negative Rate": fn_rate_by_bin,
    "# Positives": pos_counts,
    "# Negatives": neg_counts
})

# Step 7: Plot side-by-side bars
ax = rate_df[["Error Rate", "False Positive Rate", "False Negative Rate"]].plot(
    kind='bar', figsize=(12, 6), edgecolor='black', width=0.8
)

# Step 8: Annotate each bar group with counts
for idx, (pos, neg) in enumerate(zip(rate_df["# Positives"], rate_df["# Negatives"])):
    ax.text(idx, 1.02, f'Pos: {int(pos)}\nNeg: {int(neg)}',
            ha='center', va='bottom', fontsize=9)

plt.title('Error Types by Tumor Tile Ratio Bins')
plt.xlabel('Tumor Tile Ratio (Binned)')
plt.ylabel('Rate')
plt.xticks(rotation=45)
plt.ylim(0, 1.15)
plt.grid(axis='y')

# Move legend outside the plot (right side)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legend
plt.show()