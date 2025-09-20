#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 16:32:16 2025

@author: jliu6
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import numpy as np
print(sys.executable)
%matplotlib inline




####################################################################################
#Load pred df
proj_dir = "/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/pred_out_072725/"
df = pd.read_csv(proj_dir + "trainCohort_union_STNandNSTN_OPX_TCGA_Samples1000_GRLFalse/acmil/uni2/TrainOL100_TestOL0_TFT0.9/FOLD0/MSI/predictions/GAMMA_0_ALPHA_-1/n_token3_TEST_COMB_pred_df.csv")
df_hr = df.loc[df['OUTCOME'] == 'MSI']


plot_df = df_hr
boxplot = plot_df.boxplot(column = ['Pred_Prob'], by='Y_True')
# To ensure the plot displays in Spyder
plt.title('Boxplot of Pred_Prob by Y_True')
plt.suptitle('')  # Removes the default 'Boxplot grouped by Y_True' title
plt.xlabel('Y_True')
plt.ylabel('Pred_Prob')
plt.show()





#Logit adjst
# Define class priors
n_pos = plot_df[plot_df['Y_True'] == 1].shape[0]
n_neg = plot_df[plot_df['Y_True'] == 0].shape[0]

p_pos = n_pos / (n_pos + n_neg)
p_neg = 1 - p_pos

# Compute logit adjustment term
tau = 1.0  # You can tune this
adjustment = tau * torch.log(torch.tensor(p_pos / p_neg))

# During training or inference
logits = torch.tensor(np.array(df_hr['Pred_logit']))
logits_adjusted = logits + (-adjustment)


plot_df['adjusted_logit'] = logits_adjusted
plot_df['adjusted_predprob'] = torch.sigmoid(logits_adjusted)

# Use in BCEWithLogitsLoss
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits_adjusted, y)


