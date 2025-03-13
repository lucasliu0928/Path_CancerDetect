#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:32:29 2024

@author: jliu6
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,fbeta_score,average_precision_score
from sklearn import metrics
import torchvision
import numpy as np
import os
from Utils import minmax_normalize

def compute_performance(y_true,y_pred_prob,y_pred_class, cohort_name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel() #CM

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob, pos_label=1)
    
    # Average precision score = PR-AUC
    PR_AUC = average_precision_score(y_true, y_pred_prob)

    AUC = round(metrics.auc(fpr, tpr),2)
    ACC = round(accuracy_score(y_true, y_pred_class),2)
    F1 = round(f1_score(y_true, y_pred_class),2)
    F2 = round(fbeta_score(y_true, y_pred_class,beta = 2),2)
    F3 = round(fbeta_score(y_true, y_pred_class,beta = 3),2)
    Recall = round(recall_score(y_true, y_pred_class),2)
    Precision = round(precision_score(y_true, y_pred_class),2)
    Specificity = round(tn / (tn + fp),2)
    perf_tb = pd.DataFrame({"AUC": AUC,
                            "Recall": Recall,
                            "Specificity":Specificity,
                            "ACC": ACC,
                            "Precision":Precision,
                            "PR_AUC":PR_AUC,
                            "F1": F1,
                            "F2": F2,
                            "F3": F3},index = [cohort_name])
    
    return perf_tb



def plot_LOSS (train_loss, valid_loss, outdir):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(valid_loss,label="Validation")
    plt.plot(train_loss,label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(outdir + 'LOSS.png')
    
    
def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




def compute_performance_each_label(selected_label, prediction_df, prediction_type):
    
    perf_list = []
    for mut in selected_label:
        cur_pred_df = prediction_df.loc[prediction_df['OUTCOME'] == mut]
        cur_perf_df = compute_performance(cur_pred_df['Y_True'],
                                          cur_pred_df['Pred_Prob'],
                                          cur_pred_df['Pred_Class'],
                                          prediction_type)
        cur_perf_df['OUTCOME'] = mut
        perf_list.append(cur_perf_df)
    
    comb_perf = pd.concat(perf_list)

    return comb_perf


def get_attention_and_tileinfo(pt_label_df, patient_att_score):    
    #Get label
    pt_label_df.reset_index(drop = True, inplace = True)

    #Get attention
    cur_att  = pd.DataFrame({'ATT':list(minmax_normalize(patient_att_score))})
    cur_att.reset_index(drop = True, inplace = True)

    #Comb
    cur_att_df = pd.concat([pt_label_df,cur_att], axis = 1)
    cur_att_df.reset_index(drop = True, inplace = True)

    return cur_att_df



def get_performance(y_predprob, y_true, cohort_ids, outcome, THRES):

    #Prediction df
    pred_df = pd.DataFrame({"SAMPLE_IDs":  cohort_ids, 
                            "Y_True": y_true, 
                            "Pred_Prob" :  y_predprob,
                            "OUTCOME": outcome})
        
    pred_df['Pred_Class'] = 0
    pred_df.loc[pred_df['Pred_Prob'] > THRES,'Pred_Class'] = 1


    perf_df = compute_performance_each_label([outcome], pred_df, "SAMPLE_LEVEL")

    return pred_df, perf_df
