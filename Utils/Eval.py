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
                            "ACC": ACC,
                            "F1": F1,
                            "F2": F2,
                            "F3": F3,
                            "Recall": Recall,
                            "Precision":Precision,
                            "Specificity":Specificity,
                            "PR_AUC":PR_AUC},index = [cohort_name])
    
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
