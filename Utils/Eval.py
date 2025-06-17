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
from misc_utils import minmax_normalize
import torch
from torchmetrics import ROC
import os

from sklearn.isotonic import IsotonicRegression


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






def plot_roc_curve(y_pred, y_true, outdir, outcome_name):
    # Initialize ROC metric for binary classification
    roc = ROC(task='binary')
    
    # Compute FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc(torch.tensor(y_pred), torch.tensor(y_true))
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(outdir, "rocauc_"+ outcome_name + ".png"), dpi=300)
    plt.show()




def calibrate_probs_isotonic(y_val, y_prob_val, y_test_prob):
    """
    Perform Isotonic Regression to calibrate predicted probabilities.

    Args:
        y_val: Ground truth labels from the validation set.
        y_prob_val: Model-predicted probabilities from the validation set.
        y_test_prob: Model-predicted probabilities for the test set.

    Returns:
        Calibrated probabilities for the test set.
    """
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(y_prob_val, y_val)
    calibrated_probs = iso_reg.predict(y_test_prob)
    return calibrated_probs


def bootstrap_ci_from_df(df, y_true_col='y_true', y_pred_col=None, y_prob_col=None,
                         num_bootstrap=1000, ci=95, seed=None):
    """
    Compute bootstrap confidence interval for a metric using predictions in a DataFrame.

    Args:
        df: DataFrame with prediction results.
        metric_fn: Metric function. Can accept (y_true, y_pred) or (y_true, y_prob).
        y_true_col: Column name for ground truth.
        y_pred_col: Column name for predicted labels (used for accuracy, etc.).
        y_prob_col: Column name for predicted probabilities (used for AUROC, etc.).
        num_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g., 95 for 95% CI).
        seed: Optional random seed.

    Returns:
        (lower_bound, upper_bound) of CI.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    perf_list = []

    for _ in range(num_bootstrap):
        sample = df.sample(n=n, replace=True, random_state=rng.integers(0, 1e6))
        y_true = sample[y_true_col].values
        y_pred_class =  sample[y_pred_col].values
        y_pred_prob =   sample[y_prob_col].values
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel() #CM
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob, pos_label=1)
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
                                "F3": F3},index = [0])
        perf_list.append(perf_tb)
    perf_df = pd.concat(perf_list)

    mean_values = perf_df.mean()
    lower_bounds = perf_df.quantile((100 - ci) / 200)
    upper_bounds = perf_df.quantile(1 - (100 - ci) / 200)
    
    formatted_results = {
        column: f"{mean_values[column]:.2f} [{lower_bounds[column]:.2f} - {upper_bounds[column]:.2f}]"
        for column in perf_df.columns
    }
    ci_df = pd.DataFrame.from_dict(formatted_results, orient="index", columns=["Mean [CI Low, CI High]"])

    return ci_df.T


def boxplot_predprob_by_mutationclass(pred_df, outdir):
    # Get unique outcomes
    outcomes = pred_df['OUTCOME'].unique()
    n_outcomes = len(outcomes)
    
    # Create data structure for plotting
    data_to_plot = []
    labels = []
    positions = []
    
    for i, outcome in enumerate(outcomes):
        for y_val in [0, 1]:
            group = pred_df[(pred_df['OUTCOME'] == outcome) & (pred_df['Y_True'] == y_val)]
            data_to_plot.append(group['Pred_Prob'].values)
            labels.append(f"{outcome}\nY={y_val}")
            positions.append(i * 3 + y_val)  # spread boxes for each group
    
    # Plotting
    plt.figure(figsize=(8, 6))
    box = plt.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
    
    # Color boxes
    colors = ['lightblue', 'lightcoral'] * n_outcomes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xticks(positions, labels, rotation=0)
    plt.ylim(0, 1)
    plt.ylabel('Predicted Probability')
    plt.title('Predicted Probabilities by Outcome and Y_True')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pred_prob_boxplot_byoutcome.png"), dpi=300)
    plt.show()



