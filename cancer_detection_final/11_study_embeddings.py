#!/usr/bin/env python
# coding: utf-8

#NOTE: use python env acmil in ACMIL folder
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import warnings
import torch
sys.path.insert(0, '../Utils/')
from Utils import create_dir_if_not_exists
from train_utils import get_feature_idexes, get_selected_labels, get_train_test_val_data, update_label, load_model_ready_data
from train_utils import update_to_agg_feature, concate_agg_feature
warnings.filterwarnings("ignore")


#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse


############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='HR_MSI', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_overlap', default=100, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--train_cohort', default= 'OPX', type=str, help='TCGA_PRAD or OPX or Neptune or TCGA_OPX')
parser.add_argument('--external_cohort1', default= 'TCGA_PRAD', type=str, help='TCGA_PRAD or OPX or Neptune')
parser.add_argument('--external_cohort2', default= 'Neptune', type=str, help='TCGA_PRAD or OPX or Neptune')
parser.add_argument('--hr_type', default= "HR2", type=str, help='HR version 1 or 2 (2 only include 3 genes)')
parser.add_argument('--out_folder', default= 'pred_out_050625', type=str, help='out folder name')


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    ####################################
    ######      USERINPUT       ########
    ####################################
    #Feature
    SELECTED_FEATURE = get_feature_idexes(args.fe_method, include_tumor_fraction = False)
    N_FEATURE = len(SELECTED_FEATURE)
    
    #Select mutatation
    #mutations = [ "MSI_POS","HR1", "HR2", "AR", "PTEN","RB1","TP53","TMB"]
    args.mutation = "MT"
    
    #Label
    SELECTED_LABEL, selected_label_index = get_selected_labels(args.mutation, args.hr_type, args.train_cohort)
    print(SELECTED_LABEL)
    print(selected_label_index)
    

    ########################################################################
    ######              DIR                                           ######
    ########################################################################
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'        
    train_val_test_id_path =  os.path.join(proj_dir + 'intermediate_data/3B_Train_TEST_IDS', 
                                           args.train_cohort ,
                                           'TFT' + str(args.tumor_frac))

    ##################
    #Select GPU
    ##################
    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    ################################################
    #     Model ready data 
    ################################################
    data_path = proj_dir + 'intermediate_data/5_model_ready_data'
    data_ol100 = load_model_ready_data(data_path, args.train_cohort, args.train_overlap, args.fe_method, args.tumor_frac)
    data_ol0   = load_model_ready_data(data_path, args.train_cohort, args.test_overlap, args.fe_method, args.tumor_frac)
    data_external1 =  load_model_ready_data(data_path, args.external_cohort1, args.test_overlap, args.fe_method, args.tumor_frac)
    data_external2 =  load_model_ready_data(data_path, args.external_cohort2, args.test_overlap, args.fe_method, args.tumor_frac)

    #Clean (updated label and remove reduant info)
    data_ol100, _ = update_label(data_ol100, selected_label_index)
    data_ol0, _   = update_label(data_ol0, selected_label_index)
    external_data1, external_ids1 = update_label(data_external1, selected_label_index)
    external_data2, external_ids2 = update_label(data_external2, selected_label_index)

    #Aggraete each slide feature
    data_opx_ol100 = update_to_agg_feature(data_ol100)
    data_opx_ol0 =   update_to_agg_feature(data_ol0)
    data_tcga_ol0 =  update_to_agg_feature(data_external1)
    data_nep_ol0 =   update_to_agg_feature(data_external2)
    
    
    #["AR", "HR1", "HR2", "PTEN","RB1","TP53","TMB","MSI_POS"] 
    indata = external_data1
    label_idx = 1
    indata_final = [x for x in indata if x[1][0,label_idx] == 1]
    len(indata_final)

    ################################################
    #PCA and TSNE
    ################################################
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    #%matplotlib inline
    
    
    data_opx_ol100, opx_ol100_label = concate_agg_feature(data_opx_ol100, "OPX_OL100")
    data_opx_ol0, opx_ol0_label = concate_agg_feature(data_opx_ol0, "OPX_OL0")
    data_tcga_ol0, tcga_ol0_label = concate_agg_feature(data_tcga_ol0, "TCGA_OL0")
    data_nep_ol0, nep_ol0_label = concate_agg_feature(data_nep_ol0, "NEP_OL0")
    
    all_X = np.concatenate((data_opx_ol100, data_opx_ol0, data_tcga_ol0, data_nep_ol0))
    all_y = np.concatenate((opx_ol100_label, opx_ol0_label, tcga_ol0_label, nep_ol0_label))


    # Reduce to 50 components first
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(all_X)
        
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    # print(np.isnan(X_pca).any())
    # print(np.isinf(X_pca).any())  # Should be False
    # print(X_pca.dtype)
            
    
    # Step 1: Define color map
    label_to_color = {
        'NEP_OL0': 'red',
        'OPX_OL0': 'blue',
        'OPX_OL100': 'green',
        'TCGA_OL0': 'purple'
    }
    
    # Step 2: Map labels to colors
    colors = [label_to_color[label] for label in all_y]

    # Step 3: Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors)
    
    # Step 4: Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=clr, label=lbl) for lbl, clr in label_to_color.items()]
    plt.legend(handles=legend_elements, title="Classes")
    
    # Final touches
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.savefig("plot.png")
    plt.show()