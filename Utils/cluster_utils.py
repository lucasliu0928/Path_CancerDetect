#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 03:27:29 2024

@author: jliu6
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from train_utils import get_sample_feature, get_sample_label, combine_feature_and_label


# Get clustering data
def get_cluster_data(indata):
    #Get dataframe to hold feature, label and tile info
    data_list = []
    for d in indata:
        feature_df = pd.DataFrame(d[0].numpy())
        feature_df.columns = [str(x) for x in feature_df.columns] #change col to str

        tf_df = pd.DataFrame(d[2].numpy())
        tf_df.columns = ['TUMOR_FRAC']
    
        info_df = d[3]
        
        label_df = pd.DataFrame(d[1].numpy())
        label_df.columns = ["AR","HR","PTEN","RB1","TP53","TMB","MSI_POS"] #change col name
        label_df = pd.concat([label_df] * len(info_df), ignore_index=True) #repeat n_tiles times to match size
        
        comb_df = pd.concat([feature_df, info_df, tf_df, label_df], axis = 1)
        data_list.append(comb_df)
    
    all_comb_df = pd.concat(data_list)
    
    return all_comb_df

def get_cluster_label(feature_df, cluster_centers, cluster_features):
    r'''
    Get Cluster label by compute dist between test/valid embedding to the center of kmeans
    '''
    distances = cdist(cluster_centers, feature_df[cluster_features].to_numpy(), 'euclidean')
    closest_indices = np.argmin(distances, axis=0)

    cluster_labels  = closest_indices

    return cluster_labels
    
def get_updated_feature(input_df, selected_ids, selected_feature):
    feature_list = []
    ct = 0 
    for pt in selected_ids:
        if ct % 10 == 0 : print(ct)

        cur_df = input_df.loc[input_df['ID'] == pt]

        #Extract feature, label and tumor info
        cur_feature = cur_df[selected_feature].values

        feature_list.append(cur_feature)
        ct += 1

    return feature_list


def get_pca_components(feature_data, n_components = 2):
    pca = PCA(n_components = n_components)
    pcs = pca.fit_transform(feature_data)
    explained_variance = pca.explained_variance_
    print("Explained variance by each component:", explained_variance)

    return pcs

def find_elbow_point(wcss):
    wcss = np.array(wcss)
    # Compute second differences (second derivative approximation)
    second_derivative = np.diff(wcss, n=2)
    # The elbow is where the second derivative is largest
    elbow_point = np.argmax(second_derivative) + 2  # +2 because diff shifts the index
    return elbow_point

def run_elbow_method(feature_data, outdir, method = 'kmean', rs = 42):
    wcss = []
    for i in range(1, 11):
        if method == 'kmean':
            kmeans = KMeans(n_clusters=i, random_state=rs)
        kmeans.fit(feature_data)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig(os.path.join(outdir,'elbow_method.png'))
    #plt.show()
    
    return wcss

def run_clustering_on_training(train_data, selected_feature, outdir, method = 'kmean', rs = 42):
    #Get features data
    train_feature = train_data[selected_feature]

    #PCA
    pcs = get_pca_components(train_feature, n_components = 2)
    
    #Feature scaling
    scaler = StandardScaler()
    scaled_pcs = scaler.fit_transform(pcs)
    
    #Run elbow
    wcss = run_elbow_method(scaled_pcs, outdir, method = method, rs = rs)
    bestk = find_elbow_point(wcss)
    
    #Run with best k
    model = KMeans(n_clusters=bestk, random_state = rs)
    model.fit(scaled_pcs)
    
    return model, bestk

def get_cluster_labels(indata, selected_feature, model,  rs = 42):
    #Get features data
    feature = indata[selected_feature]

    #PCA
    pcs = get_pca_components(feature, n_components = 2)
    
    #Feature scaling
    scaler = StandardScaler()
    scaled_pcs = scaler.fit_transform(pcs)

    #Get cluster labels
    cluster_labels = model.predict(scaled_pcs)
    
    return cluster_labels       

def plot_cluster_distribution(info_df, selected_label, out_path, cohort_name):
    #Plot scatter cluster plot for each outcome,
    for plot_outcome in selected_label:
        plot_data = info_df[['PC1','PC2', 'Cluster'] + [plot_outcome]]
            
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate dots by Cluster but color by Outcome
        for cluster in plot_data['Cluster'].unique():
            subset = plot_data[plot_data['Cluster'] == cluster]
            ax.scatter(subset['PC1'], subset['PC2'], 
                       s=np.where(subset[plot_outcome] == 1, 20, 0.001), 
                       c=['steelblue' if outcome == 0 else 'darkred' for outcome in subset[plot_outcome]], 
                       alpha=0.6,
                       linewidth=1.5, label=f'Cluster {cluster}',
                       zorder=3 if (subset[plot_outcome] == 1).any() else 2)
        
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(plot_outcome)
        plt.grid(True)
        plt.savefig(out_path +  '/cluster_scatter_' + plot_outcome + '_' + cohort_name + '.png')
        plt.close()
        
    # Plot distribution of outcome by cluster (stacked bar plot)   
    for plot_outcome in selected_label:
        plot_data = info_df[['PC1','PC2', 'Cluster'] + [plot_outcome]]
     
        # Create a crosstab to count the occurrences of each outcome per cluster
        crosstab = pd.crosstab(plot_data['Cluster'], plot_data[plot_outcome])
        
        # Calculate the percentage of each outcome per cluster
        percentage_crosstab = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        
        # Plot the stacked bar chart
        percentage_crosstab.plot(kind='bar', stacked=True, color=['steelblue', 'darkred'])
        plt.xlabel('Cluster')
        plt.ylabel('Percentage')
        plt.title('Bar Chart of ' + plot_outcome + ' per Cluster')
        plt.legend(title=plot_outcome, loc='center left', bbox_to_anchor=(1.0, 0.5))
        
        plt.tight_layout()
        plt.savefig(out_path + "/outcome_distribution_" + plot_outcome + '_' + cohort_name + '.png')
        plt.close()
    
    # Plot combined distribution of outcome by cluster 
    for plot_outcome in selected_label:
        plot_data = info_df[['PC1','PC2', 'Cluster'] + [plot_outcome]]
    
        # Create a crosstab to count the occurrences of each outcome per cluster
        crosstab = pd.crosstab(plot_data[plot_outcome],plot_data['Cluster'])
    
        # Calculate the percentage of each outcome per cluster
        percentage_crosstab = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        
        # Plot the stacked bar chart
        percentage_crosstab.plot(kind='bar', stacked=False, color=['#440154','#3b528b','#5ec962','#fde725'])
        plt.xlabel(plot_outcome)
        plt.ylabel('Percentage')
        plt.title('Bar Chart of Clusters Per Outcome')
        plt.legend(title='Cluster', loc='center left', bbox_to_anchor=(1.0, 0.5))
    
        plt.tight_layout()
        plt.savefig(out_path  + '/cluster_distribution_' + plot_outcome + '_' + cohort_name + '.png')
        plt.close()




def load_alltile_tumor_info(input_path, patient_id, selected_labels,info_df):
    all_tumor_info_df = pd.read_csv(os.path.join(input_path, patient_id, "ft_model/", patient_id + "_TILE_TUMOR_PERC.csv"))
    selected_tumor_info_df = info_df.loc[info_df['SAMPLE_ID'] == patient_id]
    
    comb_df = all_tumor_info_df.merge(selected_tumor_info_df, on = ['SAMPLE_ID', 'MAG_EXTRACT', 'SAVE_IMAGE_SIZE', 'PIXEL_OVERLAP',
                                                          'LIMIT_BOUNDS', 'TILE_XY_INDEXES', 'TILE_COOR_ATLV0', 'WHITE_SPACE',
                                                          'TISSUE_COVERAGE', 'pred_map_location', 'TUMOR_PIXEL_PERC'], how = 'left')
    #Fill NA with the mutation label for the sample, they are the same arcoss all tiles
    cols = selected_labels + ['Bx Type','Anatomic site','Notes','SITE_LOCAL']
    for label in cols:
        if len(comb_df[label].dropna().unique()) > 0:
            comb_df[label].fillna(comb_df[label].dropna().unique()[0], inplace=True)

    return comb_df



def get_label_feature_info_comb(sample_ids, all_tile_info_df, feature_path, fe_method, id_col):
    comb_df_list = []
    ct = 0 
    for pt in sample_ids:
        if ct % 10 == 0 : print(ct)
        
        feature_df = get_sample_feature(pt, feature_path, fe_method)    

        #Get label
        label_df = get_sample_label(pt,all_tile_info_df, id_col = id_col)
        
        #Merge feature and label
        comb_df = combine_feature_and_label(feature_df,label_df)
        
        #Select tumor fraction > X tiles
        comb_df = comb_df.sort_values(by = ['TILE_XY_INDEXES'], ascending = True)
        comb_df.reset_index(inplace = True, drop = True)
        comb_df_list.append(comb_df)
        ct += 1
    
    all_comb_df = pd.concat(comb_df_list)
    
    return all_comb_df
