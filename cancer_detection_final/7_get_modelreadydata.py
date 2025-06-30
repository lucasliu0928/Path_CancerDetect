#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:45:28 2025

@author: jliu6
"""

#!/usr/bin/env python
# coding: utf-8

#NOTE: use python env acmil in ACMIL folder
import sys
import os
import matplotlib
matplotlib.use('Agg')
import warnings
import torch
import yaml
import argparse
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists
from train_utils import get_feature_idexes, get_selected_labels
from train_utils import str2bool, clean_data, get_train_test_val_data_cohort, random_sample_tiles
from train_utils import combine_data_from_stnorm_and_nostnorm
warnings.filterwarnings("ignore")


#FOR ACMIL
current_dir = os.getcwd()
grandparent_subfolder = os.path.join(current_dir, '..', '..', 'other_model_code','ACMIL-main')
grandparent_subfolder = os.path.normpath(grandparent_subfolder)
sys.path.insert(0, grandparent_subfolder)
from utils.utils import Struct
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#source /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/ACMIL-main/acmil/bin/activate

############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Train")
parser.add_argument('--s_fold', default=0, type=int, help='select fold')
parser.add_argument('--loss_method', default='', type=str, help='ATTLOSS or ''')
parser.add_argument('--tumor_frac', default= 0.9, type=int, help='tile tumor fraction threshold')
parser.add_argument('--fe_method', default='uni2', type=str, help='feature extraction model: retccl, uni1, uni2, prov_gigapath')
parser.add_argument('--learning_method', default='acmil', type=str, help=': e.g., acmil, abmil')
parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device name: cuda:0,1,2,3')
parser.add_argument('--mutation', default='MT', type=str, help='Selected Mutation e.g., MT for speciifc mutation name')
parser.add_argument('--train_overlap', default=100, type=int, help='train data pixel overlap')
parser.add_argument('--test_overlap', default=0, type=int, help='test/validation data pixel overlap')
parser.add_argument('--train_cohort', default= 'union_stnormAndnostnorm_OPX_TCGA', type=str, help='TCGA_PRAD or OPX or z_nostnorm_OPX_TCGA or union_stnormAndnostnorm_OPX_TCGA or comb_stnormAndnostnorm_OPX_TCGA')
parser.add_argument('--sample_training_n', default= 1000, type=int, help='random sample K tiles')


            
            
if __name__ == '__main__':
    
    args = parser.parse_args()
    #args.GRL = False
    #args.s_fold  = 0
    args.train_flag = True
    #args.mutation = 'MT'
    #args.train_epoch = 10
    
    fold_list = [0,1,2,3,4]
    for f in fold_list:
        
        args.s_fold = f
    
    
        ####################################
        ######      USERINPUT       ########
        ####################################
        #Feature
        SELECTED_FEATURE = get_feature_idexes(args.fe_method, include_tumor_fraction = False)
        N_FEATURE = len(SELECTED_FEATURE)
        
    
        #Label
        SELECTED_LABEL, selected_label_index = get_selected_labels(args.mutation, args.train_cohort)
        print(SELECTED_LABEL)
        print(selected_label_index)

                        
        #Model Config
        config_dir = "myconf.yml"
        with open(config_dir, "r") as ymlfile:
            c = yaml.load(ymlfile, Loader=yaml.FullLoader)
            conf = Struct(**c)
        conf.train_epoch = args.train_epoch
        conf.D_feat = N_FEATURE
        conf.D_inner = args.DIM_OUT
        conf.n_class = 1
        conf.wandb_mode = 'disabled'
        conf.lr = args.lr
        conf.n_task = len(SELECTED_LABEL)
        conf.sample_training_n = args.sample_training_n
        conf.batchsize = args.batchsize
        conf.learning_method = args.learning_method
        conf.arch = args.arch
        conf.GRL = args.GRL
       
        
        if args.learning_method == 'abmil':
            conf.n_token = 1
            conf.mask_drop = 0
            conf.n_masked_patch = 0
        elif args.learning_method == 'acmil':
            conf.n_token = 3
            conf.mask_drop = 0.6
            conf.n_masked_patch = 0
            
        # Print all key-value pairs in the conf object
        for key, value in conf.__dict__.items():
            print(f"{key}: {value}")
            

            
        ##################
        ###### DIR  ######
        ##################
        proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred'        
    
        ######################
        #Create output-dir
        ######################
        outdir = os.path.join(proj_dir,
                               "intermediate_data", 
                               "7_model_ready_data",
                               'trainCohort_' + args.train_cohort,
                               'TrainOL' + str(args.train_overlap) +  '_TestOL' + str(args.test_overlap) + '_TFT' + str(args.tumor_frac), 
                               'FOLD' + str(args.s_fold))

        create_dir_if_not_exists(outdir)
    
        ##################
        #Select GPU
        ##################
        device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
        print(device)

            
        ################################################
        #     Model ready data 
        ################################################
        data_path = proj_dir + 'intermediate_data/5_combined_data'
        
        #OPX data
        data_ol100_opx_stnorm0, _ = clean_data(data_path, 'z_nostnorm_OPX',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol100_opx_stnorm1, _ = clean_data(data_path, 'OPX',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol0_opx_stnorm0, _ = clean_data(data_path, 'z_nostnorm_OPX',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol0_opx_stnorm1, _ = clean_data(data_path, 'OPX',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_opx_stnorm0 = {'OL100': data_ol100_opx_stnorm0, 'OL0': data_ol0_opx_stnorm0}
        data_opx_stnorm1 = {'OL100': data_ol100_opx_stnorm1, 'OL0': data_ol0_opx_stnorm1}
    
        #TCGA data
        data_ol100_tcga_stnorm0, _ = clean_data(data_path, 'z_nostnorm_TCGA_PRAD',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol100_tcga_stnorm1, _ = clean_data(data_path, 'TCGA_PRAD',args.train_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol0_tcga_stnorm0, _ = clean_data(data_path, 'z_nostnorm_TCGA_PRAD',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol0_tcga_stnorm1, _ = clean_data(data_path, 'TCGA_PRAD',args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_tcga_stnorm0 = {'OL100': data_ol100_tcga_stnorm0, 'OL0': data_ol0_tcga_stnorm0}
        data_tcga_stnorm1 = {'OL100': data_ol100_tcga_stnorm1, 'OL0': data_ol0_tcga_stnorm1}
        
        #Neptune
        data_ol0_nep_stnorm0, nep_ids0   = clean_data(data_path, 'z_nostnorm_Neptune', args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
        data_ol0_nep_stnorm1, nep_ids1   = clean_data(data_path, 'Neptune', args.test_overlap, args.fe_method, args.tumor_frac, selected_label_index)
    
        #Combine stnorm and nostnorm
        data_ol100_opx_union = combine_data_from_stnorm_and_nostnorm(data_ol100_opx_stnorm0, data_ol100_opx_stnorm1, method = 'union')
        data_ol100_opx_comb = combine_data_from_stnorm_and_nostnorm(data_ol100_opx_stnorm0, data_ol100_opx_stnorm1, method = 'combine_all')
        
        data_ol0_opx_union = combine_data_from_stnorm_and_nostnorm(data_ol0_opx_stnorm0, data_ol0_opx_stnorm1, method = 'union')
        data_ol0_opx_comb = combine_data_from_stnorm_and_nostnorm(data_ol0_opx_stnorm0, data_ol0_opx_stnorm1, method = 'combine_all')
        
        data_opx_stnorm10_union = {'OL100': data_ol100_opx_union, 'OL0': data_ol0_opx_union}
        data_opx_stnorm10_comb = {'OL100': data_ol100_opx_comb, 'OL0': data_ol0_opx_comb}

        
        data_ol100_tcga_union = combine_data_from_stnorm_and_nostnorm(data_ol100_tcga_stnorm0, data_ol100_tcga_stnorm1, method = 'union')
        data_ol100_tcga_comb = combine_data_from_stnorm_and_nostnorm(data_ol100_tcga_stnorm0, data_ol100_tcga_stnorm1, method = 'combine_all')
        
        data_ol0_tcga_union = combine_data_from_stnorm_and_nostnorm(data_ol0_tcga_stnorm0, data_ol0_tcga_stnorm1, method = 'union')
        data_ol0_tcga_comb = combine_data_from_stnorm_and_nostnorm(data_ol0_tcga_stnorm0, data_ol0_tcga_stnorm1, method = 'combine_all')
        
        data_tcga_stnorm10_union = {'OL100': data_ol100_tcga_union, 'OL0': data_ol0_tcga_union}
        data_tcga_stnorm10_comb = {'OL100': data_ol100_tcga_comb, 'OL0': data_ol0_tcga_comb}
        
        
        data_ol0_nep_union = combine_data_from_stnorm_and_nostnorm(data_ol0_nep_stnorm0, data_ol0_nep_stnorm1, method = 'union')
        nep_id = [entry[-2] for i, entry in enumerate(data_ol0_nep_union)]
        data_ol0_nep_comb = combine_data_from_stnorm_and_nostnorm(data_ol0_nep_stnorm0, data_ol0_nep_stnorm1, method = 'combine_all')
        nep_id = [entry[-2] for i, entry in enumerate(data_ol0_nep_comb)]
    
        #TODO Actual: Check OPX_001 was removed beased no cancer detected in stnormed
        ################################################
        #Get Train, test, val data
        ################################################    
        id_data_dir = proj_dir + 'intermediate_data/3B_Train_TEST_IDS'
        
        if args.train_cohort == 'z_nostnorm_OPX_TCGA':
            train_cohort1 = 'z_nostnorm_OPX'
            model_data1 = data_opx_stnorm0
            train_cohort2 = 'z_nostnorm_TCGA_PRAD'
            model_data2 = data_tcga_stnorm0
            
        elif args.train_cohort == 'OPX_TCGA':
            train_cohort1 = 'OPX'
            model_data1 = data_opx_stnorm1
            train_cohort2 = 'TCGA_PRAD'
            model_data2 = data_tcga_stnorm1
            
        elif args.train_cohort == 'union_stnormAndnostnorm_OPX_TCGA':
            train_cohort1 = 'OPX'
            model_data1 = data_opx_stnorm10_union
            train_cohort2 = 'TCGA_PRAD'
            model_data2 = data_tcga_stnorm10_union
        elif args.train_cohort == 'comb_stnormAndnostnorm_OPX_TCGA':
            train_cohort1 = 'OPX'
            model_data1 = data_opx_stnorm10_comb
            train_cohort2 = 'TCGA_PRAD'
            model_data2 = data_tcga_stnorm10_comb
            

            
        ################################################################################################
        #For training and test data, take the union of tiles from stained normed and nostained normed tiles
        ################################################################################################
        
        (train_data1, train_ids1), (val_data1, val_ids1), (test_data1, test_ids1) = get_train_test_val_data_cohort(id_data_dir, 
                                                                                                                   train_cohort1 ,
                                                                                                                   model_data = model_data1, 
                                                                                                                   tumor_frac = args.tumor_frac, 
                                                                                                                   s_fold = args.s_fold)
        
        (train_data2, train_ids2), (val_data2, val_ids2), (test_data2, test_ids2) = get_train_test_val_data_cohort(id_data_dir, 
                                                                                                                   train_cohort2 ,
                                                                                                                   model_data = model_data2, 
                                                                                                                   tumor_frac = args.tumor_frac, 
                                                                                                                   s_fold = args.s_fold)
        
        


            
    
        ################################################################################
        #Get Final train and test and val data
        ################################################################################
        train_data = train_data1 + train_data2
        train_ids = train_ids1 + train_ids2
        
        val_data = val_data1 + val_data2
        val_ids = val_ids1 + val_ids2
        
        test_data = test_data1 + test_data2 #put two test together
        test_ids = test_ids1 + test_ids2
        
        if args.train_cohort != 'comb_stnormAndnostnorm_OPX_TCGA':
            if conf.sample_training_n > 0:
                #Random Sample 1000 tiles or oriingal N tiles (if total number is < 1000) for training data
                random_sample_tiles(train_data, k = conf.sample_training_n, random_seed = 42)

        if args.train_cohort == 'comb_stnormAndnostnorm_OPX_TCGA': 
            #Keep feature1, label, tf,1 dlabel, feature2, tf2
            train_data = [(item[0], item[1], item[2], item[3], item[7], item[9]) for item in train_data]
            test_data1 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data1]
            test_data2 = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data2]
            test_data = [(item[0], item[1], item[2], item[6], item[8]) for item in test_data]
            val_data = [(item[0], item[1], item[2],  item[6], item[8]) for item in val_data]
        else:
            #Exclude tile info data, sample ID, patient ID, do not needed it for training
            train_data = [item[:-3] for item in train_data]
            test_data1 = [item[:-3] for item in test_data1]   
            test_data2 = [item[:-3] for item in test_data2]   
            test_data = [item[:-3] for item in test_data]   
            val_data = [item[:-3] for item in val_data]
        

        
        ext_data_nep_st0 = [item[:-3] for item in data_ol0_nep_stnorm0] #no st norm
        ext_data_nep_st1 = [item[:-3] for item in data_ol0_nep_stnorm1] #st normed
        ext_data_nep_union = [item[:-3] for item in data_ol0_nep_union]
        
        
        #write to pt
        torch.save({"data": train_data, "id": train_ids}, 'train_data.pt')
        torch.save({"data": test_data1, "id": test_ids1}, 'test_data1.pt')
        torch.save({"data": test_data2, "id": test_ids2}, 'test_data2.pt')
        torch.save({"data": test_data, "id": test_ids}, 'test_data.pt')
        torch.save({"data": val_data, "id": val_ids}, 'val_data.pt')
        torch.save({"data": ext_data_nep_st0, "id": nep_ids0}, 'ext_data_nep_st0.pt')
        torch.save({"data": ext_data_nep_st1, "id": nep_ids1}, 'ext_data_nep_st1.pt')
        torch.save({"data": ext_data_nep_union, "id": nep_id}, 'ext_data_nep_union.pt')
        




        
        
        

            
    
