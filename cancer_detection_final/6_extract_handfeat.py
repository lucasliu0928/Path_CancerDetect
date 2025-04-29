import sys
import os
import numpy as np
import openslide
import pandas as pd
import argparse
from histomicstk import preprocessing,features
from skimage import io, measure
from cellpose import models
import warnings
import PIL
from openslide.deepzoom import DeepZoomGenerator
import glob
warnings.filterwarnings("ignore")

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")

def generate_deepzoom_tiles(slide, save_image_size, pixel_overlap, limit_bounds):
    # this is physical microns per pixel
    acq_mag = 10.0/float(slide.properties[openslide.PROPERTY_NAME_MPP_X])

    # this is nearest multiple of 20 for base layer
    base_mag = int(20 * round(float(acq_mag) / 20))

    # this is how much we need to resample our physical patches for uniformity across studies
    physSize = round(save_image_size*acq_mag/base_mag)

    # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
    tiles = DeepZoomGenerator(slide, tile_size=physSize-round(pixel_overlap*acq_mag/base_mag), overlap=round(pixel_overlap*acq_mag/base_mag/2), 
                              limit_bounds=limit_bounds)

    # calculate the effective magnification at each level of tiles, determined from base magnification
    tile_lvls = tuple(base_mag/(tiles._l_z_downsamples[i]*tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in range(0,tiles.level_count))

    return tiles, tile_lvls, physSize, base_mag

#RUN
#source ~/.bashrc
#conda activate paimg9
#python3 -u 1_extract_patches_fixed-res.py  --cohort_name Neptune --pixel_overlap 0 
############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--save_image_size', default='250', type=int, help='the size of extracted tiles')
parser.add_argument('--pixel_overlap', default='0', type=int, help='specify the level of pixel overlap in your saved tiles, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--cohort_name', default='OPX', type=str, help='data set name: TAN_TMA_Cores, OPX, TCGA_PRAD, Neptune')
parser.add_argument('--TUMOR_FRAC_THRES', default= 0.9, type=int, help='tile tumor fraction threshold')



if __name__ == '__main__':
    
    args = parser.parse_args()
    
    
    ############################################################################################################
    #USER INPUT 
    ############################################################################################################
    folder_name = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.pixel_overlap)


    #DIR
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    image_path = os.path.join(proj_dir,'data', args.cohort_name)
    tile_info_path = os.path.join(proj_dir,'intermediate_data/2_cancer_detection/', args.cohort_name, folder_name)
    tile_norm_img_path = os.path.join(proj_dir,'intermediate_data/6A_tile_for_stain_norm/')
    out_location = os.path.join(proj_dir,'intermediate_data','6_hand_crafted_feature', args.cohort_name, folder_name)  #1_feature_extraction, cancer_prediction_results110224

    #Create output dir
    create_dir_if_not_exists(out_location)


    
    ############################################################################################################
    #Select IDS
    ############################################################################################################
    selected_ids =  os.listdir(tile_info_path)
    selected_ids = [x for x in selected_ids if x != '.DS_Store']
    selected_ids.sort()
    

    ############################################################################################################
    #Start 
    ############################################################################################################
    for cur_id in selected_ids:
        cur_id = selected_ids[0]    
        
        save_location = out_location + "/" + cur_id + "/" 
        create_dir_if_not_exists(save_location)
    
        #check if processed:
        imgout = glob.glob(save_location + "*.csv")
        if len(imgout) > 0:
             print(cur_id + ': already processed')
        elif len(imgout) == 0:            
            if args.cohort_name == 'TCGA_PRAD':
                slides_name = [f for f in os.listdir(image_path + cur_id + '/') if '.svs' in f][0].replace('.svs','')
                _file = os.path.join(image_path, cur_id, slides_name + ".svs")
                _tile_file = os.path.join(tile_info_path, cur_id, 'ft_model', slides_name + "_TILE_TUMOR_PERC.csv")
            else:
                slides_name = cur_id
                _file = os.path.join(image_path, slides_name + ".tif")
                _tile_file = os.path.join(tile_info_path, cur_id, 'ft_model', cur_id + "_TILE_TUMOR_PERC.csv")
                                
            #this is a tile which we use for stain normalization
            image2 = io.imread(os.path.join(tile_norm_img_path, 'SU21-19308_A1-2_HE_40X_MH110821_40_16500-20500_500-500.png'))
            
            #load cellpose model
            model = models.Cellpose(model_type='nuclei',gpu=True)
            
            #Read Sldies
            oslide = openslide.OpenSlide(_file)
            
            #Load tile info
            tile_info_df = pd.read_csv(_tile_file)
            
            #Get slides info
            save_image_size = tile_info_df['SAVE_IMAGE_SIZE'].unique().item()
            pixel_overlap   = tile_info_df['PIXEL_OVERLAP'].unique().item()
            limit_bounds    = tile_info_df['LIMIT_BOUNDS'].unique().item()
            mag_extract     = tile_info_df['MAG_EXTRACT'].unique().item()
        
            #Generate tiles
            tiles, tile_lvls, physSize, base_mag = generate_deepzoom_tiles(oslide,save_image_size, pixel_overlap, limit_bounds)
        
            #Get feature , we have 325 features which are extracted
            all_features_list = []
            save_cols = None
            for index, row in tile_info_df.iterrows():
                if (index % 100 == 0): print(index)
                #cur_cd_prob = row['TUMOR_PIXEL_PERC']
                try:
                    #Get index
                    cur_xy = row['TILE_XY_INDEXES'].strip("()").split(", ")
                    x ,y = int(cur_xy[0]) , int(cur_xy[1])
            
                    #Extract tile for prediction
                    lvl_in_deepzoom = tile_lvls.index(mag_extract)
                    tile_pull = tiles.get_tile(lvl_in_deepzoom, (x, y))
                    tile_pull = tile_pull.resize(size=(save_image_size, save_image_size),resample=PIL.Image.LANCZOS) #resize
                    
                    #run cellcellpose
                    masks, flows, styles, diams = model.eval(np.asarray(tile_pull),channels=[0,0],diameter=15,invert=True,resample=False)
                    
                    #preprocess tile for handcrafted features
                    newimg = preprocessing.color_normalization.deconvolution_based_normalization(im_src=np.asarray(tile_pull), im_target=image2) #a color-adjusted version of your input tile 
                    simg = preprocessing.color_deconvolution.color_deconvolution(im_rgb=newimg, w=([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])) # the result of decomposing newimg into separate stain components (channels) based on the color properties
                    nucprops = measure.regionprops(masks)
                    nuc_cen = [prop.centroid for prop in nucprops] #the centroid of nucleus
                    ncen = np.array(nuc_cen).reshape(len(nuc_cen), 2) #the centroid of nucleus
                    num_cen_df = pd.DataFrame({"N_nucleus": len(nuc_cen)}, index = [0]).reset_index(drop=True)
                    
                    #get graph features
                    gpf = features.compute_global_cell_graph_features(centroids=ncen).reset_index(drop=True)
                    #get nuclei features
                    ncf = features.compute_nuclei_features(im_label=masks, im_nuclei=simg.Stains[:, :, 0],im_cytoplasm=simg.Stains[:, :, 1])
                    
                    #we take mean and stdev of nuclei features
                    ncf_mean_df = ncf.mean(axis = 0).to_frame().T
                    ncf_mean_df = ncf_mean_df.add_suffix('_mean').reset_index(drop=True)
                
                    ncf_std_df  = ncf.std(axis = 0).to_frame().T
                    ncf_std_df = ncf_std_df.add_suffix('_std').reset_index(drop=True)
                    
                    # concat together
                    allfeats = pd.concat([row.to_frame().T.reset_index(drop=True), num_cen_df, ncf_mean_df, ncf_std_df, gpf], axis=1, ignore_index=False)
                    
                    if save_cols is None:
                        save_cols = allfeats.columns
                except Exception as e:
                    print(f"Warning: Failed at index {index} with error: {e}")
                    all_feats_emp = pd.DataFrame([[np.nan] * 325], columns = [f'feature_{i}' for i in range(325)]).reset_index(drop=True)
                    allfeats = pd.concat([row.to_frame().T.reset_index(drop=True), all_feats_emp], axis=1, ignore_index=False)
        
                all_features_list.append(allfeats)
        
            #Change colum names
            for df in all_features_list:
                if 'feature_0' in df.columns:
                    df.columns = save_cols
        
            all_features_df = pd.concat(all_features_list)
            all_features_df.to_csv(os.path.join(save_location, slides_name + "_handcrafted_feature.csv"), index = False)
        
            # Save to CSV
            import time
            start = time.time()
            all_features_df.to_csv(os.path.join(save_location, slides_name + "_handcrafted_feature.csv"), index = False)
            print('CSV save time:', time.time() - start)
            
            # Save to HDF5
            start = time.time()
            all_features_df.to_hdf(os.path.join(save_location, slides_name + "_handcrafted_feature.h5"), key='hand_crafted_feature', mode='w')
            print('HDF5 save time:', time.time() - start)


#Plot normed images
# import matplotlib.pyplot as plt
# plt.imshow(tile_pull)
# plt.title('Normalized Image (newimg)')
# plt.axis('off')
# plt.show()


# # Access the actual stain-separated images
# stains = simg.Stains  # shape (height, width, channels)

# # Plot each stain
# fig, axs = plt.subplots(1, stains.shape[2], figsize=(15, 5))

# for i in range(stains.shape[2]):
#     axs[i].imshow(stains[:, :, i])
#     axs[i].axis('off')

# plt.tight_layout()
# plt.show()