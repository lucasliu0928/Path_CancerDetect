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
import glob
from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import DataLoader
sys.path.insert(0, '../Utils/')
from misc_utils import create_dir_if_not_exists
from train_utils import str2bool
warnings.filterwarnings("ignore")

#the following functions are the same one in Utils, only becauase of env issue, so add it here
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


def extract_tile_start_end_coords_tma(x_loc, y_loc, tile_size, overlap):
    r'''
    #This func returns the coordiates in the original image of TMA at original dim
    '''
    #Get stride
    stride = tile_size - overlap
    
    #Get top left pixel coordinates
    topleft_x = stride * (x_loc - 1) 
    topleft_y = stride * (y_loc - 1)
    
    #Get region size in current level 
    rsize_x = tile_size 
    rsize_y = tile_size
    
    #Get tile starts and end   
    start_loc = (topleft_x, topleft_y) #start
    end_loc = (topleft_x + rsize_x, topleft_y + rsize_y) #end
    
    #Get save coord name (first two is the starting loc, and the last two are the x and y size considering dsfactor)
    coord_name = str(topleft_x) + "-" + str(topleft_y) + "_" + '%.0f' % (rsize_x) + "-" + '%.0f' % (rsize_y)

    #Get tile_coords the same format as OPX, (start_loc, deepzzomlvl, rise)
    tile_coords = (start_loc, 'NA', (rsize_x, rsize_y))
    
    return start_loc, end_loc, coord_name, tile_coords

        
#RUN
#source /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_ppl_code/handcrafted_features/hf_env/bin/activate
#python3 -u 6_extract_handfeat.py  --cohort_name Pluvicto_TMA_Cores --select_idx_start 515 --select_idx_end 606 
############################################################################################################
#Parser
############################################################################################################
parser = argparse.ArgumentParser("Tile feature extraction")
parser.add_argument('--save_image_size', default='250', type=int, help='the size of extracted tiles')
parser.add_argument('--pixel_overlap', default='0', type=int, help='specify the level of pixel overlap in your saved tiles, do not change this, model trained at 250x250 at 20x')
parser.add_argument('--cohort_name', default='Pluvicto_TMA_Cores', type=str, help='data set name: TAN_TMA_Cores, OPX, TCGA_PRAD, Neptune, Pluvicto_TMA_Cores')
parser.add_argument('--stain_norm', default='norm', type=str, help='norm or no_norm')
parser.add_argument('--fine_tuned_model', type=str2bool, default=False, help='whether or not to use fine-tuned model, for TMA or Pluvicto_TMA_Cores do not use FT model')
parser.add_argument('--select_idx_start', default = 0, type=int)
parser.add_argument('--select_idx_end', default = 1, type=int)

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    
    ############################################################################################################
    #USER INPUT 
    ############################################################################################################
    folder_name = "IMSIZE" + str(args.save_image_size) + "_OL" + str(args.pixel_overlap)


    #DIR
    proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
    image_path = os.path.join(proj_dir,'data', args.cohort_name)
    tile_info_path = os.path.join(proj_dir,'intermediate_data/2_cancer_detection/', args.cohort_name, folder_name) #this file contians all tiles after exclude white space and non-tissue, contains all cancer fractions
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
    #Load normalization norm target image
    ############################################################################################################
    if args.stain_norm == "norm":
        norm_target_img = io.imread(os.path.join(tile_norm_img_path, 'SU21-19308_A1-2_HE_40X_MH110821_40_16500-20500_500-500.png'))
    elif args.stain_norm == "no_norm":
        norm_target_img = None
        
    ############################################################################################################
    #Start 
    ############################################################################################################
    for cur_id in selected_ids[args.select_idx_start:args.select_idx_end]:
        save_location = out_location + "/" + cur_id + "/" 
        create_dir_if_not_exists(save_location)
    
        #check if processed:
        imgout = glob.glob(save_location + "*.csv")
        if len(imgout) > 0:
             print(cur_id + ': already processed')
        elif len(imgout) == 0:            
            if 'TCGA_PRAD'in args.cohort_name:
                slides_name = [f for f in os.listdir(os.path.join(image_path, cur_id)) if '.svs' in f][0].replace('.svs','')
                _file = os.path.join(image_path, cur_id, slides_name + ".svs")
                
                if args.fine_tuned_model == True:
                    _tile_file = os.path.join(tile_info_path, cur_id, 'ft_model', slides_name + "_TILE_TUMOR_PERC.csv")
                else:
                    _tile_file = os.path.join(tile_info_path, cur_id, 'prior_model', slides_name + "_TILE_TUMOR_PERC.csv")   
            else:
                slides_name = cur_id
                _file = os.path.join(image_path, slides_name + ".tif")
                
                if args.fine_tuned_model == True:
                    _tile_file = os.path.join(tile_info_path, cur_id, 'ft_model', cur_id + "_TILE_TUMOR_PERC.csv")
                else:
                    _tile_file = os.path.join(tile_info_path, cur_id, 'prior_model', cur_id + "_TILE_TUMOR_PERC.csv")
            
            #load cellpose model
            model = models.Cellpose(model_type='nuclei',gpu=True)
            
            #Load tile info
            tile_info_df = pd.read_csv(_tile_file)
            
            #Get slides info
            save_image_size = tile_info_df['SAVE_IMAGE_SIZE'].unique().item()
            pixel_overlap   = tile_info_df['PIXEL_OVERLAP'].unique().item()
            limit_bounds    = tile_info_df['LIMIT_BOUNDS'].unique().item()
            mag_extract     = tile_info_df['MAG_EXTRACT'].unique().item()
            

            #Read Sldies
            if 'TMA' in args.cohort_name:
                oslide = PIL.Image.open(_file)
            else:
                oslide = openslide.OpenSlide(_file)
            

        
            #Generate tiles
            if 'TMA' not in args.cohort_name:
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
                    
                    if 'TMA' in args.cohort_name:
                        #Grab tile coordinates
                        #1st way
                        # tile_startend, hw = row['TILE_COOR_ATLV0'].split('_')
                        # tile_starts = [int(s) for s in tile_startend.split('-')]
                        # tile_ends = [tile_starts[0] + int(hw.split('-')[0]), tile_starts[1] + int(hw.split('-')[1])]
                        #2ndway
                        tile_starts, tile_ends, _, _ = extract_tile_start_end_coords_tma(x, y, tile_size = save_image_size, overlap = pixel_overlap)
                        tile_pull = oslide.crop(box=(tile_starts[0], tile_starts[1], tile_ends[0], tile_ends[1]))
                        tile_pull = tile_pull.convert("RGB")
                    else:
                        #Extract tile for prediction
                        lvl_in_deepzoom = tile_lvls.index(mag_extract)
                        tile_pull = tiles.get_tile(lvl_in_deepzoom, (x, y))
                        
                    tile_pull = tile_pull.resize(size=(save_image_size, save_image_size),resample=PIL.Image.LANCZOS) #resize
                    tile_pull
                    #run cellcellpose
                    masks, flows, styles, diams = model.eval(np.asarray(tile_pull),channels=[0,0],diameter=15,invert=True,resample=False)
                    
                    #preprocess tile for handcrafted features
                    if norm_target_img is not None:
                        try:
                            newimg = preprocessing.color_normalization.deconvolution_based_normalization(im_src=np.asarray(tile_pull), im_target=norm_target_img) #a color-adjusted version of your input tile 
                        except np.linalg.LinAlgError:
                            print("Deconvolution failed on a tile – skipping") #this is due to some tiles are not actuallly not tissue, all black (Neptune), just skip the norm
                            pass
                    else:
                        newimg = np.asarray(tile_pull)
                    
                    #color deconvolution
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