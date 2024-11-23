import sys
import os
import numpy as np
from PIL import Image
from PIL import ImageOps
import cv2
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pandas as pd
from skimage import draw
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
import geojson
import argparse
from shapely.ops import cascaded_union, unary_union
from pathlib import Path


class extractPatch:

    def __init__(self):
        self.proj_dir = '/fh/scratch/delete90/etzioni_r/lucas_l/michael_project/mutation_pred/'
        self.csv_location =  self.proj_dir + '/intermediate_data/cd_finetune/cancer_detection_training/ccola_opx_retrain_alldata_list.csv' 
        self.json_location = self.proj_dir + '/intermediate_data/cd_finetune/ccola_opx_annotations/' 
        self.save_path = self.proj_dir + '/intermediate_data/cd_finetune/cancer_detection_training/patches/'
        self.mag_extract = [20] # specify which magnifications you wish to pull images from
        self.save_image_size = 250  # specify image size to be saved (note this is the same for all magnifications)
        self.pixel_overlap = 0       # specify the level of pixel overlap in your saved images
        self.limit_bounds = True     # this is weird, dont change it
        self.write_all = False       # default is to only write patches that overlap with xml regions (if no xml provided, all patches written)
        self.nolabel = False         # if all regions in an annotation file belong to the same class, they are labeled as 'tumor'
                                     #      nolabel=FALSE should only be used if the "Text" attribute in xml corresponds to label
        

    def parseMeta_and_pullTiles(self):
        df = pd.read_csv(self.csv_location)
        flist = df['filename']
        pathlist = df['filepath']

        #TODO: change the 4 in range back to 0.
        for f in range(4,len(flist)):
            _file = os.path.join(pathlist[f], flist[f])
            print(_file)

            #Get slide
            oslide = openslide.OpenSlide(_file)

            #Get names
            save_name = str(Path(os.path.basename(_file)).with_suffix(''))
            json_file = save_name + '.json' #json file name

            #Create individual dir
            ind_save_path = os.path.join(self.save_path,save_name)
            self.create_dir_if_not_exists(ind_save_path)

            
            #Generate deep zoom tile object 
            tiles, tile_lvls, physSize = self.generate_deepzoom_tiles(oslide)

    
            # pull tiles from levels specified by self.mag_extract
            for lvl in self.mag_extract:
                if lvl in tile_lvls:
                    x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(lvl)]

                    for y in range(0, y_tiles):
                        for x in range(0, x_tiles):
                            #grab tile coordinates
                            tile_starts, tile_ends, save_coords = self.extract_tile_start_end_coords(tiles, tile_lvls.index(lvl), x, y)

                            # label tile based on xml region membership
                            tile_labels = self.assign_label_new2(tile_starts,tile_ends,os.path.join(self.json_location, json_file))

                            if not tile_labels:
                                pass
                            else:
                                print(x,y)
                                tile_size = tiles.get_tile_dimensions(tile_lvls.index(lvl), (x, y))
                                tile_pull = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                                self.save_to_disk(tile_pull=tile_pull, save_coords=save_coords, lvl=lvl, 
                                    tile_label=tile_labels, tile_size=tile_size, physSize=physSize, 
                                    save_location= ind_save_path, save_name = save_name)

                else:
                    print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")
        return


    def extract_tile_start_end_coords(self, all_tile, deepzoom_lvl, x_loc, y_loc):
        #Get coords
        tile_coords = all_tile.get_tile_coordinates(deepzoom_lvl, (x_loc, y_loc))

        #Get top left pixel coordinates
        topleft_x = tile_coords[0][0]
        topleft_y = tile_coords[0][1]

        #Get level (original)
        o_lvl = tile_coords[1]

        #Get downsample factor
        ds_factor = all_tile._l0_l_downsamples[o_lvl] #downsample factor

        #Get region size in current level 
        rsize_x = tile_coords[2][0] 
        rsize_y = tile_coords[2][1] 

        #Get tile starts and end   
        start_loc = tile_coords[0] #start
        end_loc = (int(topleft_x + ds_factor * rsize_x), int(topleft_y + ds_factor* rsize_y)) #end

        #Get save coord name (first two is the starting loc, and the last two are the x and y size considering dsfactor)
        coord_name = str(topleft_x) + "-" + str(topleft_y) + "_" + '%.0f' % (ds_factor * rsize_x) + "-" + '%.0f' % (ds_factor * rsize_y)
        
        return start_loc, end_loc, coord_name

    def generate_deepzoom_tiles(self,slide):
        # this is physical microns per pixel
        acq_mag = 10.0/float(slide.properties[openslide.PROPERTY_NAME_MPP_X])

        # this is nearest multiple of 20 for base layer
        base_mag = int(20 * round(float(acq_mag) / 20))
        print(base_mag)

        # this is how much we need to resample our physical patches for uniformity across studies
        physSize = round(self.save_image_size*acq_mag/base_mag)

        # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
        tiles = DeepZoomGenerator(slide, tile_size=physSize-round(self.pixel_overlap*acq_mag/base_mag), overlap=round(self.pixel_overlap*acq_mag/base_mag/2), limit_bounds=self.limit_bounds)

        # calculate the effective magnification at each level of tiles, determined from base magnification
        tile_lvls = tuple(base_mag/(tiles._l_z_downsamples[i]*tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in range(0,tiles.level_count))

        return tiles, tile_lvls, physSize

    def whitespace_check(self,im):
        bw = im.convert('L')
        bw = np.array(bw)
        bw = bw.astype('float')
        bw=bw/255
        prop_ws = (bw > 0.8).sum()/(bw>0).sum()
        return prop_ws


    def assign_label_new2(self, tile_starts,tile_ends,path):
        ''' calculates overlap of tile with xml regions and creates dictionary based on unique labels '''

        tile_box = [tile_starts[0],tile_starts[1]],[tile_starts[0],tile_ends[1]],[tile_ends[0],tile_starts[1]],[tile_ends[0],tile_ends[1]]
        tile_box = list(tile_box)
        tile_box = MultiPoint(tile_box).convex_hull

        with open(path) as f:
            allobjects = geojson.load(f)

        # Added1023: Filter out objects without a classification label
        # Note: checked OPX024, there is one repelicate annotation without label
        allobjects = [obj for obj in allobjects if 'classification' in obj['properties']]

        allshapes = [shape(obj["geometry"]) for obj in allobjects]
        alllabels = [obj['properties'] for obj in allobjects]
        roilabels = list()
        for roi_num in range(0,len(alllabels)):
            try:
                roi_label = alllabels[roi_num]['classification']['name']
            except:
                roi_label = 'Tumor'
            roi_label = str.replace(roi_label,'Tissue','Lung')
            roilabels.append(roi_label)

        tile_label = {}
        for label in np.unique(roilabels):
            # loop over every region associated with a given label, sum the overlap
            box_label = False  # initialize
            ov = 0  # initialize
            ov_mask = np.zeros((int(max(tile_box.exterior.xy[0]) - min(tile_box.exterior.xy[0])),
                                int(max(tile_box.exterior.xy[0]) - min(tile_box.exterior.xy[0]))), dtype=int)
            for roi_num in range(0,len(alllabels)):
                if roilabels[roi_num] == label:
                    roi = allshapes[roi_num]
                    if tile_box.intersects(roi):
                        box_label = True
                        ov_reg = tile_box.intersection(roi)
                        ov += ov_reg.area / tile_box.area

                        if ov_reg.geom_type == 'Polygon':
                            reg_mask = self.poly2mask(
                                vertex_row_coords=[x - min(tile_box.exterior.xy[1]) for x in ov_reg.exterior.xy[1]],
                                vertex_col_coords=[x - min(tile_box.exterior.xy[0]) for x in ov_reg.exterior.xy[0]],
                                shape=(int(max(tile_box.exterior.xy[0]) - min(tile_box.exterior.xy[0])),
                                       int(max(tile_box.exterior.xy[0]) - min(tile_box.exterior.xy[0]))))
                            # print(reg_mask.shape)
                            ov_mask += reg_mask
                        elif ov_reg.geom_type == 'MultiPolygon':
                            for roii in ov_reg.geoms:
                                reg_mask = self.poly2mask(
                                    vertex_row_coords=[x - min(tile_box.exterior.xy[1]) for x in roii.exterior.xy[1]],
                                    vertex_col_coords=[x - min(tile_box.exterior.xy[0]) for x in roii.exterior.xy[0]],
                                    shape=(int(max(tile_box.exterior.xy[0]) - min(tile_box.exterior.xy[0])),
                                           int(max(tile_box.exterior.xy[0]) - min(tile_box.exterior.xy[0]))))
                                # print(reg_mask.shape)
                                ov_mask += reg_mask

            if box_label == True:
                tile_label[label] = ov_mask

        # p.s. if you are curious, you can plot the polygons by the following
        # for polygon in roi.geoms:
        #     plt.plot(*polygon.exterior.xy) and plt.plot(*tile_box.exterior.xy)
        
        return tile_label



    def poly2mask(self,vertex_row_coords, vertex_col_coords, shape):
        ''''''
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=int)
        mask[fill_row_coords, fill_col_coords] = 1
        return mask


    def save_to_disk(self, tile_pull, save_coords, lvl, tile_label, tile_size, physSize, save_location,save_name):

        if 'cancer' in tile_label:
            tile_mask = tile_label['cancer']
        elif 'benign' in tile_label:
            tile_mask = tile_label['benign']
            
        tile_mask = tile_mask.astype(bool)
        mask_pull = Image.fromarray(tile_mask)
        # edge tiles will not be correct size (too small), so we reflect image data until correct size
        if tile_size[0]<physSize or tile_size[1]<physSize:
            # tile_pull = Image.fromarray(cv2.copyMakeBorder(np.array(tile_pull), 0, physSize - int(tile_size[1]), 0, physSize - int(tile_size[0]),cv2.BORDER_REFLECT))
            # mask_pull = Image.fromarray(cv2.copyMakeBorder(tile_mask, 0, physSize - int(tile_size[1]), 0,physSize - int(tile_size[0]), cv2.BORDER_REFLECT))
            return
        else:
            # check whitespace amount
            ws = self.whitespace_check(im=tile_pull)
            if ws < 0.95:
                mask_np = np.array(mask_pull)
                #if benign is the label, change the original mask to all 0s
                if 'benign' in tile_label:
                    mask_np.fill(0) 

                tumorperc = (mask_np == True).sum()/(mask_pull.size[0]*mask_pull.size[1])
                mask_np = mask_np.astype('uint8')
                mask_save = Image.fromarray(mask_np)
                tile_savename = save_name + "_" + str(lvl) + "_" \
                                + save_coords + "_" \
                                + "ws-" + '%.2f' % (ws) \
                                + "_tumor-" + '%.2f' % (tumorperc)
                tile_pull = tile_pull.resize(size=(self.save_image_size, self.save_image_size), resample=Image.LANCZOS)
                mask_save = mask_save.resize(size=(self.save_image_size, self.save_image_size), resample=Image.LANCZOS)
                tile_pull.save(os.path.join(save_location, tile_savename + ".png"))
                mask_save.save(os.path.join(save_location, tile_savename + "_mask.png"))
        return

    def create_dir_if_not_exists(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory '{dir_path}' created.")
        else:
            print(f"Directory '{dir_path}' already exists.")


if __name__ == '__main__':
    c = extractPatch()
    c.parseMeta_and_pullTiles()