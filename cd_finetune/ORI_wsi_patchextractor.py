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
        self.csv_location = '/fh/scratch/delete90/haffner_m/user/hrichards/digital_pathology/CUT_data/TAN_WSI_HE_train_test_split.csv' #'/fh/scratch/delete90/haffner_m/user/hrichards/WCDT/fh_samples_train_test_splits.xlsx' 
        self.json_location = '/fh/scratch/delete90/haffner_m/user/hrichards/digital_pathology/CUT_data/CUT_WCDT/TAN_tissue' #'/fh/scratch/delete90/haffner_m/user/hrichards/digital_pathology/CUT_data/CUT_WCDT/annotations/edited_annotations'
        self.save_path = '/fh/scratch/delete90/haffner_m/user/hrichards/digital_pathology/CUT_data/CUT_WCDT/20x_data_FH_cleaned'
        self.mag_extract = [20] # specify which magnifications you wish to pull images from
        self.save_image_size = 256  # specify image size to be saved (note this is the same for all magnifications)
        self.pixel_overlap = 0       # specify the level of pixel overlap in your saved images
        self.limit_bounds = True     # this is weird, dont change it
        self.write_all = False       # default is to only write patches that overlap with xml regions (if no xml provided, all patches written)
        self.nolabel = False         # if all regions in an annotation file belong to the same class, they are labeled as 'tumor'
                                     #      nolabel=FALSE should only be used if the "Text" attribute in xml corresponds to label
        

    def parseMeta_and_pullTiles(self):
        df = pd.read_csv(self.csv_location)
        #df = pd.read_excel(self.csv_location)
        flist = df['filename']
        pathlist = df['filepath']

        for f in range(len(flist)):
            _file = os.path.join(pathlist[f], flist[f])
            print(_file)
            print(df['group'][f])
            oslide = openslide.OpenSlide(_file)
            savnm = os.path.basename(_file)
            self.save_name = str(Path(savnm).with_suffix(''))
            json_file = self.save_name + '_tissue.json'

            if df['group'][f] == 'test':
                self.save_location = os.path.join(self.save_path, 'testB')

                #CHANGE EVERY BACK AND UNINDENT AFTER
            # else:
            #     self.save_location = os.path.join(self.save_path, 'trainA')

            #     print(self.save_location)
    
                if not os.path.exists(os.path.join(self.save_location)):
                    os.mkdir(os.path.join(self.save_location))
    
                
            # for index, scan in allscans.iterrows():
            #     self.image_file = scan['image_file']
            #     self.xml_file = scan['xml_file']
            #     self.xml_location = scan['xml_location']
            #     self.save_name = str.replace(scan['image_file'],'.tif','')
            #     self.save_location = '/fh/scratch/delete90/haffner_m/user/hrichards/digital_pathology/CUT_data/CUT_WCDT/tiles'+self.save_name
    
            #     if not os.path.exists(os.path.join(self.save_location)):
            #         os.mkdir(os.path.join(self.save_location))
    
            #         print(os.path.join(self.file_location,self.image_file))
            #         oslide = openslide.OpenSlide(os.path.join(self.file_location,self.image_file))
    
                # this is physical microns per pixel
                acq_mag = 10.0/float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
    
                # this is nearest multiple of 20 for base layer
                base_mag = int(20 * round(float(acq_mag) / 20))
                print(base_mag)
                # this is how much we need to resample our physical patches for uniformity across studies
                physSize = round(self.save_image_size*acq_mag/base_mag)
    
                # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
                tiles = DeepZoomGenerator(oslide, tile_size=physSize-round(self.pixel_overlap*acq_mag/base_mag), overlap=round(self.pixel_overlap*acq_mag/base_mag/2), limit_bounds=self.limit_bounds)
    
                # calculate the effective magnification at each level of tiles, determined from base magnification
                tile_lvls = tuple(base_mag/(tiles._l_z_downsamples[i]*tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in range(0,tiles.level_count))
    
    
                ######### THIS IS NEW ###########
                # previously we read in xml during assign_label, now we read it in ahead of time
                with open(os.path.join(self.json_location, json_file)) as f:
                    allobjects = geojson.load(f)
    
                allshapes = [shape(obj["geometry"]) for obj in allobjects]
                tumorshapes2 = unary_union(allshapes)
                # alllabels = [obj['properties'] for obj in allobjects]
                # roilabels = list()
                # tumorshapes = list()
                # for roi_num in range(0, len(alllabels)):
                #     try:
                #         # grab roi labels and names
                #         roi_label = alllabels[roi_num]['name']
                #         roilabels.append(roi_label)
                #         tumorshapes.append(allshapes[roi_num])
                #     except:
                #         pass
                # # I only had 1 class, so I just did a unary_union on all tumorshapes
                # print(np.unique(roilabels))
                # tumorshapes2 = unary_union(tumorshapes)
                # what you could do instead is write a dictionary of tumor shapes where
                #tumor_dict = {}
                #for labels in roilabels:
                #    tumor_dict[label] = unary_union(#only grab tumor rois from that label class)
    
                # pull tiles from levels specified by self.mag_extract
                for lvl in self.mag_extract:
                    if lvl in tile_lvls:
                        # pull tile info for level
                        x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(lvl)]
                        # print(str(x_tiles) + ' ' + str(y_tiles))
                        # note to self, we have to iterate b/c deepzoom does not allow casting all at once at list (??)
    
                        for y in range(0, y_tiles):
                            for x in range(0, x_tiles):
                                # if x == (x_tiles - 2):
                                    # print(str(y) + ' of ' + str(y_tiles))
                                # grab tile coordinates
                                tile_coords = tiles.get_tile_coordinates(tile_lvls.index(lvl), (x, y))
                                save_coords = str(tile_coords[0][0]) + "-" + str(tile_coords[0][1]) + "_" + '%.0f' % (
                                            tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][0]) + "-" + '%.0f' % (
                                                          tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][1])
    
                                # label tile based on xml region membership
                                ######### THIS IS NEW ###########
                                # i now pass the unary_union of tumor shapes to assign_label function
                                # you could loop over each class or change assign_label to do that
                                tile_ends = (
                                int(tile_coords[0][0] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][0]),
                                int(tile_coords[0][1] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][1]))
                                tile_starts = tile_coords[0]
                                tile_box = [tile_starts[0], tile_starts[1]], [tile_starts[0], tile_ends[1]], [
                                    tile_ends[0], tile_starts[1]], [
                                               tile_ends[0], tile_ends[1]]
                                tile_box = list(tile_box)
                                tile_box = MultiPoint(tile_box).convex_hull
    
                                if tile_box.intersects(tumorshapes2):
                                    tile_size = tiles.get_tile_dimensions(tile_lvls.index(lvl), (x, y))
                                    tile_pull = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                                    self.save_to_disk(tile_pull=tile_pull, save_coords=save_coords, lvl=lvl,
                                                      tile_label='NA', tile_size=tile_size, physSize=physSize)
    
                    else:
                        print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")
            # # except:
            # #     pass
        return

    def whitespace_check(self,im):
        bw = im.convert('L')
        bw = np.array(bw)
        bw = bw.astype('float')
        bw=bw/255
        prop_ws = (bw > 0.8).sum()/(bw>0).sum()
        return prop_ws

    def get_box(self,poly):
        poly_x = []
        poly_y = []
        if poly.type == 'Polygon':
            polyvals = poly.exterior.xy
            poly_x.append(min(polyvals[0]))
            poly_x.append(max(polyvals[0]))
            poly_y.append(min(polyvals[1]))
            poly_y.append(max(polyvals[1]))
        if poly.type == 'MultiPolygon':
            for roii in poly.geoms:
                polyvals = roii.exterior.xy
                poly_x.append(min(polyvals[0]))
                poly_x.append(max(polyvals[0]))
                poly_y.append(min(polyvals[1]))
                poly_y.append(max(polyvals[1]))
        xy_lim=[min(poly_x),max(poly_x),min(poly_y),max(poly_y)]
        return xy_lim

    def assign_label(self, tile_starts, tile_ends, path, roi):
        ''' calculates overlap of tile with xml regions and creates dictionary based on unique labels '''
        tile_label = {}
        tile_box = [tile_starts[0], tile_starts[1]], [tile_starts[0], tile_ends[1]], [tile_ends[0], tile_starts[1]], [
            tile_ends[0], tile_ends[1]]
        tile_box = list(tile_box)
        tile_box = MultiPoint(tile_box).convex_hull

        label = 'LN'
        # loop over every region associated with a given label, sum the overlap
        box_label = True  # initialize
        ov = 0  # initialize
        ov_mask = np.zeros((int(max(tile_box.exterior.xy[0]) - min(tile_box.exterior.xy[0])),
                            int(max(tile_box.exterior.xy[0]) - min(tile_box.exterior.xy[0]))), dtype=np.int)
        if tile_box.intersects(roi):
            box_label = True
            ov_reg = tile_box.intersection(roi)
            ov += ov_reg.area / tile_box.area

            if ov_reg.type == 'Polygon':
                reg_mask = self.poly2mask(
                    vertex_row_coords=[x - min(tile_box.exterior.xy[1]) for x in ov_reg.exterior.xy[1]],
                    vertex_col_coords=[x - min(tile_box.exterior.xy[0]) for x in ov_reg.exterior.xy[0]],
                    shape=(int(max(tile_box.exterior.xy[0]) - min(tile_box.exterior.xy[0])),
                           int(max(tile_box.exterior.xy[0]) - min(tile_box.exterior.xy[0]))))
                # print(reg_mask.shape)
                ov_mask += reg_mask
            elif ov_reg.type == 'MultiPolygon':
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

        return tile_label



    def poly2mask(self,vertex_row_coords, vertex_col_coords, shape):
        ''''''
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=np.int)
        mask[fill_row_coords, fill_col_coords] = 1
        return mask

    def save_to_disk(self, tile_pull, save_coords, lvl, tile_label, tile_size, physSize):
        #tile_mask = tile_label['tumor']
        #tile_mask = tile_mask.astype(np.bool)
        #mask_pull = Image.fromarray(tile_mask)
        # edge tiles will not be correct size (too small), so we reflect image data until correct size
        if tile_size[0]<physSize or tile_size[1]<physSize:
            # tile_pull = Image.fromarray(cv2.copyMakeBorder(np.array(tile_pull), 0, physSize - int(tile_size[1]), 0, physSize - int(tile_size[0]),cv2.BORDER_REFLECT))
            # mask_pull = Image.fromarray(cv2.copyMakeBorder(tile_mask, 0, physSize - int(tile_size[1]), 0,physSize - int(tile_size[0]), cv2.BORDER_REFLECT))
            return
        else:
            # check whitespace amount
            ws = self.whitespace_check(im=tile_pull)
            if ws < 0.95:
                #tumorperc = (np.array(mask_pull) == True).sum()/(mask_pull.size[0]*mask_pull.size[1])
                #mask_np = np.array(mask_pull)
                #mask_np = mask_np.astype('uint8')
                #mask_save = Image.fromarray(mask_np)
                tile_savename = self.save_name + "_" + str(lvl) + "_" \
                                + save_coords + "_" \
                                + "ws-" + '%.2f' % (ws) \
                                + "_normal"
                tile_pull = tile_pull.resize(size=(self.save_image_size, self.save_image_size), resample=Image.LANCZOS)
                #mask_save = mask_save.resize(size=(self.save_image_size, self.save_image_size), resample=Image.LANCZOS)
                tile_pull.save(os.path.join(self.save_location, tile_savename + ".png"))
                #mask_save.save(os.path.join(self.save_location, tile_savename + "_mask.png"))
                #print(tile_savename)
        return


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file_location')
    # parser.add_argument('--image_file')
    # parser.add_argument('--xml_file')
    # parser.add_argument('--save_name')
    # args = parser.parse_args()
    c = extractPatch()
    c.parseMeta_and_pullTiles()