import sys
import os
import numpy as np
import cv2
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import xml.etree.ElementTree as ET
from xml.dom import minidom
import geojson
import argparse
import matplotlib.pyplot as plt
import fastai
from fastai.vision.all import *
import PIL
matplotlib.use('Agg')
import pandas as pd
import datetime
from skimage import draw, measure, morphology, filters
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
from shapely.ops import cascaded_union, unary_union
import json
import shapely
import warnings
from scipy import ndimage
warnings.filterwarnings("ignore")

def get_y(row): return row['label'].split(" ")
     
class extractPatch:

    def __init__(self):
        self.save_location = cur_wd + '/data/cancer_output200and3/'  # args.save_location
        self.mag_extract = [20] # do not change this, model trained at 250x250 at 20x
        self.save_image_size = 250  # do not change this, model trained at 250x250 at 20x
        self.run_image_size = 250 # do not change this, model trained at 250x250 at 20x
        self.pixel_overlap = 200  # specify the level of pixel overlap in your saved images
        self.limit_bounds = True  # this is weird, dont change it
        self.tiff_lvl = 3 # low res pyramid level to grab
        self.model_path2 = cur_wd + '/code_s/cancer_detection/dlv3_2ep_2e4_update-07182023_RT.pkl'

    def parseMeta_and_pullTiles(self,flist):
        if not os.path.exists(os.path.join(self.save_location)):
            os.mkdir(os.path.join(self.save_location))

        # first load pytorch model
        learn = load_learner(self.model_path2,cpu=False) #Change from cpu = False, cuz CUDA version for pytorch needs to be updated

        for _file in flist:
            print(_file)
            oslide = openslide.OpenSlide(_file)
            savnm = os.path.basename(_file)
            save_name = str(Path(savnm).with_suffix(''))

            if not os.path.exists(os.path.join(self.save_location,save_name+'_tissue.json')):
                # this is physical microns per pixel
                acq_mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])

                # this is nearest multiple of 20 for base layer
                base_mag = int(20 * round(float(acq_mag) / 20))

                # this is how much we need to resample our physical patches for uniformity across studies
                physSize = round(self.save_image_size * acq_mag / base_mag)

                # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
                tiles = DeepZoomGenerator(oslide, tile_size=physSize - round(self.pixel_overlap * acq_mag / base_mag),
                                          overlap=round(self.pixel_overlap * acq_mag / base_mag / 2),
                                          limit_bounds=self.limit_bounds)

                # calculate the effective magnification at each level of tiles, determined from base magnification
                tile_lvls = tuple(base_mag / (tiles._l_z_downsamples[i] * tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in range(0, tiles.level_count))

                # intermeadiate level for probability map
                lvl_img = oslide.read_region((0, 0), self.tiff_lvl, oslide.level_dimensions[self.tiff_lvl])
                lvl_resize = oslide.level_downsamples[self.tiff_lvl]
                lvl_img.save(os.path.join(self.save_location,save_name+'_low-res.png'))

                # send to get tissue polygons
                print('detecting tissue')
                tissue, he_mask = self.do_mask(lvl_img,lvl_resize)

                # print(oslide.level_dimensions[1])
                x_map = np.zeros((lvl_img.size[1], lvl_img.size[0]), float)
                x_count = np.zeros((lvl_img.size[1], lvl_img.size[0]), float)

                print('starting inference')
                # pull tiles from levels specified by self.mag_extract
                for lvl in self.mag_extract:
                    if lvl in tile_lvls:
                        # pull tile info for level
                        x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(lvl)]

                        for y in range(0, y_tiles):
                            for x in range(0, x_tiles):

                                # grab tile coordinates
                                tile_coords = tiles.get_tile_coordinates(tile_lvls.index(lvl), (x, y))
                                save_coords = str(tile_coords[0][0]) + "-" + str(tile_coords[0][1]) + "_" + '%.0f' % (tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][0]) + "-" + '%.0f' % (tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][1])
                                tile_ends = (int(tile_coords[0][0] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][0]),int(tile_coords[0][1] + tiles._l0_l_downsamples[tile_coords[1]] * tile_coords[2][1]))

                                # check for tissue membership
                                tile_tiss = self.check_tissue(tile_starts=tile_coords[0], tile_ends=tile_ends,roi=tissue)
                                if tile_tiss > 0.9:
                                    tile_pull = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                                    tile_copy = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                                    ws = self.whitespace_check(im=tile_pull)
                                    if ws < 0.9:
                                        tile_pull = tile_pull.resize(size=(self.save_image_size, self.save_image_size),resample=PIL.Image.LANCZOS)
                                        #tile_copy = tile_copy.resize(size=(self.save_image_size, self.save_image_size),resample=PIL.Image.ANTIALIAS)
                                        #tile_pull = tile_pull.resize(size=(self.run_image_size, self.run_image_size),resample=PIL.Image.ANTIALIAS)
                                        #tile_pull = np.array(tile_pull)

                                        #classification
                                        # #tile_pull = subpatch.resize(size=(saveSz, saveSz), resample=PIL.Image.ANTIALIAS)
                                        # img = np.array(tile_pull)
                                        # pred_class, pred_idx, outputs = learn.predict(img)
                                        # outputs_np = outputs.numpy()

                                        #segmentation
                                        tile_pull = np.array(tile_pull)

                                        with learn.no_bar():
                                            pred_class, pred_idx, outputs = learn.predict(tile_pull[:, :, 0:3])
                                        outputs_np = outputs.numpy()
                                        output = PIL.Image.fromarray(outputs_np[1])

                                        #resize for low-res
                                        output = output.resize(size=(int(np.floor(tile_ends[1] / lvl_resize))-int(np.floor(tile_coords[0][1] / lvl_resize)),int(np.floor(tile_ends[0] / lvl_resize))-int(np.floor(tile_coords[0][0] / lvl_resize))),resample=PIL.Image.LANCZOS)

                                        output_np = np.array(output)
                                        
                                        #LUCAS
                                        #LUCAS return tile_coords, tile_ends, lvl_resize, x_count, x_map,output_np
                                        # inp, pred_class, pred_idx, outputs = learn_class.predict(tile_pull, with_input=True)
                                        # outputs_np = outputs.numpy()
                                        #print(outputs_np)
                                        #Add the following because it will give error for images that hard to detect the egdes example: OPX006,007,010...
                                        try: 
                                            x_count[int(np.floor(tile_coords[0][1] / lvl_resize)):int(np.floor(tile_ends[1] / lvl_resize)),int(np.floor(tile_coords[0][0] / lvl_resize)):int(np.floor(tile_ends[0] / lvl_resize))] += 1
                                            x_map[int(np.floor(tile_coords[0][1] / lvl_resize)):int(np.floor(tile_ends[1] / lvl_resize)),int(np.floor(tile_coords[0][0] / lvl_resize)):int(np.floor(tile_ends[0] / lvl_resize))] += output_np[1]
                                        except:
                                            pass
                    else:
                        print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")

                print('post-processing')
                x_count = np.where(x_count < 1, 1, x_count)
                x_map = x_map / x_count
                slideimg = PIL.Image.fromarray(np.uint8(x_map * 255))
                slideimg = slideimg.convert('L')
                # slideimg.save(os.path.join(self.save_location, save_name + '_cancer_prob.jpeg'))
                # lvl_img.save(os.path.join(self.save_location, save_name + '_lowres.tiff'))
                cmap = plt.get_cmap('jet')
                rgba_img = cmap(x_map)
                rgb_img = np.delete(rgba_img, 3, 2)
                colimg = PIL.Image.fromarray(np.uint8(rgb_img * 255))
                colimg.save(os.path.join(self.save_location, save_name + '_cancer_prob.jpeg'))
                binary_preds = self.cancer_mask(slideimg,he_mask)
                polygons = self.tile_ROIS(mask_arr=binary_preds, lvl_resize=lvl_resize)
                self.slide_ROIS(polygons=polygons, mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                                savename=os.path.join(self.save_location,save_name+'_cancer.json'), labels='AI_tumor', ref=[0,0], roi_color=-16711936)
                self.slide_ROIS(polygons=tissue, mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]),
                                savename=os.path.join(self.save_location,save_name+'_tissue.json'), labels='tissue', ref=[0,0], roi_color=-16770432)

        return

    def do_mask(self,img,lvl_resize):
        ''' create tissue mask '''
        # get he image and find tissue mask
        he = np.array(img)
        he = he[:, :, 0:3]
        heHSV = cv2.cvtColor(he, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(heHSV, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imagem = cv2.bitwise_not(thresh1)
        tissue_mask = morphology.binary_dilation(imagem, morphology.disk(radius=5))
        tissue_mask = morphology.remove_small_objects(tissue_mask, 1000)
        tissue_mask = ndimage.binary_fill_holes(tissue_mask)

        # create polygons for faster tiling in cancer detection step
        polygons = []
        contours, hier = cv2.findContours(tissue_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cvals = contour.transpose(0, 2, 1)
            cvals = np.reshape(cvals, (cvals.shape[0], 2))
            cvals = cvals.astype('float64')
            for i in range(len(cvals)):
                cvals[i][0] = np.round(cvals[i][0]*lvl_resize,2)
                cvals[i][1] = np.round(cvals[i][1]*lvl_resize,2)
            try:
                poly = Polygon(cvals)
                if poly.length > 0:
                    polygons.append(Polygon(poly.exterior))
            except:
                pass
        tissue = unary_union(polygons)
        while not tissue.is_valid:
            print('pred_union is invalid, buffering...')
            tissue = tissue.buffer(0)

        return tissue, tissue_mask

    def check_tissue(self, tile_starts, tile_ends, roi):
        ''' checks if tile in tissue '''
        tile_box = [tile_starts[0], tile_starts[1]], [tile_starts[0], tile_ends[1]], [tile_ends[0], tile_starts[1]], [tile_ends[0], tile_ends[1]]
        tile_box = list(tile_box)
        tile_box = MultiPoint(tile_box).convex_hull
        ov = 0  # initialize
        if tile_box.intersects(roi):
            ov_reg = tile_box.intersection(roi)
            ov += ov_reg.area / tile_box.area

        return ov

    def whitespace_check(self, im):
        ''' checks if meets whitespace requirement'''
        bw = im.convert('L')
        bw = np.array(bw)
        bw = bw.astype('float')
        bw = bw / 255
        prop_ws = (bw > 0.8).sum() / (bw > 0).sum()
        return prop_ws

    def cancer_mask(self,pred_image,hetissue):
        ''' smooth cancer map and find high probability areas '''
        # get pred image, mask at 50% and find regions
        preds = np.array(pred_image).astype('float64')
        preds = preds / 255
        preds[hetissue < 1] = 0
        preds = filters.gaussian(preds,sigma=10)
        preds_mask = np.zeros(preds.shape)
        preds_mask[preds > 0.3] = 1
        preds_mask = morphology.binary_dilation(preds_mask, morphology.disk(radius=2))
        preds_mask = morphology.binary_erosion(preds_mask, morphology.disk(radius=2))
        preds_mask = ndimage.binary_fill_holes(preds_mask)
        labels = measure.label(preds_mask)
        regions = measure.regionprops(labels,preds)
        for reg in regions:
            if reg.max_intensity<0.6:
                labels[labels==reg.label]=0
        labels[labels>0]=1
        return labels


    def slide_ROIS(self,polygons,mpp,savename,labels,ref,roi_color):
        ''' generate geojson from polygons '''
        all_polys = unary_union(polygons)
        final_polys = []
        if all_polys.type == 'Polygon':
            poly = all_polys
            polypoints = poly.exterior.xy
            polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
            polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
            newpoly = Polygon(zip(polyx, polyy))
            if newpoly.area*mpp*mpp > 0.1:
                final_polys.append(newpoly)

        else:
            for poly in all_polys.geoms:
                # print(poly)
                if poly.type == 'Polygon':
                    polypoints = poly.exterior.xy
                    polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
                    polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
                    newpoly = Polygon(zip(polyx, polyy))
                    if newpoly.area*mpp*mpp > 0.1:
                        final_polys.append(newpoly)
                if poly.type == 'MultiPolygon':
                    for roii in poly.geoms:
                        polypoints = roii.exterior.xy
                        polyx = [np.round(number - ref[0], 1) for number in polypoints[0]]
                        polyy = [np.round(number - ref[1], 1) for number in polypoints[1]]
                        newpoly = Polygon(zip(polyx, polyy))
                        if newpoly.area*mpp*mpp > 0.1:
                            final_polys.append(newpoly)

        final_shape = unary_union(final_polys)
        try:
            trythis = '['
            for i in range(0, len(final_shape)):
                trythis += json.dumps(
                    {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape[i]),
                    "properties": {"classification": {"name": labels, "colorRGB": roi_color}, "isLocked": False,
                                    "measurements": []}}, indent=4)
                if i < len(final_shape) - 1:
                    trythis += ','
            trythis += ']'
        except:
            trythis = '['
            trythis += json.dumps(
                {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape),
                "properties": {"classification": {"name": labels, "colorRGB": roi_color}, "isLocked": False,
                                "measurements": []}}, indent=4)
            trythis += ']'

        with open(savename, 'w') as outfile:
            outfile.write(trythis)
        return

    def tile_ROIS(self,mask_arr,lvl_resize):
        ''' get cancer polygons '''
        polygons = []
        contours, hier = cv2.findContours(mask_arr.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cvals = contour.transpose(0, 2, 1)
            cvals = np.reshape(cvals, (cvals.shape[0], 2))
            cvals = cvals.astype('float64')
            for i in range(len(cvals)):
                cvals[i][0] = np.round(cvals[i][0]*lvl_resize,2)
                cvals[i][1] = np.round(cvals[i][1]*lvl_resize,2)
            try:
                poly = Polygon(cvals)
                if poly.length > 0:
                    polygons.append(Polygon(poly.exterior))
            except:
                pass

        return polygons

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--by_folder')
    # parser.add_argument('--by_csv')
    # parser.add_argument('--single_image')
    # parser.add_argument('--save_location')
    # args = parser.parse_args()
    # print(args)
    os.chdir('../..')           #change working directory two level back
    cur_wd = os.getcwd()
    flist=sorted(glob.glob(cur_wd + '/data/OPX/*.tif'))
    
    #LUCAS Already prossessed files
    prossessed_flist=sorted(glob.glob(cur_wd + '/data/cancer_output200and3/*png'))
    prossessed_flist = [re.sub(cur_wd + '/data/cancer_output200and3/', "", x) for x in prossessed_flist]
    prossessed_flist = [re.sub('_low-res.png', "", x) for x in prossessed_flist]
    prossessed_flist = [cur_wd + '/data/OPX/' + x + '.tif' for x in prossessed_flist]
    #prossessed_flist = prossessed_flist + [cur_wd + '/data/OPX/OPX_006.tif', cur_wd + '/data/OPX/OPX_007.tif', cur_wd + '/data/OPX/OPX_010.tif'] #Problemset: 006,007, 010

    #Not propossed files
    nprossessed_flist = [x for x in flist if x not in prossessed_flist]
    c = extractPatch()
    c.parseMeta_and_pullTiles(flist=nprossessed_flist)
    #LUCAS
    #p_file = nprossessed_flist[0:1]
    #p_file = [cur_wd + '/data/OPX/OPX_007.tif']
    #print(p_file)
    #tile_coords, tile_ends, lvl_resize, x_count, x_map, output_np = c.parseMeta_and_pullTiles(flist=p_file)
    #x_count[int(np.floor(tile_coords[0][1] / lvl_resize)):int(np.floor(tile_ends[1] / lvl_resize)),int(np.floor(tile_coords[0][0] / lvl_resize)):int(np.floor(tile_ends[0] / lvl_resize))] += 1
    #x_map[int(np.floor(tile_coords[0][1] / lvl_resize)):int(np.floor(tile_ends[1] / lvl_resize)),int(np.floor(tile_coords[0][0] / lvl_resize)):int(np.floor(tile_ends[0] / lvl_resize))] += output_np[1]
