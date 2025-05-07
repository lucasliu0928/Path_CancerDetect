#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 16:56:55 2025

@author: jliu6
"""

import shutil
import os

proj_dir = '/fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/'
data_folder = proj_dir + '/intermediate_data/4_tile_feature/Neptune/IMSIZE250_OL100/'

folders_to_remove = ['NEP-053PS1-1_HE_MH_03252024',
 'NEP-054PS1_HE_MH_03252024',
 'NEP-108PS2-1_HE_MH_05220224',
 'NEP-167PS1-1_HE_MH_06102024',
 'NEP-169PS1-1_HE_MH_06102024',
 'NEP-171PS1-1_HE_MH_06102024',
 'NEP-172PS1-1_HE_MH_06102024',
 'NEP-173PS5-1_HE_MH_06242024',
 'NEP-175PS1-1_HE_MH_06242024',
 'NEP-177PS1-1_HE_MH_06242024',
 'NEP-178PS1-1_HE_MH_06242024',
 'NEP-189PS1-1_HE_MH_06242024',
 'NEP-190PS2-1_HE_MH_06242024',
 'NEP-194PS1-1_HE_MH_06242024',
 'NEP-197PS1-1_HE_MH_06242024',
 'NEP-235PS1-1_HE_MH_02052025',
 'NEP-244PS2-1_HE_MH_02052025',
 'NEP-284PS1-1_HE_MH_02072025',
 'NEP-286PS1-1_HE_MH_02072025',
 'NEP-310PS1-1_HE_MH_03202025',
 'NEP-315PS1-1_HE_MH_03202025',
 'NEP-316PS1-1_HE_MH_03202025',
 'NEP-316PS2-1_HE_MH_03202025',
 'NEP-317PS1-1_HE_MH_03202025',
 'NEP-325PS1-1_HE_MH_03202025',
 'NEP-328PS1-1_HE_MH_03202025',
 'NEP-332PS1-1_HE_MH_03212025',
 'NEP-333PS1-1_HE_MH_03202025',
 'NEP-336PS1-1_HE_MH_03202025',
 'NEP-338PS2-1_HE_MH_03202025']

ct = 0
for r_file in folders_to_remove:
    f_to_remove = data_folder + r_file
    if os.path.isdir(f_to_remove):
        shutil.rmtree(f_to_remove)
        ct += 1
print(ct)