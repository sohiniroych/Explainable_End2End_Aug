#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:30:45 2020

@author: schowdhu

"""
import os
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
#Preprocess DRIVE to fit U-net format
img_path='/home/local/VCCNET/schowdhu/ML_test/unet-master/fundus/DRIVE/test/images'
GT_path='/home/local/VCCNET/schowdhu/ML_test/unet-master/fundus/DRIVE/test/GT'
image_name_arr = os.listdir(img_path)
GT_name_arr=os.listdir(GT_path)
        #rename the GT image same as imagename
for idx,item in enumerate(image_name_arr):
        sub=item[0:2]
        print(sub)
        indx=[s for s in GT_name_arr if sub in s]  
        listToStr = ' '.join(map(str, indx)) 
        GT=io.imread(os.path.join(GT_path,listToStr))
        io.imsave(os.path.join(GT_path, item),GT) 
