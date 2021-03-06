#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:16:15 2020

@author: sohini roychowdhury
MIT Copyright enforced
"""
from main_fundus_integration import *
from prepare_datasets_DRIVE_integral import *
from u_net_with_patches_train import *
from U_net_predict_mod import *


#Step 1: Image augmentation (zoom/pan using Keras)
data_aug_keras()
#Step 2: Data preparation, N=total number of training images generated after keras augmentation in previous step
prep_data(N=480)
#Step 3: Run u-net #Modify unet/gnet in line number 186 in file u_net_with_patches_train.py
run_non_olap_pathes()
#Step 4:test patches
test_u_net()
