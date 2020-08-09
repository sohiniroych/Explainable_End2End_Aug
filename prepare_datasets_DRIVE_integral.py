#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt




def prep_data(N_train):   
    
    #------------Path of the images --------------------------------------------------------------
    #train
    original_imgs_train = "/fundus/DRIVE/training/aug/images/"
    groundTruth_imgs_train = "/fundus/DRIVE/training/aug/GT/"
    borderMasks_imgs_train = "/fundus/DRIVE/training/aug/mask/"
    #test
    original_imgs_test = "/fundus/DRIVE/test/images/"
    groundTruth_imgs_test = "/fundus/DRIVE/test/GT/"
    borderMasks_imgs_test = "/fundus/DRIVE/test/mask/"
    #---------------------------------------------------------------------------------------------
    
    #N_train = 32# <= enter the number of total training images here
    N_test=20
    channels = 3
    train_height = 512
    train_width = 512
    test_height=584
    test_width=565
    dataset_path = "./DRIVE_datasets_training_testing_int/"
    #-----------------------------------------------------------------------------------
    
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    #getting the training datasets
    pp=os.getcwd()
    #print(pp)
    imgs_train, groundTruth_train, border_masks_train = get_datasets(N_train,train_height,train_width,pp+original_imgs_train,pp+groundTruth_imgs_train,pp+borderMasks_imgs_train,"train")
    print("saving train datasets")
    write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
    write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")
    write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")
    
    #getting the testing datasets
    imgs_test, groundTruth_test, border_masks_test = get_datasets(N_test,test_height, test_width,pp+original_imgs_test,pp+groundTruth_imgs_test,pp+borderMasks_imgs_test,"test")
    print("saving test datasets")
    write_hdf5(imgs_test,dataset_path + "DRIVE_dataset_imgs_test.hdf5")
    write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test.hdf5")
    write_hdf5(border_masks_test,dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

    
def get_datasets(Nimgs,height,width,imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
        channels=3
        imgs = np.empty((Nimgs,height,width,channels))
        groundTruth = np.empty((Nimgs,height,width))
        border_masks = np.empty((Nimgs,height,width))
        
        for path, subdirs, files in os.walk(imgs_dir): 
            
            
                
          #list all files, directories in the path
            for i in range(len(files)):
                #original
                
                print("original image: " +files[i])
                img = Image.open(imgs_dir+files[i])
                imgs[i] = np.asarray(img)
                #corresponding ground truth
                
                groundTruth_name = files[i]
                print("ground truth name: " + groundTruth_name)
                g_truth = Image.open(groundTruth_dir + groundTruth_name)
                groundTruth[i] = np.asarray(g_truth)
                #corresponding border masks
                border_masks_name = ""
                if train_test=="train":
                    border_masks_name = files[i]
                    b_mask_1 = np.array(Image.open(borderMasks_dir + border_masks_name))
                    b_mask=np.reshape(b_mask_1[:,:,1],(512,512))
                    border_masks[i] = np.asarray(b_mask)
                elif train_test=="test":
                    border_masks_name = files[i]
                    b_mask = Image.open(borderMasks_dir + border_masks_name)
                    border_masks[i] = np.asarray(b_mask)
                else:
                    print("specify if train or test!!")
                    exit()
                
                print("border masks name: " + border_masks_name)
                
    
        print("imgs max: " +str(np.max(imgs)))
        print("imgs min: " +str(np.min(imgs)))
        #assert(np.max(groundTruth)==255 and np.max(border_masks)==1)
        #assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
        print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
        #reshaping for my standard tensors
        imgs = np.transpose(imgs,(0,3,1,2))
        assert(imgs.shape == (Nimgs,channels,height,width))
        groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
        border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
        assert(groundTruth.shape == (Nimgs,1,height,width))
        assert(border_masks.shape == (Nimgs,1,height,width))
        return imgs, groundTruth, border_masks   
    

