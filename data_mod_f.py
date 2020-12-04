from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
import scipy.misc as sc
from evaluation import *


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img /= 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
       
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
            
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
        
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        
        #print(np.shape(mask),np.shape(img))
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale", image_save_prefix  = "images",
                    mask_save_prefix  = "GT",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
      
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir+image_save_prefix,
        #save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir+mask_save_prefix,
        #save_prefix  = mask_save_prefix,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
         img,mask= adjustData(img,mask,flag_multi_class,num_class)
         yield (img,mask)




def Img2maskgenerator(img_path,mask_path):
    files=os.listdir(img_path)
    num_image=len(files)
    for i in range(num_image):
        img = io.imread(os.path.join(img_path,files[i]))
        im= img[:,:,1]/255
        img1=np.zeros(np.shape(im))
        
                      #print(np.max(img))
        #print(np.min(img))
        img1[im>=0.1]=255
        img1[im<0.1]=0
        img2=np.zeros(np.shape(img))
        img2[:,:,0]=img1
        img2[:,:,1]=img1
        img2[:,:,2]=img1
        print(np.shape(img2))
        plt.imsave(os.path.join(mask_path,files[i]),img2)
        #plt.imshow(img1)





def testGenerator(test_path,num_image = 30,target_size = (512,512),flag_multi_class = False,as_gray = True):
    files=os.listdir(test_path)
    num_image=len(files)
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,files[i]),as_gray = as_gray)
#        img= img / 255 #try removing 255 here
        img = trans.resize(img,target_size)
#        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
#        img = np.reshape(img,(1,)+img.shape)
        #print(np.shape(img))
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = os.listdir(image_path)
    image_arr = []
    mask_arr = []
    target_size=(512,512,3)
    for index,item in enumerate(image_name_arr):
        img = io.imread(os.path.join(image_path,item),as_gray = image_as_gray)
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(os.path.join(mask_path,item).replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        img = trans.resize(mask,target_size)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
       # plt.imshow(img[:,:,0]+mask[:,:,0])
        #print(np.shape(img))
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i] = color_dict[i]
        img_out / 255
        #img_out[img_out>0.3]=1
        #img_out[img_out<=0.3]=0
    return img_out


#def find_Dice_Jac_sen_spec_acc(results,GT_path):
#    files=os.listdir(GT_path)
#    for i,item in enumerate()


def saveResult(img_path,save_path,npyfile,flag_multi_class = False,num_class = 2):
    files=os.listdir(img_path)
    #print(len(img_path))
    #print(len(npyfile))
    
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #img[img>0.2]=1
        #img[img<=0.2]=0
        io.imsave(os.path.join(save_path, files[i]+'_predict_1.png'),img)
        
        
        
def saveResult_1(img_path,save_path,npyfile,flag_multi_class = False,num_class = 2):
    files=os.listdir(img_path)
    #print(len(img_path))
    #print(len(npyfile))
    
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        img[img>0.2]=1
        img[img<=0.2]=0
        io.imsave(os.path.join(save_path, files[i]+'_predict_2.png'),img)        
        
def saveResult_2(img_path,save_path,npyfile,flag_multi_class = False,num_class = 2):
    files=os.listdir(img_path)
    #print(len(img_path))
    #print(len(npyfile))
    
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        img[img>0.2]=1
        img[img<=0.2]=0
        io.imsave(os.path.join(save_path, files[i]+'_predict_3.png'),img)        

def evalResult(gth_path,npyfile,flag_multi_class = False,num_class = 2):
    files=os.listdir(gth_path)
    print(files)
    prec=0
    rec=0
    acc=0
    IoU=0
    f1_score=0
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        #print(files[i])
        gth = io.imread(os.path.join(gth_path,files[i]))
        gth=sc.imresize(gth,(512,512),interp='nearest')
        img1=np.array(((img - np.min(img))/np.ptp(img))>0.5).astype(float)
        gth1=np.array(((gth - np.min(gth))/np.ptp(gth))>0.5).astype(float)
        #sc.imshow(img1)
        #sc.imshow(gth1)
        p,r,I,a,f=get_validation_metrics(gth1,img1)
        prec=prec+p
        rec=rec+r
        acc=acc+a
        IoU=IoU+I
        f1_score=f1_score+f
    print("Precision=",prec/(i+1), "Recall=",rec/(i+1), "IoU=",IoU/(i+1), "acc=",acc/(i+1), "F1=",f1_score/(i+1))
