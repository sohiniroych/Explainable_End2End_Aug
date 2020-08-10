
###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
Refarence: https://github.com/orobix/retina-unet
##################################################


import numpy as np
from six.moves import configparser as ConfigParser
from keras import layers




from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD

import sys
sys.path.insert(0, './lib/')
from help_funs import *

#function to obtain data for training/testing (validation)
from extract_patches_mod import get_data_training



#Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)
    conv3 = BatchNormalization()(conv3)
    drop3 = Dropout(0.5)(conv3)
    

    up1 = UpSampling2D(size=(2, 2))(drop3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    #conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    #conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

#Define the neural network gnet
#you need change function call "get_unet" to "get_gnet" in line 166 before use this network
#change this with other code base, there is error here!
def get_gnet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    
    conv1 = Conv2D(32, (3,3), activation = 'relu', padding = 'same', data_format='channels_first')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3,3), activation = 'relu', padding = 'same', data_format='channels_first')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    
    
    conv2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', data_format='channels_first')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', data_format='channels_first')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', data_format='channels_first')(pool2)
    conv3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', data_format='channels_first')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    
    conv4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', data_format='channels_first')(pool3)
    conv4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', data_format='channels_first')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(drop4)

    conv5 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', data_format='channels_first')(pool4)
    conv5 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', data_format='channels_first')(conv5)
    drop5 = Dropout(0.5)(conv5)
    

    up6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', data_format='channels_first')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([conv4,up6], axis = 1)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', data_format='channels_first')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', data_format='channels_first')(conv6)

    up7 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format='channels_first')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 1)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format='channels_first')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format='channels_first')(conv7)

    up8 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format='channels_first')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 1)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format='channels_first')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format='channels_first')(conv8)

    up9 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_first')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 1)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_first')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format='channels_first')(conv9)
    
    conv10 = Conv2D(2, (1,1), activation='relu', padding = 'same', data_format='channels_first')(conv9)
    print(conv10._keras_shape)
    conv10 = core.Reshape((2,patch_height*patch_width))(conv10)
    conv10 = core.Permute((2,1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

#========= Load settings from Config file
def run_non_olap_pathes():
    config = ConfigParser.RawConfigParser()
    config.read('configuration.txt')
    #patch to the datasets
    path_data = config.get('data paths', 'path_local')
    #Experiment name
    name_experiment = config.get('experiment name', 'name')
    #training settings
    N_epochs = int(config.get('training settings', 'N_epochs'))
    batch_size = int(config.get('training settings', 'batch_size'))
    
    
    
    #============ Load the data and divided in patches
    patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
        DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
        DRIVE_train_borders = path_data + config.get('data paths', 'train_border_masks'),
        patch_height = int(config.get('data attributes', 'patch_height')),
        patch_width = int(config.get('data attributes', 'patch_width')),
        N_subimgs = int(config.get('training settings', 'N_subimgs')),
        inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
    )
    
    
    #========= Save a sample of what you're feeding to the neural network ==========
    N_sample = min(patches_imgs_train.shape[0],40)
    visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs").show()
    visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks").show()
    
    
    #=========== Construct and save the model arcitecture =====
    n_ch = patches_imgs_train.shape[1]
    patch_height = patches_imgs_train.shape[2]
    patch_width = patches_imgs_train.shape[3]
    model = get_unet(n_ch, patch_height, patch_width)  #the U-net model (Modify here for gnet)
    model.summary()
    print("Check: final output of the network:")
    print(model.output_shape)
    plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
    json_string = model.to_json()
    open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)
    
    
    
    #============  Training ==================================
    checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
    patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
    model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])
    
    
    #========== Save and test the last model ===================
    model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)



















#
