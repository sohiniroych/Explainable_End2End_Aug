#Created by Sohini Roychowdhury
#Copyright included
from model_bf import *
from data_mod_f import *
import os


def data_aug_keras():

    data_gen_args = dict(rotation_range=0.3,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.1,
                        zoom_range=[0.5,1],
                        horizontal_flip=True,
                        fill_mode='nearest')
    vendor='DRIVE'
    #create a foder for training data augmentaion and storing predictions
    if not os.path.exists('fundus/'+vendor+'/training/aug'):
        os.makedirs('fundus/'+vendor+'/training/aug')
        
    myGene = trainGenerator(3,'fundus/'+vendor+'/training','images','GT',data_gen_args,save_to_dir ='fundus/'+vendor+'/training/aug/',image_color_mode="rgb")
        
    
    model = unet()
    model_checkpoint = ModelCheckpoint('unet_drive_color_merge.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=5,epochs=30,callbacks=[model_checkpoint])
    model.summary()
    
    Img2maskgenerator('fundus/'+vendor+'/training/aug/images/','fundus/'+vendor+'/training/aug/mask/')

