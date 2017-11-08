#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:29:05 2017

@author: ali
"""

import pandas as pd
import numpy as np
import scipy
from imagedataset import ImgDataSet
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal


dataset_dirs = ['./trn_dataset/', './val_dataset/', './tst_dataset/']
# target image sizes after pre-processing  
img_rows = 64
img_cols = 64
img_ch= 3
img_norm = 0.5 # max/min of normalized pixel values


def load_and_pre_process_dataset(index):
    '''
    load and pre-process the datasets
    '''
    # load the dataset
    df1 = pd.read_excel(dataset_dirs[index]+'labels.xlsx', 'Sheet1')
    labels = np.asarray(df1[0].tolist())
    fnames = df1.index.tolist()
    
    # load image data into memory  
    loaded_images = []
    num_loaded = len(fnames)
    for i in range(num_loaded): 
      loaded_images.append(scipy.misc.imread(fnames[i]))
    loaded_images = np.asarray(loaded_images)
    
    # creating image dataset and pre-processing the images
    img_data_set = ImgDataSet(loaded_images, labels, norm_max_min=img_norm, scaled_dim=(img_rows,img_cols))
    img_data_set.normalize()
    
    return img_data_set.images, loaded_images, labels


def get_model_2():
  
    # image sizes after pre-processing  
    global img_rows
    global img_cols
    global img_ch  
  
    model = Sequential()
    
    # Define the model architecture here
    
    #
    
    return model


def main():
    pass


if __name__ == '__main__':
    main()