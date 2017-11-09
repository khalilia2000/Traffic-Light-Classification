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
from sklearn.preprocessing import OneHotEncoder


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
    loaded_images = np.array(loaded_images)
    
    # creating image dataset and pre-processing the images
    img_data_set = ImgDataSet(loaded_images, labels, norm_max_min=img_norm, scaled_dim=(img_rows,img_cols))
    img_data_set.normalize()
    
    # chaning labels to numerical values
    labels_num = [x.replace('green', '0') for x in labels]
    labels_num = [x.replace('yellow', '1') for x in labels_num]
    labels_num = [x.replace('red', '2') for x in labels_num]
    labels_num = np.asarray([x.replace('other', '3') for x in labels_num])
    labels_num = labels_num.astype(np.float32)
    
    # defining the OneHotEncoder class, reshaping the label data and fitting/transforming to ohe format
    enc = OneHotEncoder(dtype=np.float32)
    y_out = enc.fit_transform(np.array(labels_num).reshape(-1,1)).toarray()
    
    return img_data_set.images, y_out


def get_model_2():
  
    # image sizes after pre-processing  
    global img_rows
    global img_cols
    global img_ch  
  
    model = Sequential()
    
    # Define the model architecture here
    k_i = TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
    
    # Convolution #1
    model.add(Conv2D(32, (5,5), kernel_initializer=k_i, input_shape=(img_rows, img_cols, img_ch)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    
    # Convolution #2
    model.add(Conv2D(64, (5,5), kernel_initializer=k_i, input_shape=(img_rows, img_cols, img_ch)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    
    # Convolution #3
    model.add(Conv2D(128, (4,4), kernel_initializer=k_i, input_shape=(img_rows, img_cols, img_ch)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    
    # Dropout
    model.add(Dropout(0.5))
    
    # Flatten - #3200 variables
    model.add(Flatten())
    
    # Fully connected 1
    model.add(Dense(1024))
    # Fully connected 2
    model.add(Dense(256))
    # Fully connected 3
    model.add(Dense(64))
    # Fully connected 4
    model.add(Dense(16))
    # Fully connected 5
    model.add(Dense(4))
    
    # Final activation
    model.add(Activation('softmax'))
    
    #using default hyper parameters when creating new network
    Adam_Optimizer = Adam(lr=0.0005)
    model.compile(optimizer=Adam_Optimizer, loss='categorical_crossentropy', metrics=['accuracy']) 
    
    return model


def build_model_and_train(epochs=10, b_size=64):
    """
    builds a model with random weights and trains the model
    X_trn, X_val: ndarray of file names pertaining to training and validation datasets
    y_trn, y_val: ndarray of steering angles for training and validation datasets.
    epochs: number of epochs to train the model
    b_size: batch size
    """
    X_trn, y_trn = load_and_pre_process_dataset(0)
    X_val, y_val = load_and_pre_process_dataset(1)
    
    # create the model
    model = get_model_2()
    
    # train the model
    model.fit(X_trn, y_trn, batch_size=b_size, epochs=epochs, validation_data=(X_val, y_val))
    
    return model



def main():
    pass


if __name__ == '__main__':
    main()