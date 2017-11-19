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
import json
import sys


work_dir = './'
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


def get_model_2(l_rate):
  
    # image sizes after pre-processing  
    global img_rows
    global img_cols
    global img_ch  
  
    model = Sequential()
    
    # Define the model architecture here
    k_i = TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
    
    # Convolution #1 - output: 30x30x64
    model.add(Conv2D(64, (5,5), kernel_initializer=k_i, input_shape=(img_rows, img_cols, img_ch)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    
    # Convolution #2 - output: 14x14x128
    model.add(Conv2D(128, (3,3), kernel_initializer=k_i, input_shape=(img_rows, img_cols, img_ch)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    
    # Convolution #3 - output: 6x6x256
    model.add(Conv2D(256, (3,3), kernel_initializer=k_i, input_shape=(img_rows, img_cols, img_ch)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    
    # Dropout
    model.add(Dropout(0.5))
    
    # Flatten - #3200 variables
    model.add(Flatten())
    
    # Fully connected 1
    model.add(Dense(512))
    # Fully connected 2
    model.add(Dense(128))
    # Fully connected 3
    model.add(Dense(64))
    # Fully connected 4
    model.add(Dense(4))
    
    
    # Final activation
    model.add(Activation('softmax'))
    
    #using default hyper parameters when creating new network
    Adam_Optimizer = Adam(lr=l_rate)
    model.compile(optimizer=Adam_Optimizer, loss='categorical_crossentropy', metrics=['accuracy']) 
    
    return model



def save_model_and_weights(model):
    """
    save the model structure and the weights to model.json and model.h5 respectively.
    """
    global work_dir
    # saving the model and its weights
    print()
    print('Saving model and weights...')
    model.save_weights(work_dir+'model-sim.h5')
    with open(work_dir+'model-sim.json','w') as outfile:
      json.dump(model.to_json(), outfile)
    print('Done.')
    pass



def load_model_and_weights(model_file, l_rate):
    # load the model
    with open(model_file, 'r') as jfile:
      json_str = json.loads(jfile.read())
      model = model_from_json(json_str)
    
    #use default hyper parameters when creating new network
    Adam_Optimizer = Adam(lr=l_rate)
    model.compile(optimizer=Adam_Optimizer, loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # load weights
    weights_file = model_file.replace('json', 'h5')
    model.load_weights(weights_file)
    
    return model



def build_model_and_train(epochs=10, b_size=64, learning_rate=0.0005):
    """
    build a model with random weights and trains the model
    epochs: number of epochs to train the model
    b_size: batch size
    """
    
    # load trn and val datasets
    X_trn, y_trn = load_and_pre_process_dataset(0)
    X_val, y_val = load_and_pre_process_dataset(1)
    
    # create the model
    model = get_model_2(l_rate=learning_rate)
    
    # train the model
    model.fit(X_trn, y_trn, batch_size=b_size, epochs=epochs, validation_data=(X_val, y_val))
    
    # save model and weights
    save_model_and_weights(model)
    
    return model



def load_model_and_train(epochs=10, b_size=64, learning_rate=0.0005):
    '''
    loads the model and weights that were previously saved, and
    continues training the model for the number of epochs specified.
    epochs: number of epochs to train the model
    b_size: batch size
    '''
    # load trn and val datasets
    X_trn, y_trn = load_and_pre_process_dataset(0)
    X_val, y_val = load_and_pre_process_dataset(1)
     
    # load model
    model = load_model_and_weights(work_dir+'model-sim.json', l_rate=learning_rate)
    
    # train the model
    model.fit(X_trn, y_trn, batch_size=b_size, epochs=epochs, validation_data=(X_val, y_val))
    
    # save model and weights
    save_model_and_weights(model)  
      
    return model


def evaluate_test():
    '''
    evaluate the saved model on test dataset
    '''
    X_tst, y_tst = load_and_pre_process_dataset(2)
    model = load_model_and_weights(work_dir+'model-sim.json', l_rate=0.00005)
    metrics = model.evaluate(X_tst, y_tst)
    
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))



def main():
    print(sys.argv)
    if len(sys.argv)<2:
        print('Syntax Error: ')
        print('Use: python model-3.py num_epochs learning_rate [reset]')
        print('[reset] is optional and will result in re-initialization of the CNN.')
    else:
        if len(sys.argv)<3 and 'evaluate' in sys.argv:
            print('evaluating test dataset ...')
            evaluate_test()
        elif 'reset' in sys.argv:
            print('re-initializing the network ...')
            n_epochs = int(sys.argv[1])
            l_rate = float(sys.argv[2])
            build_model_and_train(epochs=n_epochs, b_size=64, learning_rate=l_rate)
        else:
            print('loading and training the network ...')
            n_epochs = int(sys.argv[1])
            l_rate = float(sys.argv[2])
            load_model_and_train(epochs=n_epochs, b_size=64, learning_rate=l_rate)

    pass


if __name__ == '__main__':
    main()