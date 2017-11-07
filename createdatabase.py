# -*- coding: utf-8 -*-
"""
Created on Mon Nov 6 2017
@author: ali.khalili
"""


import pandas as pd
import cv2
import numpy as np
import yaml
import glob
import os
import scipy


yaml_dirs = ['./dataset_additional_rgb/', './dataset_train_rgb.zip/']
yaml_files = ['additional_train.yaml', 'train.yaml']

target_dir = './trn_dataset/'

all_data = {}
all_data['green'] = []
all_data['red'] = []
all_data['yellow'] = []
all_data['other'] = []

all_labels = {}

target_size = (64, 64)

def add_data_from_yaml(yaml_dir, yaml_fname):
    '''
    adds to the global dictionary with characteristics of bounding boxes
    '''
    
    global all_data
    data = yaml.safe_load(open(yaml_dir+yaml_fname))
    
    # add to the global dictionary
    for entry in data:
        for traffic_light in entry['boxes']:
            if traffic_light['label'].find('Green') != -1:
                all_data['green'].append({
                        'x_min': traffic_light['x_min'],
                        'x_max': traffic_light['x_max'],
                        'y_min': traffic_light['y_min'],
                        'y_max': traffic_light['y_max'],
                        'path': yaml_dir+entry['path'][2:]})
            elif traffic_light['label'].find('Red') != -1:
                all_data['red'].append({
                        'x_min': traffic_light['x_min'],
                        'x_max': traffic_light['x_max'],
                        'y_min': traffic_light['y_min'],
                        'y_max': traffic_light['y_max'],
                        'path': yaml_dir+entry['path'][2:]})  
            elif traffic_light['label'].find('Yellow') != -1:
                all_data['yellow'].append({
                        'x_min': traffic_light['x_min'],
                        'x_max': traffic_light['x_max'],
                        'y_min': traffic_light['y_min'],
                        'y_max': traffic_light['y_max'],
                        'path': yaml_dir+entry['path'][2:]})  
            else:
                all_data['other'].append({
                        'x_min': traffic_light['x_min'],
                        'x_max': traffic_light['x_max'],
                        'y_min': traffic_light['y_min'],
                        'y_max': traffic_light['y_max'],
                        'path': yaml_dir+entry['path'][2:]}) 
    
    return data


def delete_current_datasets(verbose=True):
    '''
    Delete all files in the current dataset of vehicle and non-vehicle images
    '''
    
    # Verbose mode
    if verbose:
        print('Deleting the current contents of the database.')
        
    # Make a list of all files in the vehicle dataset folder and delete each file
    file_names = glob.glob(target_dir+'*.*')
    for file_name in file_names:
        os.remove(file_name)
        


def build_data_sets(min_records=100, add_variations=False, verbose=False):
    '''
    adds all images to the dataset
    '''
    global all_labels
    
    max_height = 0
    max_width = 0
    
    for status in all_data.keys():
        counter = 0
        for i in range(min_records):
            # randomly select a box
            rand_index = np.random.randint(0, len(all_data[status]))
            box = all_data[status][rand_index]
            # load image
            path = box['path']
            img = scipy.misc.imread(path)
            # get the box dimensions
            x_min = max(round(box['x_min']),0)
            x_max = min(round(box['x_max']),img.shape[1]) 
            y_min = max(round(box['y_min']),0)
            y_max = min(round(box['y_max']),img.shape[0])
            cur_width = y_max-y_min
            cur_height = x_max-x_min
            # verbose
            if verbose: 
                if max_width < cur_width:
                    max_width = cur_width
                if max_height < cur_height:
                    max_height = cur_height
                print('No: {} - height: {} - width: {} - path: {}'.format(counter, x_max - x_min, y_max - y_min, path))
            # get and save image in the box
            if cur_height > 0 and cur_width > 0:
                img_box = img[y_min:y_max, x_min:x_max, :]
                fname = target_dir+status+str(counter)+'.png'
                scipy.misc.imsave(fname, img_box)
                all_labels[fname] = status
                counter += 1
            
    if verbose:
        print(max_height, ' ', max_width)        
        

def main():
    
    for i in range(len(yaml_dirs)):
        add_data_from_yaml(yaml_dirs[i], yaml_files[i])
        

if __name__ == '__main__':
    main()


