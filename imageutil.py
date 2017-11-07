#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:07:47 2017

@author: ali
"""

import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np


wdir_trn1 = './dataset_train_rgb.zip/'
wfile_trn1 = 'train.yaml'
wdir_tst1 = './dataset_test_rgb.zip/'
wfile_tst1 = 'test.yaml'
wdir_trn2 = './dataset_additional_rgb/'
wfile_trn2 = 'additional_train.yaml'



def get_image_label(entry, approach=1):
    """
    returns the label for an image
    approach:
        if 0: the largest bounding box will determine the label
        if 1: the highest boundib box (average) will determine the label
    """
    
    max_label = 'other'
    
    if approach==0:
        max_area = 0
        for traffic_light in entry['boxes']:
            x_max = traffic_light['x_max']
            x_min = traffic_light['x_min']
            y_max = traffic_light['y_max']
            y_min = traffic_light['y_min']
            area = (x_max-x_min)*(y_max-y_min)
            if area > max_area:
                max_area = area
                max_label = traffic_light['label']
    
    if approach==1:
        max_height = 1000
        for traffic_light in entry['boxes']:
            y_max = traffic_light['y_max']
            y_min = traffic_light['y_min']
            y_ave = (y_min+y_max)/2
            if y_ave < max_height:
                max_height = y_ave
                max_label = traffic_light['label']
    
    if max_label.find('Green') != -1:
        return 0
    elif max_label.find('Yellow') != -1:
        return 1
    elif max_label.find('Red') != -1:
        return 2
    else:
        return 3
               


def gen_filenames_and_labels_from_yaml(yaml_dir, yaml_fname):
    """
    generates 2 arrays the first one contains the filenames, 
    the 2nd one contains labels
    """
    
    data = load_yaml(yaml_dir+yaml_fname)
    
    fnames = []
    labels =[]
    
    for entry in data:
        fnames.append(yaml_dir+entry['path'][2:])
        labels.append(get_image_label(entry, 1))
    
    return fnames, labels
    


def load_yaml(yaml_path, verbose=False):
    """
    loads a yaml file and returns the contents as a list of dictionaries
    yaml_path: path to the yaml file
    """
    data = yaml.safe_load(open(yaml_path))
      
      
    if verbose:
        print()
        print('dataset contains {} images'.format(len(data)))
        # all labels
        all_labels = {}
        all_labels['no_light'] = 0
        for entry in data:
            if len(entry['boxes'])==0:
                all_labels['no_light'] += 1
            else:    
                for traffic_light in entry['boxes']:
                    if traffic_light['label'] not in all_labels:
                        all_labels[traffic_light['label']] = 1
                    else:
                        all_labels[traffic_light['label']] += 1
        print(all_labels)
        # reduced labels
        reduced_labels = {}
        reduced_labels['no_light'] = 0
        reduced_labels['red'] = 0
        reduced_labels['yellow'] = 0
        reduced_labels['green'] = 0
        for entry in data:
            if len(entry['boxes'])==0:
                reduced_labels['no_light'] += 1
            else:    
                for traffic_light in entry['boxes']:
                    if traffic_light['label'].find('Red') != -1:
                        reduced_labels['red'] += 1
                    if traffic_light['label'].find('Green') != -1:
                        reduced_labels['green'] += 1
                    if traffic_light['label'].find('Yellow') != -1:
                        reduced_labels['yellow'] += 1
        print(reduced_labels)
        # number of conflicts
        # reduced labels
        num_conflicts = 0
        for entry in data:
            red_light = 0
            green_light = 0
            yellow_light = 0
            for traffic_light in entry['boxes']:
                if traffic_light['label'].find('Red') != -1:
                    red_light = 1
                if traffic_light['label'].find('Green') != -1:
                    green_light = 1
                if traffic_light['label'].find('Yellow') != -1:
                    yellow_light = 1
            if red_light + yellow_light + green_light > 1:
                num_conflicts += 1
                #print(entry)
                #print()
        print('number of entries with conflict: {}'.format(num_conflicts))
        print()
        
    return data
    

def get_trn_val_data(test_ratio=0.15):
    """
    helper function for constructing arrays of labels and image filenames and
    splitting them into training and validation sets
    """
      
    trn_fnames_1, trn_labels_1 = gen_filenames_and_labels_from_yaml(wdir_trn1, wfile_trn1)
    trn_fnames_2, trn_labels_2 = gen_filenames_and_labels_from_yaml(wdir_trn2, wfile_trn2)
    trn_fnames_all = trn_fnames_1 + trn_fnames_2
    trn_labels_all = trn_labels_1 + trn_labels_2
      
    # splitting the data into training and validation sets  
    X_train, X_val, y_train, y_val = train_test_split(trn_fnames_all,trn_labels_all,test_size=test_ratio,stratify=trn_labels_all)
    
    # defining the OneHotEncoder class, reshaping the label data and fitting/transforming to ohe format
    enc = OneHotEncoder(dtype=np.float32)
    y_trn_out = enc.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
    y_val_out = enc.fit_transform(np.array(y_val).reshape(-1,1)).toarray()
    
    return np.asarray(X_train), np.asarray(y_trn_out), np.asarray(X_val), np.asarray(y_val_out)



def get_tst_data():
    """
    helper function for constructing arrays of labels and image filenames 
    corresponding to the test data
    """
      
    tst_fnames, tst_labels = gen_filenames_and_labels_from_yaml(wdir_tst1, wfile_tst1)
      
    # defining the OneHotEncoder class, reshaping the label data and fitting/transforming to ohe format
    enc = OneHotEncoder(dtype=np.float32)
    y_tst_out = enc.fit_transform(np.array(tst_labels).reshape(-1,1)).toarray()
    
    return np.asarray(tst_fnames), np.asarray(y_tst_out)



if __name__ == '__main__':
    print('imageutil.py is loaded.')