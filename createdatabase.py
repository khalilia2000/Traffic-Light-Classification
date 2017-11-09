# -*- coding: utf-8 -*-
"""
Created on Mon Nov 6 2017
@author: ali.khalili
"""


import pandas as pd
import numpy as np
import yaml
import glob
import os
import scipy


yaml_dirs = ['./dataset_train_rgb.zip/', './dataset_additional_rgb/', './dataset_test_rgb.zip/']
yaml_files = ['train.yaml', 'additional_train.yaml', 'test.yaml']

target_dir_trn = './trn_dataset/'
target_dir_val = './val_dataset/'
target_dir_tst = './tst_dataset/'

all_data = {}
all_data['green'] = []
all_data['red'] = []
all_data['yellow'] = []
all_data['other'] = []

all_labels = {}
data = []

target_size = (64, 64)


def load_yaml(yaml_dir, yaml_fname):
    '''
    loads the contents of tha yaml file
    '''
    
    return yaml.safe_load(open(yaml_dir+yaml_fname))



def add_data_from_yaml(yaml_dir):
    '''
    adds to the global dictionary with characteristics of bounding boxes
    '''
    
    global all_data
    global data
    
    # add to the global dictionary
    for entry in data:
        for traffic_light in entry['boxes']:
            if traffic_light['label'].find('Green') != -1:
                all_data['green'].append({
                        'x_min': traffic_light['x_min'],
                        'x_max': traffic_light['x_max'],
                        'y_min': traffic_light['y_min'],
                        'y_max': traffic_light['y_max'],
                        'path': entry['path'],
                        'other_lights': np.delete(np.asarray(entry['boxes']), np.argwhere(np.asarray(entry['boxes'])==traffic_light)).tolist()})
            elif traffic_light['label'].find('Red') != -1:
                all_data['red'].append({
                        'x_min': traffic_light['x_min'],
                        'x_max': traffic_light['x_max'],
                        'y_min': traffic_light['y_min'],
                        'y_max': traffic_light['y_max'],
                        'path': entry['path'],
                        'other_lights': np.delete(np.asarray(entry['boxes']), np.argwhere(np.asarray(entry['boxes'])==traffic_light)).tolist()})
            elif traffic_light['label'].find('Yellow') != -1:
                all_data['yellow'].append({
                        'x_min': traffic_light['x_min'],
                        'x_max': traffic_light['x_max'],
                        'y_min': traffic_light['y_min'],
                        'y_max': traffic_light['y_max'],
                        'path': entry['path'],
                        'other_lights': np.delete(np.asarray(entry['boxes']), np.argwhere(np.asarray(entry['boxes'])==traffic_light)).tolist()})
            else:
                all_data['other'].append({
                        'x_min': traffic_light['x_min'],
                        'x_max': traffic_light['x_max'],
                        'y_min': traffic_light['y_min'],
                        'y_max': traffic_light['y_max'],
                        'path': entry['path'],
                        'other_lights': np.delete(np.asarray(entry['boxes']), np.argwhere(np.asarray(entry['boxes'])==traffic_light)).tolist()})
                        
    
    return data


def modify_filenames(yaml_dir):
    
    global data

    for entry in data:
        entry['path'] = yaml_dir+entry['path'][2:]
        


def modify_test_filenames():
    
    global data
    
    for entry in data:
        path = os.path.normpath(entry['path'])
        new_path = './rgb/test/' + os.path.split(path)[1]
        entry['path'] = new_path
    


def delete_current_datasets(target_dir, verbose=True):
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
        

def get_box_coords(x_min, x_max, y_min, y_max, image_x_max, image_y_max, random=True):
    '''
    calculate and return bounding box coordinates for creating training sets
    '''
    target_x_min = 0
    target_y_min = 0
    if random:
        cur_width = y_max-y_min
        cur_height = x_max-x_min
        if target_size[0] < cur_height:
            target_x_min = x_min + np.random.randint(0, cur_height-target_size[0])
        elif target_size[0] > cur_height:
            target_x_min = max(x_min - np.random.randint(0, target_size[0]-cur_height),0)
        if target_size[1] < cur_width:
            target_y_min = y_min + np.random.randint(0, cur_width-target_size[1])
        elif target_size[1] > cur_width:
            target_y_min = max(y_min - np.random.randint(0, target_size[1]-cur_width),0)    
    else:
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        target_x_min = max(center_x - target_size[0]//2, 0)
        target_y_min = max(center_y - target_size[1]//2, 0)
    target_x_max = min(target_x_min + target_size[0], image_x_max)
    target_y_max = min(target_y_min + target_size[1], image_y_max)
    
    return target_x_min, target_x_max, target_y_min, target_y_max 
        

def does_intersect(box1, box2):
    '''
    return true if box1 intersects box2
    box1 = (x1, x2, y1, y2)
    box2 = (x1, x2, y1, y2)
    '''
    
    bool1= False
    bool1 = bool1 or box1[0] >= box2[1]
    bool1 = bool1 or box1[1] <= box2[0]
    bool1 = bool1 or box1[2] >= box2[3]
    bool1 = bool1 or box1[3] <= box2[2]
    
    return not bool1
    


def build_data_sets(target_dir, min_records=100, add_variations=False, verbose=False, add_others=0):
    '''
    adds all images to the dataset
    '''
    global all_labels
    global data
    
    max_height = 0
    max_width = 0
    
    for status in all_data.keys():
        #
        counter = 0
        target_num = min_records
        if status == 'other':
            target_num -= add_others
        # 
        while counter < target_num:
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
                x1, x2, y1, y2 = get_box_coords(x_min, x_max, y_min, y_max, img.shape[1], img.shape[0], add_variations)
                # check to see if the box intersects other traffic lights in the same image
                intersect_bool = False
                for other_tl in box['other_lights']:
                    intersect_bool = intersect_bool or does_intersect([x1, x2, y1, y2], [other_tl['x_min'], other_tl['x_max'], other_tl['y_min'], other_tl['y_max']])
                if not intersect_bool:
                    img_box = img[y1:y2, x1:x2, :]
                    if (img_box.shape[0]==target_size[0] and img_box.shape[1]==target_size[1]):
                        fname = target_dir+status+str(counter)+'.png'
                        scipy.misc.imsave(fname, img_box)
                        all_labels[fname] = (status, path)
                        counter += 1
        
        
        if add_others:
            while counter < min_records:
                rand_index = np.random.randint(0, len(data))
                img = scipy.misc.imread(data[rand_index]['path'])
                x1 = np.random.randint(0,img.shape[1]-target_size[0])
                y1 = np.random.randint(0,img.shape[0]-target_size[1])
                x2 = x1 + target_size[0]
                y2 = y1 + target_size[1]
                # check to see if the box intersects other traffic lights in the same image
                intersect_bool = False
                for other_tl in data[rand_index]['boxes']:
                    intersect_bool = intersect_bool or does_intersect([x1, x2, y1, y2], [other_tl['x_min'], other_tl['x_max'], other_tl['y_min'], other_tl['y_max']])
                if not intersect_bool:
                    img_box = img[y1:y2, x1:x2, :]
                    if (img_box.shape[0]==target_size[0] and img_box.shape[1]==target_size[1]):
                        fname = target_dir+status+str(counter)+'.png'
                        scipy.misc.imsave(fname, img_box)
                        all_labels[fname] = (status, path)
                        counter += 1    
            
            
    df1 = pd.DataFrame.from_dict(all_labels, orient='index')
    writer = pd.ExcelWriter(target_dir+'labels.xlsx')
    df1.to_excel(writer,'Sheet1')
    writer.save()
       
    if verbose:
        print(max_height, ' ', max_width)        



def reset_data():
    
    global all_data
    global all_labels
    global data
    
    all_data = {}
    all_data['green'] = []
    all_data['red'] = []
    all_data['yellow'] = []
    all_data['other'] = []
    
    all_labels = {}
    
    data = []


def main():
    
    global data

    # training dataset
    reset_data()
    delete_current_datasets(target_dir_trn)
    data = load_yaml(yaml_dirs[0], yaml_files[0])
    modify_filenames(yaml_dirs[0])
    add_data_from_yaml(yaml_dirs[0])
    build_data_sets(target_dir_trn, min_records=100, add_variations=True, verbose=False, add_others=50)
      
    # validation dataset
    reset_data()
    delete_current_datasets(target_dir_val)
    data = load_yaml(yaml_dirs[1], yaml_files[1])
    modify_filenames(yaml_dirs[1])
    add_data_from_yaml(yaml_dirs[1])
    build_data_sets(target_dir_val, min_records=15, add_variations=True, verbose=False, add_others=5)
       
    # test dataset
    reset_data()
    delete_current_datasets(target_dir_tst)
    data = load_yaml(yaml_dirs[2], yaml_files[2])
    modify_test_filenames()
    modify_filenames(yaml_dirs[2])
    add_data_from_yaml(yaml_dirs[2])
    build_data_sets(target_dir_tst, min_records=50, add_variations=True, verbose=False)
       
    pass


if __name__ == '__main__':
    main()


