#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:29:05 2017

@author: ali
"""

import pandas as pd
import numpy as np


dataset_dirs = ['./trn_dataset/', './val_dataset/', './tst_dataset/']


def load_and_pre_process_dataset(index):
    '''
    load and pre-process the datasets
    '''
    # load the dataset
    df1 = pd.read_excel(dataset_dirs[index]+'labels.xlsx', 'Sheet1')
    labels = df1[0].tolist()
    fnames = df1.index.tolist()
    
    return fnames, labels


def main():
    pass


if __name__ == '__main__':
    main()