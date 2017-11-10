# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:03:29 2017

@author: Ali Khalili
"""

import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib import gridspec

class ImgDataSet(object):

  def __init__(self,images,labels,norm_max_min=0.5,scaled_dim=(48,96)):
               
    """
    Construct a DataSet of Images.
    norm_max_min: the maximum/minimum range for normalizing pixel values
    scaled_dim: scaled dimension of images after pre-processing
    """
        
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._size_factor = 3
    self._aspect_ratio = 1.0 * images.shape[1] / images.shape[2]
    self._normalization_factor = norm_max_min
    self._resize_dim = scaled_dim
    
    # The following parameters will be used for retrieving data batches    
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples


  def add_flipped(self):
    """
    Adds horizontally flipped images to the dataset
    """
    self._images = np.append(self._images, np.asarray([np.fliplr(image) for image in self._images]), axis=0)
    self._labels = np.append(self._labels, np.asarray([label_value for label_value in self._labels]), axis=0)
    self._num_examples = self._images.shape[0]
    pass  


  def crop_top(self, crop_ratio):
    """
    Crops the top half of the image
    """
    img_height = round(self._images[0].shape[0] * crop_ratio)
    if len(self._images[0].shape) == 2:
      self._images = np.asarray([img[img_height:,:] for img in self._images])
    else:
      self._images = np.asarray([img[img_height:,:,:] for img in self._images])
    pass
  
    
  def crop_bottom(self, crop_ratio):
      """
      Crops the bottom half of the images
      """
      img_height = round(self._images[0].shape[0] * crop_ratio)
      if len(self._images[0].shape) == 2:
          self._images = np.asarray([img[:img_height,:] for img in self._images])
      else:
          self._images = np.asarray([img[:img_height,:,:] for img in self._images])
      pass
  

  def resize(self):
    """
    Resizes the images
    """
    self._images = np.asarray([scipy.misc.imresize(img, self._resize_dim) for img in self._images])
    pass

  
  def normalize(self):
    """
    Normalizes the piexel values of the images
    """
    self._images = self._images.astype('float32')
    self._images -= 127.5
    self._images /= 127.5
    self._images *= self._normalization_factor
    pass
  
  
  def pre_process(self, add_flipped=True, crop_ratio=0.5):
    """
    Executes the pipeline for preprocessing the images
    add_flipped: flippes the images horizontally and adds them to the dataset if True
    """
    if add_flipped:    
      self.add_flipped()
    self.crop_bottom(crop_ratio)
    self.resize()
    self.normalize()
    pass
  
  
  def next_batch(self, batch_size):
    """
    Return the next "batch_size" examples from this data set.
    batch_size: batch size to return
    """
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Start next epoch 
      # No need for suffling the data as generator shuffles the files before loading
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
 

  def plot_image(self, ax_list, grid_fig, grid_index, image_index, show_label=True, cmap=None):
    """
    plots one image in the grid space and passess the appended axis list
    ax_list: list of axes pertaining to the grid space
    grid_fig: grid space object
    grid_index: index in the grid to plot onto
    image_index: image index to plot
    show_label: whether to show the labels or not
    cmap: cmap
    """
    ax_list.append(plt.subplot(grid_fig[grid_index]))
    
    img = self._images[image_index]
    if img.shape[2] == 1:
      img = img.reshape(img.shape[0],img.shape[1])
            
    ax_list[-1].imshow(img, cmap=cmap)
    ax_list[-1].axis('off')
    ax_list[-1].set_aspect('equal')
    y_lim = ax_list[-1].get_ylim()
    if show_label:
      ax_list[-1].text(0,int(-1*y_lim[0]*0.1),'Steering Angle = {:.4f}'.format(self._labels[image_index]),color='r')
      ax_list[-1].text(0,int(-1*y_lim[0]*0.3),'Image Index = {}'.format(image_index),color='b')

    return ax_list

  
  def plot_random_grid(self, n_rows, n_cols, show_labels=True, cmap=None):
    """
    plots a random grid of images to verify
    n_rows: number or rows for the image grid
    n_cols: number of cols for the image grid
    show_labels: whether to show the labels or not
    cmap: cmap
    """
    
    # creating the grid space
    g_fig = gridspec.GridSpec(n_rows,n_cols) 
    g_fig.update(wspace=0.5, hspace=0.75)
    
    # setting up the figure
    plt.figure(figsize=(n_cols*self._size_factor,n_rows*self._size_factor*self._aspect_ratio))
    selection = np.random.choice(self._num_examples, n_rows*n_cols, replace=False)
    
    ax_list = []
    for i in range(n_rows*n_cols):
      ax_list = self.plot_image(ax_list, g_fig, i, selection[i], cmap=cmap)

