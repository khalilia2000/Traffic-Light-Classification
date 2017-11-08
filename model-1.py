from imageutil import get_trn_val_data
from imageutil import get_tst_data
from imagedataset import ImgDataSet
import numpy as np
import scipy
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal
import json


# defining global variables
# image sizes after pre-processing  
img_rows = 90
img_cols = 320
img_ch= 3
img_norm = 0.5 # max/min of normalized pixel values

# maximum number of images to read from disk into memory
max_mem = 128
work_dir = '.'
  
  

def data_generator(num_images_to_load, batch_size, training_filenames, training_labels):
  """
  data generator for training data
  num_iamges_to_load: is the number images that will be loaded into memory each time the files are read from disk
  batch_size: is the batch_size of training data that is yielded in the generator
  training_filenames: is the ndarray of filenames containing the training data
  training_labels: is the ndarray of labels corresponding to the filenames
  """
  # at least batch_size number of images should be loaded into memory each time  
  assert num_images_to_load >= batch_size
  # number of images and labels should be the same
  assert training_filenames.shape[0] == training_labels.shape[0]
  
  # image sizes after pre-processing  
  global img_rows
  global img_cols
  global img_dir
  global img_norm
  
  # length of the training dataset
  total_trn_images = len(training_filenames)
  
  while 1:     
    # Shuffling the images before loading
    perm = np.arange(len(training_filenames))
    np.random.shuffle(perm)
    filenames = training_filenames[perm]    
    labels = training_labels[perm]
    
    # loading images into memory in batches and passing on to the optimizer
    start_index = 0
    while start_index < total_trn_images-1:
      
      # loading image data into memory  
      loaded_images = []
      num_loaded = min(num_images_to_load,total_trn_images-start_index)
      for i in range(num_loaded): 
        loaded_images.append(scipy.misc.imread(filenames[start_index+i]))
      loaded_images = np.asarray(loaded_images)
            
      # creating image dataset and pre-processing the images
      sliced_labels = labels[start_index:start_index+num_loaded]
      img_data_set = ImgDataSet(loaded_images, sliced_labels, norm_max_min=img_norm, scaled_dim=(img_rows,img_cols))
      img_data_set.pre_process(add_flipped=False)
      
      # passing on batches of data
      num_batches = img_data_set.num_examples // batch_size
      for i in range(num_batches):
        yield img_data_set.next_batch(min(batch_size,num_loaded))
        
      # adjusting and moving forward the start_index
      start_index += num_loaded      



def get_model_1():
  
    # image sizes after pre-processing  
    global img_rows
    global img_cols
    global img_ch  
  
    model = Sequential()
    # Convolution 1
    kernel_size = (5,5)
    nb_filters = 36
    k_i = TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
    model.add(Conv2D(nb_filters, kernel_size, padding='valid', 
                     kernel_initializer=k_i, input_shape=(img_rows, img_cols, img_ch)))
    # Pooling
    pool_size = (2,2)
    model.add(MaxPooling2D(pool_size=pool_size))
    # Dropout
    keep_prob = 0.5
    model.add(Dropout(keep_prob))
    # Convolution 2
    kernel_size = (5,5)
    nb_filters = 36
    model.add(Conv2D(nb_filters, kernel_size, padding='valid', kernel_initializer=k_i))
    # Pooling
    pool_size = (2,2)
    model.add(MaxPooling2D(pool_size=pool_size))
    # Activation
    model.add(Activation('relu'))
    # Convolution 3
    kernel_size = (4,4)
    nb_filters = 48
    model.add(Conv2D(nb_filters, kernel_size, padding='valid', kernel_initializer=k_i))
    # Pooling
    pool_size = (2,2)
    model.add(MaxPooling2D(pool_size=pool_size))
    # Activation
    model.add(Activation('relu'))
    # Convolution 4
    kernel_size = (3,3)
    nb_filters = 64
    model.add(Conv2D(nb_filters, kernel_size, padding='valid', kernel_initializer=k_i))
    # flatten
    model.add(Flatten())
    # fully connected 1
    model.add(Dense(100))
     # fully connected 2
    model.add(Dense(50))
    # fully connected 3
    model.add(Dense(16))
    # fully connected 4
    model.add(Dense(4))
    model.add(Activation('softmax'))
    # compiling the model
    #using default hyper parameters when creating new network
    Adam_Optimizer = Adam(lr=0.0005)
    model.compile(optimizer=Adam_Optimizer, loss='categorical_crossentropy', 
                  metrics=['accuracy']) 
    #
    return model


def get_model_vgg():
  
    # image sizes after pre-processing  
    global img_rows
    global img_cols
    global img_ch  

    # define initializer
    k_i = TruncatedNormal(mean=0.0, stddev=0.01, seed=None)

    model = Sequential()
    model.add(Conv2D(64, 3, strides=(1,1), activation='relu', kernel_initializer=k_i, input_shape=(img_rows, img_cols, img_ch)))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(128, 3, strides=(1,1), activation='relu', kernel_initializer=k_i))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(256, 3, strides=(1,1), activation='relu', kernel_initializer=k_i))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(512, 3, strides=(1,1), activation='relu', kernel_initializer=k_i))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
  
    # compiling the model
    #using default hyper parameters when creating new network
    Adam_Optimizer = Adam(lr=0.0005)
    model.compile(optimizer=Adam_Optimizer, loss='categorical_crossentropy', 
                  metrics=['accuracy']) 
    #
    return model



def load_model_and_train(model_file, X_trn, y_trn, X_val, y_val, epochs=10, b_size=64):
  """
  loads the model and weights that were previously saved, and
  continues training the model for the number of epochs specified.
  X_trn, X_val: ndarray of file names pertaining to training and validation datasets
  y_trn, y_val: ndarray of steering angles for training and validation datasets.
  epochs: number of epochs to train the model
  b_size: batch size
  """
  global max_mem
  # loading the model
  with open(model_file, 'r') as jfile:
    json_str = json.loads(jfile.read())
    model = model_from_json(json_str)

  Adam_Optimizer = Adam(lr=0.0005) # reducing learning rate for trainin on additional data
  model.compile(optimizer=Adam_Optimizer, loss='categorical_crossentropy', 
                  metrics=['accuracy'])
  weights_file = model_file.replace('json', 'h5')
  model.load_weights(weights_file)
  
  # training the model
  num_trn_samples = ((len(X_trn)*2) // b_size) * b_size
  num_val_samples = ((len(X_val)*2) // b_size) * b_size
  if num_val_samples:
    model.fit_generator(data_generator(max_mem,b_size,X_trn,y_trn),
                        steps_per_epoch=num_trn_samples//b_size,epochs=epochs, 
                        validation_data=data_generator(max_mem,b_size,X_val,y_val), 
                        validation_steps=num_val_samples//b_size)
  else:
    model.fit_generator(data_generator(max_mem,b_size,X_trn,y_trn),
                        steps_per_epoch=num_trn_samples//b_size,epochs=epochs)
    
  
  return model
  


def build_model_and_train(X_trn, y_trn, X_val, y_val, epochs=10, b_size=64):
  """
  builds a model with random weights and trains the model
  X_trn, X_val: ndarray of file names pertaining to training and validation datasets
  y_trn, y_val: ndarray of steering angles for training and validation datasets.
  epochs: number of epochs to train the model
  b_size: batch size
  """
  global max_mem
  # creating the model
  #model = get_model_1()
  model = get_model_vgg()
  # training the model
  num_trn_samples = ((len(X_trn)*2) // b_size) * b_size
  num_val_samples = ((len(X_val)*2) // b_size) * b_size
  model.fit_generator(data_generator(max_mem,b_size,X_trn,y_trn),
                      steps_per_epoch=num_trn_samples//b_size,epochs=epochs, 
                      validation_data=data_generator(max_mem,b_size,X_val,y_val), 
                      validation_steps=num_val_samples//b_size)
  return model
  
  

def save_model_and_weights(model):
  """
  saves the model structure and the weights to model.json and model.h5 respectively.
  """
  global work_dir
  # saving the model and its weights
  print()
  print('Saving model and weights...')
  model.save_weights(work_dir+'model.h5')
  with open(work_dir+'model.json','w') as outfile:
    json.dump(model.to_json(), outfile)
  print('Done.')
  pass

  
  
def main():
  
  # splitting data into training and validatoin sets
  test_ratio = 0.15 # ratio of validation images to total number of images
  X_trn, y_trn, X_val, y_val = get_trn_val_data(test_ratio)
  #X_tst, y_tst = get_tst_data()
  
  build_from_scratch = False
  if build_from_scratch:
      # building from scratch and training
      print('buiding model from scratch ...')
      model = build_model_and_train(X_trn, y_trn, X_val, y_val, epochs=5, b_size=128)
      save_model_and_weights(model)
  else: 
      # loading model and training
      print('loading model ...')
      model_file = work_dir+'model.json'
      model = load_model_and_train(model_file, X_trn, y_trn, X_val, y_val, epochs=3, b_size=128)
      save_model_and_weights(model)
  
  return model
  



if __name__ == '__main__':
  print('model.py is loaded')
  model = main()
    