import sys
import numpy as np
import os.path
import glob
import matplotlib.image as mpimg
import cv2
from scipy.ndimage import imread
from scipy.misc import imresize
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.layers.core import Dropout
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

dirname='/Volumes/personal/SDC-course-notes/project5/data/'
modelfile='vehicle-model.h5'

def load_data(imagefiles, label, X_input, Y_output):

  for file in imagefiles:
    try:
      image = mpimg.imread(file)
      img = rescale_resize_image(image)
      X_input.append(img)
      Y_output.append(label)
    except ValueError:
      print("Error for ",file)
  print("Read inputs",len(X_input),"from",len(imagefiles))

  return X_input, Y_output

def rescale_resize_image(image):
  #model is trained in 32x32 images
  size = (32,32)
  img = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

  max = np.max(img)
  if(max <= 1):
      img = img.astype(np.float32)*255.
  return img

def build_model():
  input_shape = (32,32,3)
  model = Sequential()
  model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape,
                          activation='relu',W_regularizer=l2(0.001)))
  model.add(MaxPooling2D())
  model.add(Convolution2D(64, 3, 3, border_mode='same',activation='relu',W_regularizer=l2(0.001)))
  model.add(Convolution2D(64, 3, 3, border_mode='same',activation='relu',W_regularizer=l2(0.001)))
  model.add(MaxPooling2D())
  model.add(Flatten())
  model.add(Dense(512, name="hidden1",activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2,name="new-output",activation='softmax'))

  return model

def train_model(model, X_input, Y_output):
  learning_rate = .00001

  opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

  X_input = np.array(X_input)
  Y_output = np.array(Y_output)

  X_train, X_test, y_train, y_test = train_test_split(X_input, Y_output,
                                         test_size=0.1, random_state=11)

  y_train = np_utils.to_categorical(y_train, 2)
  y_test = np_utils.to_categorical(y_test, 2)

  print("Training model with inputsize",len(X_train))

  hist = model.train_on_batch(X_train, y_train)
  print("train = ",hist)
  hist = model.test_on_batch(X_test, y_test)
  print("test = ",hist)
  #
  #batch_size = 64
  #nb_epoch = 2
  #hist = model.fit(X_train, y_train,
  #                    batch_size=batch_size, nb_epoch=nb_epoch,
  #                    validation_data=(X_test,y_test), verbose=1)
  model.save(modelfile)

  return hist

def udacity_images(type, limit, suffix):
  images = []
  prefixes = np.genfromtxt(dirname + 'udacity-file.lst',dtype='str')
  for prefix in prefixes:
    imagefile = dirname +type+'/Udacity/'+prefix+'-'+suffix+'.jpg'
    if os.path.isfile(imagefile):
        images.append(imagefile)
    if len(images) >= limit:
        break
  #print(len(images),"udacity images selected")
  return images

def generate_arrays():
  cars_input = []
  i=0
  batch=32
  while True:

    if(len(cars_input)/(i+1) < batch):
      i=0 #reinitialize for the loop
      images = glob.glob(dirname + 'vehicles/GTI_*/*.png')
      cars_input, cars_output = load_data(images, 1, [], [])
      images = glob.glob(dirname + 'vehicles/KITTI_extracted/*.png')
      cars_input, cars_output = load_data(images, 1,cars_input,cars_output)
      images = glob.glob(dirname + 'non-vehicles/GTI/*.png')
      nocars_input, nocars_output = load_data(images, 0, [], [])
      images = glob.glob(dirname + 'non-vehicles/Extras/*.png')
      nocars_input, nocars_output = load_data(images, 0, nocars_input, nocars_output)
      print("Read cars",len(cars_input),"nocars",len(nocars_input))
      if len(cars_input) < len(nocars_input):
        nocars_input = nocars_input[0:len(cars_input)]
      elif len(nocars_input) < len(cars_input):
        cars_input = cars_input[0:len(nocars_input)]
      print("adjusted cars",len(cars_input),"nocars",len(nocars_input))

    X_input = np.array(cars_input[i*batch:(i+1)*batch])
    #print("x",X_input.shape)
    X_input = np.append(X_input, nocars_input[i*batch:(i+1)*batch],axis=0)
    #print("x",X_input.shape)

    Y_output = np.array(cars_output[i*batch:(i+1)*batch])
    Y_output = np.append(Y_output, nocars_output[i*batch:(i+1)*batch])
    Y_output = np.array(Y_output)
    y = np_utils.to_categorical(Y_output, 2)

    #print("for",i,"passing",X_input.shape, Y_output.shape)
    i+=1
    yield (X_input, y)

def train_with_generator():
  model = build_model()
  learning_rate = 0.00001
  opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

  hist = model.fit_generator(generate_arrays(), samples_per_epoch=64,
               nb_epoch=4000, verbose=1)
  #print(hist.history)
  model.save(modelfile)
def main():

  model = build_model()

  #load pre-trained model
  #model.load_weights(modelfile, by_name=True)
  # Read in car and non-car images
  if(0):
    Y_output = []
    X_input = []
    images = glob.glob(dirname + 'vehicles/KITTI_extracted/*.png')
    X_input, Y_output = load_data(images, 1, X_input, Y_output)
    images = glob.glob(dirname + 'non-vehicles/Extras/*.png')
    X_input, Y_output = load_data(images, 0, X_input, Y_output)
    train_model(model, X_input, Y_output)
  if(0):
    Y_output = []
    X_input = []
    images = glob.glob(dirname + 'vehicles/GTI_*/*.png')
    X_input, Y_output = load_data(images, 1, X_input, Y_output)
    images = glob.glob(dirname + 'non-vehicles/GTI/*.png')
    X_input, Y_output = load_data(images, 0, X_input, Y_output)
    train_model(model, X_input, Y_output)
  if(0):
    for suffix in ['5','10','11','15']:
      Y_output = []
      X_input = []
      images = udacity_images('vehicles',10000,suffix)
      X_input, Y_output = load_data(images, 1, X_input, Y_output)
      images = udacity_images('non-vehicles',10000,suffix)
      X_input, Y_output = load_data(images, 0, X_input, Y_output)
      train_model(model, X_input, Y_output)
  if(0):
    Y_output = []
    X_input = []
    images = glob.glob('/Users/sjamthe/Documents/SDCND/project5/data/non-vehicles/Project/*.jpg')
    X_input, Y_output = load_data(images, 0, X_input, Y_output)
    train_model(model, X_input, Y_output)

  #model.save(modelfile)
""" Main function for testing purpose. Input sequence of images """
if __name__ == '__main__':
   # main()
   train_with_generator()
   #generate_arrays()

