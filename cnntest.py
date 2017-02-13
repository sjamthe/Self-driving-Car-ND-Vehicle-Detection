import numpy as np
import cv2
import sys
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras import models
from keras.utils import np_utils
from vehicle_model import load_data, rescale_resize_image


def evaluate(X_input, Y_output, model):
  X_input = np.array(X_input)
  Y_output = np.array(Y_output)
  Y_output = np_utils.to_categorical(Y_output, 2)
  #results = model.evaluate(X_input, Y_output, batch_size=32, verbose=1, sample_weight=None)
  probs = model.predict_proba(X_input, batch_size=32, verbose=1)
  cars = np.zeros_like(probs[:,1])
  cars[probs[:,1] > .5] = 1
  pctcars = np.sum(cars)/len(X_input)
  print("% of cars=",pctcars)

# python3 cnntest.py 'steering-model.h5' vehicles 1
def main():
  modelfile = sys.argv[1]
  imagetype = sys.argv[2]
  suffix = sys.argv[3]

  dirname='/Volumes/personal/SDC-course-notes/project5/data/'
  prefixes = np.genfromtxt(dirname + 'udacity-file.lst',dtype='str')

  if imagetype == 'vehicles':
    label = 1
  else:
    label = 0
  model = models.load_model(modelfile)
  batch=1000
  cnt = 0
  imagefiles = []
  X_input = []
  Y_output = []
  for prefix in prefixes:
    imagefile = dirname +imagetype+'/Udacity/'+prefix+'-'+suffix +'.jpg'
    if os.path.isfile(imagefile):
      imagefiles.append(imagefile)
      cnt += 1

    if cnt >= batch:
      break
  X_input, Y_output = load_data(imagefiles, label, X_input,Y_output)
  evaluate(X_input, Y_output, model)

""" Main function for testing purpose. Input sequence of images """
if __name__ == '__main__':
    main()
