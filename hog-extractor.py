import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
import random
from skimage.feature import hog

#Define a function to convert to correct color image
def color_convertor(image, cspace):
    image = rescale_image(image)
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif cspace == 'Gray':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        feature_image = np.copy(image)
    feature_image = rescale_image(feature_image)
    return feature_image

def rescale_image(image):
    max = np.max(image)
    if(max <= 1):
        image = image.astype(np.float32)*255.
        #print("rescaled ")
    return image

# Define a function to return HOG features and visualization
def get_hog_features(image, cspace='HSV', hog_channel='ALL', orient=9,
                     pix_per_cell=16, cell_per_block=4, feature_vector=True):
    img = color_convertor(image, cspace)
    if hog_channel == 'ALL' and cspace != 'Gray':
        hog_features = []
        for channel in range(img.shape[2]):
            features, hog_image = hog(img[:,:,channel], orientations=orient,
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block),
                        transform_sqrt=False,
                        visualise=True, feature_vector=feature_vector)
            hog_features.append(features)
        hog_features = np.ravel(hog_features)
    elif cspace == 'Gray':
        hog_features = hog(img, orientations=orient,
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block),
                        transform_sqrt=True,
                        visualise=False, feature_vector=feature_vector)
    else:
        hog_features = hog(img[:,:,hog_channel], orientations=orient,
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block),
                        transform_sqrt=True,
                        visualise=False, feature_vector=feature_vector)
    return hog_features, hog_image

def main():
  cspace = 'YCrCb' #
  orient = 9
  pix_per_cell = 12
  cell_per_block = 2
  hog_channel = 'ALL'
  dirname='/Volumes/personal/SDC-course-notes/project5/data/'
  images = glob.glob(dirname + 'vehicles/KITTI_extracted/*.png')
  #cars = ['./test_images/car.jpg']
  for file in images:
    image = mpimg.imread(file)
    #image = cv2.resize(image,(64, 64), interpolation = cv2.INTER_AREA)
    hog_features, hog_image = get_hog_features(image, cspace, hog_channel, orient,
                                    pix_per_cell, cell_per_block)
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(hog_image,cmap = 'gray')
    plt.show()

""" Main function for testing purpose. Input sequence of images """
if __name__ == '__main__':
    main()
