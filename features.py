import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import sys
import time
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.externals import joblib

"""
This program extracts various features
"""
dirname='/Volumes/personal/SDC-course-notes/project5/data/'

# Define a function to compute binned color features
def bin_spatial(image, size=(32, 32),cspace='RGB'):
    # Use cv2.resize().ravel() to create the feature vector
    img = color_convertor(image, cspace)
    features = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(image, nbins=32, bins_range=(0, 256), cspace='RGB'):
    img = color_convertor(image, cspace)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0])).astype(np.float32)
    hist_features = hist_features/4096. #normalize them 4096 = 64x64 image shape
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(image, cspace='HSV', hog_channel='ALL', orient=9,
                     pix_per_cell=16, cell_per_block=4, feature_vector=True):
    img = color_convertor(image, cspace)
    if hog_channel == 'ALL' and cspace != 'Gray':
        hog_features = []
        for channel in range(img.shape[2]):
            features = hog(img[:,:,channel], orientations=orient,
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block),
                        transform_sqrt=False,
                        visualise=False, feature_vector=feature_vector)
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
    return hog_features

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
#extract features for a given image
def img_features(image, version='v1'):
    # apply color conversion if other than 'RGB'
    # Apply bin_spatial() to get spatial color features
    #spatial_features = bin_spatial(image, size=(12,12))
    #print("spatial_features max = ",np.min(spatial_features),np.max(spatial_features))
    # Apply color_hist() also with a color space option now
    #rgb_hist_features = color_hist(image, nbins=32, cspace='RGB')
    #print("rgb_hist_features max = ",np.min(rgb_hist_features),np.max(rgb_hist_features))
    #hsv_hist_features = color_hist(image, nbins=32, cspace='HSV')
    #print("hsv_hist_features max = ",np.min(hsv_hist_features),np.max(hsv_hist_features))
    #hls_hist_features = color_hist(image, nbins=28, cspace='HLS')
    #print("hls_hist_features max = ",np.min(hls_hist_features),np.max(hls_hist_features))
    #luv_hist_features = color_hist(image, nbins=28, cspace='LUV')
    #print("luv_hist_features max = ",np.min(luv_hist_features),np.max(luv_hist_features))
    #yuv_hist_features = color_hist(image, nbins=28, cspace='YUV')
    #print("yuv_hist_features max = ",np.min(yuv_hist_features),np.max(yuv_hist_features))
    #ycrcb_hist_features = color_hist(image, nbins=28, cspace='YCrCb')
    #print("ycrcb_hist_features max = ",np.min(ycrcb_hist_features),np.max(ycrcb_hist_features))

    #color_features = np.concatenate((spatial_features, rgb_hist_features,
    #            hsv_hist_features, hls_hist_features,
    #            luv_hist_features,yuv_hist_features,
    #            ycrcb_hist_features))
    #HOG features
    # TODO: Tweak these parameters and see how the results change.

    #print("hog_features max = ",np.min(hog_features),np.max(hog_features))
    # Append the new feature vector to the features list
    #All features together give 99.21 accuracy
    if(version == 'v1'):
        cspace = 'Gray' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb (Gray is best)
        orient = 9
        pix_per_cell = 12
        cell_per_block = 2
        hog_channel = 0 # Can be 0, 1, 2, 'Gray' or "ALL"

        hog_features1 = get_hog_features(image, cspace, hog_channel, orient,
                                    pix_per_cell, cell_per_block)
        #features = np.concatenate((hsv_hist_features,rgb_hist_features)) #v1 features
        features = np.array(hog_features1)
    elif(version == 'v2'):
        cspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb (HSV, 9,12,2,ALL)
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        hog_channel = "ALL" # Can be 0, 1, 2, 'Gray' or "ALL"

        hog_features2 = get_hog_features(image, cspace, hog_channel, orient,
                                    pix_per_cell, cell_per_block)
        features = np.array(hog_features2)
        #features = np.concatenate((spatial_features, hog_features)) #v2 features

    return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, version='v1'):
    # Create a list to append feature vectors to
    features = []
    errors = 0
    # Iterate through the list of images
    for file in imgs:
        try:
        # Read in each one by one
            image = mpimg.imread(file)
            image_features = img_features(image, version)
            features.append(image_features)
        except ValueError:
            errors += 1
            print("Error for file",imgs)
            raise
    # Return list of feature vectors
    return features

def dump_features(images, type, version):
    #version in ['v1']: #,'v2'
    features = extract_features(images, version=version)
    joblib.dump(features, '../data/'+type+'-features-'+version+'.pkl')
    print("Extracted %d %s-%s features from %d images"% (
        len(features[0]), type, version, len(images)))

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

#python3 features.py v2
def main():
    version = sys.argv[1]
    # Read in car and non-car images
    if(1):
        images = glob.glob(dirname + 'vehicles/KITTI_extracted/*.png')
        dump_features(images,'KITTI-cars',version)
        images = glob.glob(dirname + 'non-vehicles/Extras/*.png')
        dump_features(images,'KITTI-non-cars',version)
    if(1):
        images = glob.glob(dirname + 'vehicles/GTI_*/*.png')
        dump_features(images,'GTI-cars',version)
        images = glob.glob(dirname + 'non-vehicles/GTI/*.png')
        dump_features(images,'GTI-non-cars',version)
    if(1):
        for suffix in ['5','10','11','15']:
            images = udacity_images('non-vehicles',10000,suffix)
            dump_features(images,'Udacity-non-cars'+suffix,version)
            images = udacity_images('vehicles',10000,suffix)
            dump_features(images,'Udacity-cars'+suffix,version)
    if(1):
        images = glob.glob('../data/non-vehicles/Project/*.jpg')
        dump_features(images,'Project-non-cars',version)

""" Main function for testing purpose. Input sequence of images """
if __name__ == '__main__':
    main()
