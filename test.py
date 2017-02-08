import numpy as np
import cv2
import sys
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from features import img_features
from sliding_window import all_sliding_windows, draw_boxes


def classify(img, X_scaler, svc, version):
  features = img_features(img, version=version)
  test_features = X_scaler.transform(np.array(features).reshape(1, -1))
  prediction = svc.predict(test_features)
  return prediction

# python3 test.py hogGray-v1.pklvehicles 1
# python3 test.py hogGray-v1.pkl non-vehicles 1
def main():
  model = sys.argv[1]
  imagetype = sys.argv[2]
  suffix = sys.argv[3]
  if 'v1' in model:
    version = 'v1'
  else:
    version = 'v2'
  dirname='/Volumes/personal/SDC-course-notes/project5/data/'
  prefixes = np.genfromtxt(dirname + 'udacity-file.lst',dtype='str')
  errorfile = imagetype + '-err.pkl'
  if os.path.isfile(errorfile):
    errors = joblib.load(errorfile)
    print ("Read %d errors from errorfile" % len(errors))
  else:
    errors = []
  notcars = 0
  cars = 0
  X_scaler, svc = joblib.load(model)

  for prefix in prefixes:
    imagefile = dirname +imagetype+'/Udacity/'+prefix+'-'+suffix +'.jpg'
    if not os.path.isfile(imagefile):
      #print("skipping", imagefile)
      continue
    #for imagefile in glob.glob('../data/'+imagetype+'/Udacity/'+prefix+'-*')
    try:
      image = mpimg.imread(imagefile)
      prediction = classify(image, X_scaler, svc, version)
      if(prediction):
        cars+=1
        #plt.imshow(image)
        #plt.show()
        if imagetype != 'vehicles':
          #print (imagefile)
          errors.append(imagefile)
      else:
        notcars+=1
        if imagetype == 'vehicles':
          #print (imagefile)
          errors.append(imagefile)
        #plt.imshow(image)
        #plt.show()
    except:
      print ("error", imagefile)
      print("Unexpected error:", sys.exc_info())
      raise

    if(cars+notcars > 1000):
      break

  print("Found",cars,"cars and ",notcars,"not car images")
  #joblib.dump(errors, errorfile)

""" Main function for testing purpose. Input sequence of images """
if __name__ == '__main__':
    main()
