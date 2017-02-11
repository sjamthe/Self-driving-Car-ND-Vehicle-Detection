import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
import fnmatch
from collections import deque
from features import img_features
from sliding_window import all_sliding_windows, draw_boxes, focused_windows

"""
Use sliding window technique to break image in windows
find features for each window, apply classifier and find all the windows
that may have probability (above PROB_CAR) that it has a vehicle
"""
def search_classify(image, windows, X_scaler, svc, version):
  on_windows = []
  probs = []
  features = []
  size = (64, 64)
  for window in windows:
    #print("searching ",window)
    img = cv2.resize(image[window[0][1]:window[1][1],
                      window[0][0]:window[1][0]],
                     size, interpolation = cv2.INTER_AREA)

    image_features = img_features(img, version)
    features.append(image_features)

  features = np.array(features).astype(np.float64)
  #print(features.shape)
  test_features = X_scaler.transform(features)
  #test_features = X_scaler.transform(np.array(features).reshape(1, -1))
  #predictions = svc.predict(test_features)
  probabilities = svc.predict_proba(test_features)
  cnt = 0
  PROB_CAR = 0.5 #0.8
  for pnocar,pcar in probabilities:
    if pcar > PROB_CAR:
      window = windows[cnt]
      #on_windows.append([window, pcar])
      #print("selecting",window)
      on_windows.append(window)
      probs.append(pcar)
      #window_img = draw_boxes(image, on_windows, color=(0, 0, 255), thick=6)
      #plt.imshow(window_img)
      #plt.show()
    cnt+=1
  print ("Found ",len(on_windows),"car windows out of ",len(windows))
  return on_windows, probs


def add_heat(heatmap, bbox_list, probs):
    # Iterate through list of bboxes
  for box,pcar in zip(bbox_list, probs):
    # Add += 1 for all pixels inside each bbox
    # Assuming each "box" takes the form ((x1, y1), (x2, y2))
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += pcar

  # Return updated heatmap
  return heatmap

def apply_threshold(heatmap, threshold):
  # Zero out pixels below the threshold
  heatmap[heatmap <= threshold] = 0
  # Return thresholded map
  return heatmap

def draw_labeled_bboxes(image, labels):
  img = np.copy(image)
  # Iterate through all detected cars
  for car_number in range(1, labels[1]+1):
      # Find pixels with each car_number label value
      nonzero = (labels[0] == car_number).nonzero()
      # Identify x and y values of those pixels
      nonzeroy = np.array(nonzero[0])
      nonzerox = np.array(nonzero[1])
      # Define a bounding box based on min/max x and y
      bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
      if (np.max(nonzerox) - np.min(nonzerox)) < 10:
        continue

      # Draw the box on the image
      cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
  # Return the image
  return img

def historic_persp(image, on_windows, probs, hist_windows, sequence, HIST):
  if(len(hist_windows) >= HIST):
    print("popping", len(hist_windows), HIST)
    hist_windows.popleft()

  hist_windows.append([on_windows, probs])

  heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
  all_windows = []
  count=0
  for on_windows, probs in hist_windows:
    heatmap = add_heat(heatmap, on_windows, probs)
    count+=1

  #create average
  heatmap = heatmap/count
  maxcars = np.max(heatmap)
  histogram=np.max(heatmap,axis=0) #get sum
  threshold = 1 #1.5
  #threshold = (min(counter,HIST) +1) * 3.1
  heatmap = apply_threshold(heatmap, threshold)
  labels = label(heatmap)
  print(labels[1], 'cars found on threshold %f with max %f of %d' %(threshold, maxcars, len(hist_windows)))

  #plt.imshow(heatmap, cmap='gray')
  #plt.show()
  #window_img = draw_boxes(image, on_windows, color=(0, 0, 255), thick=6)
  final_img = draw_labeled_bboxes(image, labels)
  if(1 and  __name__ == '__main__'):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    plt.xlim(0,image.shape[1])
    #ax1.set_ylim(image.shape[0], 0)
    ax1.imshow(final_img)
    ax2.plot(histogram)
    histogram_th = np.zeros_like(histogram)
    histogram_th += (threshold)
    ax2.plot(histogram_th)
    ax2.plot(np.median(histogram))
    #plt.show()
    plt.savefig('test_project_video/frame'+str(sequence)+'.jpg') #,bbox_inches='tight'
    plt.close(f)
  return final_img, hist_windows

def process_images(files, X_scaler, svc, version):
  #number of consecutive images used to predict
  HIST=10
  hist_windows = deque()
  sequence = 0
  for filename in files:
    image = mpimg.imread(filename)
    #first pass
    windows = all_sliding_windows(image, max_image_y=340)
    on_windows,probs = search_classify(image, windows, X_scaler, svc, version)
    #window_img = draw_boxes(image, on_windows, color=(20, 200, 255), thick=2)
    #plt.imshow(window_img)
    #plt.show()
    #focused search
    #on_windows = focused_windows(image, on_windows)
    #on_windows, probs = search_classify(image, on_windows, X_scaler, svc, version)
    #window_img = draw_boxes(image, on_windows, color=(20, 200, 255), thick=2)
    #plt.imshow(window_img)
    #plt.show()
    final_img, hist_windows = historic_persp(image, on_windows, probs, hist_windows,
                                              sequence, HIST)
    sequence += 1
#python3 search_classify.py /Volumes/personal/SDC-course-notes/project4/CarND-Advanced-Lane-Lines/project_video/frame[0-9].jpg /Volumes/personal/SDC-course-notes/project4/CarND-Advanced-Lane-Lines/project_video/frame[1-9][0-9].jpg /Volumes/personal/SDC-course-notes/project4/CarND-Advanced-Lane-Lines/project_video/frame[1-9][1-9][0-9].jpg /Volumes/personal/SDC-course-notes/project4/CarND-Advanced-Lane-Lines/project_video/frame[1-2][1-9][1-9][0-9].jpg

""" Main function for testing purpose. Input sequence of images """
if __name__ == '__main__':
  model = 'hogGraysvc-v2.pkl'
  #model = 'hogGray-v1.pkl'
  if 'v1' in model:
    version = 'v1'
  else:
    version = 'v2'

  X_scaler, svc = joblib.load(model)
  if '.jpg' in sys.argv[1]:
    process_images(sys.argv[1:], X_scaler, svc, version)

