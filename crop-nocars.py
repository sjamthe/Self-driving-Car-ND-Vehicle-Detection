
"""
This program reads crops images of left half of project_video
as these images need to be used to train the model.
"""
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sliding_window import draw_boxes, all_sliding_windows
import random

"""
save images to vehicle or non-vehicle folders
"""
def save_images(image, windows, imgcnt):
  dir = "non-vehicles/Project/"
  cnt = 0
  for window in windows:
    img = cv2.resize(image[window[0][1]:window[1][1],
                   window[0][0]:window[1][0]],
                   (64, 64), interpolation = cv2.INTER_AREA)

    imagefile = dir + str(imgcnt) +"-"+ str(cnt) + ".jpg"
    plt.imsave(imagefile, img)
    cnt +=1

def process_images(files):
  imgcnt=0
  for filename in files:
      image = mpimg.imread(filename)
      #crop image to see the left half
      height,width,depth = image.shape
      image = image[:,0:int(width/2):,]
      windows = all_sliding_windows(image, max_image_y=320)
      #We get 190 windows that are too many
      #pick 20 windows randomly
      windows = random.sample(windows, 20)
      #call save
      save_images(image, windows, imgcnt)
      imgcnt += 1
      #print(len(windows))
      if(0):
        window_img = draw_boxes(image, windows, color=(20, 200, 255), thick=2)
        plt.imshow(window_img)
        plt.show()

def main():
   if '.jpg' in sys.argv[1]:
    process_images(sys.argv[1:])

""" Main function for testing purpose. Input sequence of images """
if __name__ == '__main__':
    main()
