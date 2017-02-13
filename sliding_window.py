import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(255, 255, 255), thick=6):
  # Make a copy of the image
  imcopy = np.copy(img)
  # Iterate through the bounding boxes
  #print (bboxes)
  #colors = [(0, 0, 255),(0, 255, 255),(255, 0, 0),(255, 255, 0),(255, 0, 255)]
  cnt=0
  for bbox in bboxes:
    #print (bbox[0], bbox[1])
    # Draw a rectangle given bbox coordinates
    cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    cnt+=1
    #print(cnt)
  # Return the image copy with boxes drawn
  return imcopy


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(window_list, img, x_start_stop=[None, None],
                    y_start_stop=[None, None], xy_window=(64, 64),
                    xy_overlap=(0.5, 0.5)):

  # Compute the span of the region to be searched
  xspan = x_start_stop[1] - x_start_stop[0]
  yspan = y_start_stop[1] - y_start_stop[0]
  # Compute the number of pixels per step in x/y
  nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
  ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
  # Compute the number of windows in x/y
  nx_windows = np.int(xspan/nx_pix_per_step) #- 1
  ny_windows = np.int(yspan/ny_pix_per_step) - 1
  # Initialize a list to append window positions to

  # Loop through finding x and y window positions
  # Note: you could vectorize this step, but in practice
  # you'll be considering windows one by one with your
  # classifier, so looping makes sense
  maxy = img.shape[0]
  maxx = img.shape[1]
  for ys in range(ny_windows):
    for xs in range(nx_windows):
      # Calculate window position
      startx = xs*nx_pix_per_step + x_start_stop[0]
      endx = startx + xy_window[0]
      if endx > maxx:
        endx = maxx
      starty = ys*ny_pix_per_step + y_start_stop[0]
      endy = starty + xy_window[1]
      if endy > maxy:
        endy = maxy
      # Append window position to list
      if endx > startx and endy > starty:
        window_list.append(((startx, starty), (endx, endy)))
  # Return the list of windows
  #print ("scanning windows=",y_start_stop,xy_window,len(window_list))
  return window_list

def all_sliding_windows(image, max_image_y=420):
  windows = []
  max_range = 7 #7
  for n in range(0, max_range):
    winsize = np.int(max_image_y * (1- n/max_range))
    #print("win size %dx%d" %(winsize,winsize))
    y_start = image.shape[0] - max_image_y
    y_stop =  np.int(image.shape[0]-max_image_y*n/max_range/1.5)
    windows = slide_window(windows, image, x_start_stop=[0, image.shape[1]],
                      y_start_stop=[y_start, y_stop],
                      xy_window=(winsize, winsize), xy_overlap=(0.5, 0.5))
  return windows

def focused_windows(image, windows, size=(64,64)):
  maxy = image.shape[0]
  maxx = image.shape[1]
  window_list = []
  for window in windows:
    #print("processing",window)
    (startx, starty), (endx, endy) = window
    # Compute the number of windows in x/y (our input is square windows)
    nx_windows = np.int(np.round((endx-startx)/size[0]))
    if nx_windows <= 1:
      nx_windows = 2
    ny_windows = np.int(np.round((endy-starty)/size[1]))
    if ny_windows <= 1:
      ny_windows = 2

    for ys in range(ny_windows):
      newstarty = ys*size[1] + starty
      newendy = newstarty + size[1]
      if newendy > maxy:
        newendy = maxy
      for xs in range(nx_windows):
        # Calculate window position
        newstartx = xs*size[0] + startx
        newendx = newstartx + size[0]
        if newendx > maxx:
          newendx = maxx
        # Append window position to list
        if newendx > newstartx and newendy > newstarty:
          newwindow = ((newstartx, newstarty), (newendx, newendy))
          #print("creating from ", window,newwindow)
          window_list.append(newwindow)
    # Return the list of windows
    #print ("focused windows=",y_start_stop,xy_window,len(window_list))
  return window_list

def main():
  image = mpimg.imread('test_images/frame1259.jpg')
  windows = all_sliding_windows(image, max_image_y=340)
  #print ("Found ",len(windows),"windows")
  windows=windows[31:32]
  windows = focused_windows(image, windows)
  window_img = draw_boxes(image, windows, color=(20, 200, 255), thick=2)
  plt.imshow(window_img)
  #plt.savefig('output_images/all_sliding_windows.jpg',bbox_inches='tight')
  plt.show()

  """ Main function for testing purpose. Input sequence of images """
if __name__ == '__main__':
    main()
