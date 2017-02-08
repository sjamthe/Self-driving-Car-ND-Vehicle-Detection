
"""
This program reads the labels.csv file for the Udacity annotated dataset
https://github.com/udacity/self-driving-car/tree/master/annotations
and extracts cars and non-cars images for training the classifier
"""
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sliding_window import draw_boxes, all_sliding_windows
import random

"""
function to resize the image and remap the box locations
"""
def resize_image(image, bboxes, size=(1280, 720)):
  #print ("old",image.shape)
  oldheight,oldwidth,depth = image.shape
  width, height = size
  image = cv2.resize(image,size, interpolation = cv2.INTER_AREA)
  #print ("new",image.shape)
  newbboxes = []
  for bbox in bboxes:
    xmin = int(bbox[0][0]*width/oldwidth)
    ymin = int(bbox[0][1]*height/oldheight)
    xmax = int(bbox[1][0]*width/oldwidth)
    ymax = int(bbox[1][1]*height/oldheight)
    newbbox = ((xmin,ymin),(xmax,ymax))
    newbboxes.append(newbbox)
    #print (bbox, newbbox)

  return image, newbboxes

"""
extract car and truck boxes from the labels.csv data and group per image
"""
def extract_bboxes(data):
  prevfile = None
  bboxes = []
  imagedata = []
  for row in data:
    #print (row)
    file = row[0]
    obj = row[5]

    if(obj == '"car"' or obj == '"truck"'):
      bbox = ((int(row[1]),int(row[2])),(int(row[3]),int(row[4])))
      bboxes.append(bbox)

    if(prevfile is not None and file != prevfile):
      prevfile = file
      if(len(bboxes) > 0):
        imagedata.append([file, bboxes])
      bboxes = []
    elif(prevfile is None):
      prevfile = file
  #Make sure we capture last file too
  if(len(bboxes) > 0):
    imagedata.append([file, bboxes])

  print("rows", len(data), "images", len(imagedata))
  return imagedata

"""
Add heat where the bboxes are.
"""
def add_heat(image, bbox_list):
  # Iterate through list of bboxes
  heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
  for box in bbox_list:
    # Add += 1 for all pixels inside each bbox
    # Assuming each "box" takes the form ((x1, y1), (x2, y2))
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] = 255
    #print("Added heat for ", box, np.max(heatmap))
  # Return updated heatmap
  return heatmap

"""
Separate bboxes with cars and non-cars
bboxes with heatmap > 0 have cars
"""
def find_cars(image, heatmap, file):
  windows = all_sliding_windows(image, max_image_y=420)
  carimages = []
  notcarimages = []
  #print("processing " + file)
  for window in windows:
    img = cv2.resize(image[window[0][1]:window[1][1],
                     window[0][0]:window[1][0]],
                     (64, 64), interpolation = cv2.INTER_AREA)
    heatimg = cv2.resize(heatmap[window[0][1]:window[1][1],
                     window[0][0]:window[1][0]],
                     (64, 64), interpolation = cv2.INTER_AREA)
    #if 20% of image is hot mark it as car
    hot = np.count_nonzero(heatimg!=0)
    #Only imges with 50% car are cars
    if(hot/4096 >= 0.5):
      # we have found a car image
      #plt.title(str(hot) + " " + file)
      #plt.subplot(121)
      #plt.imshow(img)
      #plt.subplot(122)
      #plt.imshow(heatimg)
      #plt.show()
      carimages.append(img)
    elif(hot == 0): #We want absolutely no car parts
      # not a car image
      notcarimages.append(img)
  #print("for ", file, "found car", len(carimages), "images and ", len(notcarimages),"not car images")
  return carimages, notcarimages

"""
save images to vehicle or non-vehicle folders
"""
def save_images(type,file,images):
  cnt = 1
  file = file.replace(".jpg","-")
  for image in images:
    if(type == "cars"):
      dir = "vehicles/Udacity/"
    else:
      dir = "non-vehicles/Udacity/"
    imagefile = dir + file + str(cnt) + ".jpg"
    cv2.imwrite(imagefile, image)
    cnt +=1

def main():
  data = np.genfromtxt("object-dataset/labels.csv", dtype='str', usecols=(0,1,2,3,4,6))
  data = extract_bboxes(data)

  totalcars = 0
  totalnotcars = 0
  for file, bboxes in data:
    image = mpimg.imread("object-dataset/"+file)
    #Rescale image and bboxes to new size
    image, newbboxes = resize_image(image, bboxes)
    #create heatmap where the cars are (as defined by bboxes)
    heatmap = add_heat(image, newbboxes)
    carimages, notcarimages = find_cars(image, heatmap, file)
    cars = len(carimages)
    notcars = len(notcarimages)
    totalcars += cars
    totalnotcars += notcars
    print(file, cars,notcars)
    if totalcars > 0:
      save_images("cars",file,carimages)
      #save same number of cars and notcars
      if notcars > cars:
        indicies = random.sample(range(notcars), cars)
        notcarimages = [notcarimages[i] for i in indicies]
      save_images("notcars",file,notcarimages)
    #img = draw_boxes(image, newbboxes)
    #plt.subplot(122)
    #plt.title(newbboxes)
    #plt.imshow(img)
    #plt.show()
    #if(totalcars > 100):
    #  break
  print ("END: Found %d cars and %d not cars" % (totalcars, totalnotcars))
""" Main function for testing purpose. Input sequence of images """
if __name__ == '__main__':
    main()
