##Vehicle Detection - [Project 5](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/README.md)


**Summary**

I followed the following steps for this project:

* Explore various techniques to extract [features](./features.py) from an image. The features explored were:
####1. Histogram of Oriented Gradients (HOG)
####2. Binned Spatial Features
####3. Histograms on various color spaces

* Train [classifier](./classifier.py) on various combination of above feature settings using images from  [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/)

* In my iterative process of trying several models I thought the image data set was not enough. I downloaded [Udacity's labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) and wrote a program ([parse-object-dataset.py](./parse-object-dataset.py)) to split the images into vehicles and non-vehicles.

* Wrote a [test utility](./test.py) to test models on a set of labeled images kept separate from the training set.

* Implemented a [sliding-window technique ](./sliding_window.py) to split images into windows for detecting vehicles.

* Wrote a wrapper [search_classify](./search_classify.py) that calls classifier on all windows in an image and applies heat map on the windows as well as on concurrent images to bounding boxes for cars.

* Created a pipeline [process_video.py](./process_video.py) that runs on a video stream and creates a video with bounding boxes.

[//]: # (Image References)
[image1]: ./output_images/hog-example.png
[image2]: ./output_images/all_sliding_windows.jpg
[image3]: ./output_images/threshold1.jpg
[image4]: ./output_images/threshold2.jpg
[image5]: ./output_images/processed_img.jpg
[video1]: ./project_video.mp4

---

##1. [Feature Selection](./features.py)
This module lets me experiment with various features and feature combinations.
* I started with color bin spatial features which resizes the image into a smaller size (32x32) and creates features based on location of pixels.
* color_hist function in the same program allows you to create features from histograms on color spaces. I experimented with a lot of color spaces RGB, HSV, LUV, HLS, YUV, YCrCb
* Histogram of Oriented Gradients (HOG) - I tried variation on color spaces (Gray, HSV, RGB) as well as  orientation (6, 9) pixel_per_cell (8,12) and cell_per_block (2,4). I found HOG features were best predictors of cars.


Here is an example using the `Gray` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(12, 12)` and `cells_per_block=(2, 2)`:


![alt text][image1]

##2. Explain how you settled on your final choice of HOG parameters.

I settled down on two version of models one with Gray, 9,12,2 and another with HSV, 9,12,2. Though HSV as slightly better during validation (82% vs 80% accuracy) both were very close when used on real images so I settled for Gray features as they were less and classifier ran faster. color_hist and bin_spatial didn't really help a lot so I remove them after a lot of trials on combined features testing.

##3. Training a [Classifier](./classifier.py)

I tested linear SVM, LogisticClassifier and SGDClassifier using a GridSearch with various parameters. In the end I settled with LogisticClassifier with the parameters in the file.


##4.[Sliding Window Search](./sliding_window.py)
I focused only on the road portion of the image ignoring the sky. The maximum window was 340x340 and sliding was 50%. The function all_sliding_windows produced all windows of 8 sizes and 50% overlap. All windows start drawing at (720-340) height from top and had width of 340, 340*6/7, 340*5,7 ... 340*1/7. Thus smaller images were near the horizon. This results in 360 windows for images size of 1280x720
See the image below to see how the windows looked.

![alt text][image2]

##5.[Search & Classify](./search_classify.py)
search_classfy function takes in an image and calls sliding_windows on the image. These 360 images are resized to 64x64 and then we extract features from them and feed it to the classifier. This function returns a list of windows that have >= 50% probability of matching a vehicle. We also return the probability of the window.

At this point we can decide if we are marking windows based on only one image or based on a sequence of images. We create a collection of these sequential images (10 in my case) and pass them to a function historic_perp.
This function creates a heatmap of the image and each window illuminates the pixels inside the window with the probability of finding a pixel (not 1 for all windows). This gives a bit granular control on the windows.
The I threshold the heatmap. To understand what threshold to use I created a histogram and plotted a graph for each image, see example below.

![alt text][image3]
![alt text][image4]


##6. Using additional dataset from Udacity.
I downloaded [Udacity's labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) and wrote a program ([parse-object-dataset.py](./parse-object-dataset.py)) to split the images into vehicles and non-vehicles. This program runs a sliding window on the images in the dataset, then applies the human annotations to find out which images have a >50% pixels of a car. Those windows are saved as cars and about same number of windows chosen randomly from that image are saved as non-vehicles samples.
This dataset results in 200K car and non-car images. I used about 15,000 cars and non-car images from this data set as train and validation set.
I also used about 10k images from this set kept separately to test my models.

##7 [Video Implementation](./process_video.py)

My final video was produced using process_video wrapper. I used cv2.VideoCapture to read each frame and used moviepy.editor to stitch the frame together.

Here's a [link to my video result](./project_video-out.mp4)

![Final Video][image5]

---

## Discussion

I see my pipeline detects false positives when there is metal railing on the road. At one point the left wall was showing up as a vehicle so I cropped images of the wall using [crop-nocars.py](./crop-nocars.py) and collected 12k images to train. I need to create more non-vehicle images that show railing. Also, the black cars are not detected as well. I need to investigate if this is due to gray scaling of the image, in that case I should experiment more with color spaces before getting HOG features.


I also feel that a DNN model will be lot better to detect cars. I plan to use the DNN I created for traffic signs on this and see if it works better.

