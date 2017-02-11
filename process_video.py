"""
This is the main driver program that implements a pipeline to process images
in a video.
It reads the mp4 video file from argv[1], calls search_classify function
for each image, writes the image in a folder named by argv[2].
In the end it writes converts all the output images into a video named argv[1]-out.mp4

"""
from search_classify import search_classify, historic_persp
import cv2
import sys
import os
import shutil
from moviepy.editor import ImageSequenceClip
from sklearn.externals import joblib
from collections import deque

"Function to write list of image filenames to a mp4 outfile"
def makemovie(images, outfile):
  clip = ImageSequenceClip(images,fps=30)
  clip.write_videofile(outfile)

def load_model():
  model = 'hogGray-v1.pkl'
  if 'v1' in model:
    version = 'v1'
  else:
    version = 'v2'

  X_scaler, svc = joblib.load(model)
  return X_scaler, svc, version

def open_video(video):
  cap = cv2.VideoCapture(video)
  while not cap.isOpened():
      cap = cv2.VideoCapture(video)
      cv2.waitKey(1000)
      print("Wait for the header")
  print(cap.get(cv2.CAP_PROP_FRAME_COUNT),"frames found in",video)
  return cap

def get_frame(cap):
  frame_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
  while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        frame_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print(str(frame_position)+" frames")
        yield frame

    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position-1)
        print("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break

    #if(frame_position >= 10):
    #  break

def process_frame(frame, counter, hist_windows, X_scaler, svc, version):
  #history image to look at
  HIST=2
  #search_classify expects RGB
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  #first pass
  windows = all_sliding_windows(frame, max_image_y=340)
  on_windows, probs = search_classify(frame, windows, X_scaler, svc, version)
  #focused search
  on_windows = focused_windows(frame, on_windows)
  on_windows, probs = search_classify(frame, on_windows, X_scaler, svc, version)
  print("HIST",HIST)
  final_img, hist_windows = historic_persp(frame, on_windows, probs, hist_windows, counter, HIST)
  if(0):
    test_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    cv2.imshow('image',test_img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

  return final_img, hist_windows

def process_video(video, directory):
  X_scaler, svc, version = load_model()
  cap = open_video(video)
  hist_windows = deque()

  images = []
  counter = 0
  for frame in get_frame(cap):
    final_img, hist_windows = process_frame(frame, counter, hist_windows,
                                            X_scaler, svc, version)
    counter+=1
    images.append(final_img)
    if(0):
      test_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
      cv2.imwrite(directory+'/frame'+str(counter)+'.jpg',test_img)
  cap.release()
  return images

#rm -rf test_video/; python3 process_video.py project_video.mp4 test_video
if __name__ == '__main__':

  if(len(sys.argv) != 3):
    print("Usage:",sys.argv[0],"<mp4 file> <target dir>")
    exit(1)

  video = sys.argv[1]
  directory = sys.argv[2] #to put processed images
  os.mkdir(directory)
  images = process_video(video, directory)

  if(1):
    filename, file_extension = os.path.splitext(video)
    videoout = filename+'-out'+ ".mp4" #Output video filename
    makemovie(images, videoout)
    #shutil.rmtree(directory)
  print("All done!")
