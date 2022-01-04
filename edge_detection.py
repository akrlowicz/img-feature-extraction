import numpy as np
import cv2

def apply_sobel(images):
  sobel_features = [cv2.Sobel(src=im, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=7).flatten() for im in images]

  return sobel_features

#other we can try Canny 