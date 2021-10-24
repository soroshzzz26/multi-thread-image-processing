import numpy as np
import cv2 
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb_to_grayscale(threadName, img ):
  print(threadName + " started " + time.ctime(time.time()))

  Avg = _getAvg(img, threadName)  
  grayImage = img.copy()

  for i in range(3):
    grayImage[:,:,i] = Avg
      
  print(threadName + " finished " + time.ctime(time.time()))  
  return grayImage  

def _getAvg(img, threadName):
  H, W, C = img.shape
  grayImage = np.zeros((H, W))

  for i in range(H):
    for j in range(W):
      grayImage[i][j] = img[i][j][0] * 0.21 + img[i][j][1] * 0.72 + img[i][j][2] * 0.07
      
  return grayImage    