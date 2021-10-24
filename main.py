from _grayScale import rgb_to_grayscale
from _edgeDetection import edgeDetect 
from _blur import run_blur
import matplotlib.image as mpimg
import threading
import time
import cv2 
# import os

image = cv2.imread("./assets/sorosh.jpg")   
threads=[]

x=threading.Thread(target=edgeDetect, args=("Thread edgeDetection",image))
threads.append(x)
x.start()

x=threading.Thread(target=rgb_to_grayscale, args=("Thread GrayScale", image))
threads.append(x)
x.start()

x=threading.Thread(target=run_blur, args=("Thread Blur", image))
threads.append(x)
x.start()

for thread in threads :
  thread.join()