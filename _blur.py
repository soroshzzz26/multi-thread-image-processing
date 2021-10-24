import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time


def gaussian_kernel(k=3, sigma=1):
    a = np.zeros((k, k), dtype=np.float64)
    for x in range(k):
        for y in range(k):
            a[x][y] = math.exp(-(x*x + y*y)/(2*sigma)) / (math.sqrt(2 * math.pi) * sigma)
    return a



def getIndex(x, y, H, W):
    if x - H >= 0:
        x -= 2*(x - H) + 1
    if y - W >= 0:
        y -= 2*(y - W) + 1
    return abs(x), abs(y)


def blur(image, kernel):
    H, W, C = image.shape
    k = kernel.shape[0]//2
    # sum_ = np.sum(kernel)

    image_temp = np.zeros(image.shape, dtype=np.float64)
    
    for x in range(H):
        for y in range(W):
            
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):

                    x_new, y_new = getIndex(x+i, y+j, H, W)
                    p = kernel[i + k][j + k]
                    image_temp[x][y][0] += p * image[x_new][y_new][0]
                    image_temp[x][y][1] += p * image[x_new][y_new][1]
                    image_temp[x][y][2] += p * image[x_new][y_new][2]
                    
            # image_temp[x][y][0] /= sum_
            # image_temp[x][y][1] /= sum_
            # image_temp[x][y][2] /= sum_
    
    return image_temp


def run_blur(log, image):
  print("Process " + log + " started " + time.ctime(time.time()))
  # k, sigma = 3, 1
  # kernel = gaussian_kernel(k, sigma)
  # kernel = 1/16 * np.array([[1, 2, 1], 
  #                           [2, 4, 2], 
  #                           [1, 2, 1]])
  kernel =  1/256 * np.array([[1, 4, 6, 4, 1], 
                    [4 , 16 ,24 , 16 , 4], 
                    [6, 24, 36, 24, 6],
                    [4 , 16 ,24 , 16 , 4],
                    [1, 4, 6, 4, 1]])
  image_out = blur(image, kernel)
  # image_out = blur(image_out, kernel)
  # image_out = blur(image_out, kernel)
  cv2.imwrite('output/blured3.jpg', image_out)
  print("Process " + log + " ended " + time.ctime(time.time()))





