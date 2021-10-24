import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb_to_grayscale(img):
        grayImage = np.zeros(img.shape)
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        R = (R *.21)
        G = (G *.72)
        B = (B *.07)

        Avg = (R+G+B)
        grayImage = img.copy()

        for i in range(3):
           grayImage[:,:,i] = Avg
           
        return grayImage       

image = mpimg.imread("./assets/sorosh.jpg")   
grayImage = rgb_to_grayscale(image)  
plt.axis('off')
plt.savefig('./output/sorosh_output.jpg')