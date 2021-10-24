# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
import numpy as np
import cv2 
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

image = cv2.imread("./assets/sw.jpg") 
image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
output = convolve2D(image, kernel, padding=2) 
cv2.imwrite('edgedetected.jpg', output)






# def gaussia_kernel(k=3, sigma=1):
#     a = np.zeros((k, k), dtype=np.float64)
#     for x in range(k):
#         for y in range(k):
#             a[x][y] = math.exp(-(x*x + y*y)/(2*sigma)) / (math.sqrt(2 * math.pi) * sigma)
#     return a




# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np

# image = np.array(mpimg.imread('./assets/sw.jpg'), dtype=np.float64)
# image /= 255
# image_temp = np.zeros(image.shape, dtype=np.float64)
# H, W, C = image.shape


# def getIndex(x, y):
#     if x - H >= 0:
#         x -= 2*(x - H) + 1
#     if y - W >= 0:
#         y -= 2*(y - W) + 1
#     return abs(x), abs(y)



# def conv(kernel):
#     K = kernel.shape[0]
#     k = K//2
#     # sum_ = np.sum(kernel)
#     sum_ = 1.0
#     for x in range(H):
#         for y in range(W):
            
#             r, g, b = 0.0, 0.0, 0.0;
            
#             for i in range(-k, k + 1):
#                 for j in range(-k, k + 1):
#                     x_new, y_new = getIndex(x, y)
#                     p = kernel[i + k][j + k]
#                     r += p * image[x_new][y_new][0]
#                     g += p * image[x_new][y_new][1]
#                     b += p * image[x_new][y_new][2]
#             r /= sum_
#             g /= sum_
#             b /= sum_
#             image_temp[x][y] = [r, g, b]

# # kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
# kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
# conv(kernel)
# # image_temp = image_temp - np.min(image_temp)
# image_temp = np.abs(image_temp)
# image_temp = image_temp / np.max(image_temp)


# # [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
# mpimg.imsave('./out.png', image_temp, format='png')