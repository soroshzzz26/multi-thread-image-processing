# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np

# image = np.array(mpimg.imread('./assets/sw.jpg'))
# image_temp = np.zeros(image.shape, dtype=np.float32)
# H, W, C = image.shape

# def conv2(kernel):
#   k = kernel.shape[0]
#   # k = K//2
#   # sum_ = sum(kernel)
#   # sum_ = 1
  
#   for x in range(H):
#     for y in range(W):
#       for i in range(-k/2, k/2 + 1):
#         for j in range(-k/2, k/2 + 1):
#             image_temp[x][y][0] = kernel[i][j] * image[x + i][0][ y + j][0]
#             image_temp[x][y][1] = kernel[i][j] * image[x + i][0][ y + j][1]
#             image_temp[x][y][2] = kernel[i][j] * image[x + i][0][ y + j][2]






import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = np.array(mpimg.imread('./assets/sw.jpg'), dtype=np.float64)
image /= 255
image_temp = np.zeros(image.shape, dtype=np.float64)
H, W, C = image.shape


def getIndex(x, y):
    if x - H >= 0:
        x -= 2*(x - H) + 1
    if y - W >= 0:
        y -= 2*(y - W) + 1
    return abs(x), abs(y)



def conv(kernel):
    K = kernel.shape[0]
    k = K//2
    # sum_ = np.sum(kernel)
    sum_ = 1.0
    for x in range(H):
        for y in range(W):
            
            r, g, b = 0.0, 0.0, 0.0;
            
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x_new, y_new = getIndex(x, y)
                    p = kernel[i + k][j + k]
                    r += p * image[x_new][y_new][0]
                    g += p * image[x_new][y_new][1]
                    b += p * image[x_new][y_new][2]
            r /= sum_
            g /= sum_
            b /= sum_
            image_temp[x][y] = [r, g, b]

# kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
conv(kernel)
image_temp = image_temp - np.min(image_temp)
image_temp = image_temp / np.max(image_temp)


# [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
mpimg.imsave('./out.png', image_temp, format='png')