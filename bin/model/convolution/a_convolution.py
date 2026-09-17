import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

def conv2d_single(image, kernel):
    #二维卷积（步长为1）
    h, w = image.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

#水平边缘检测卷积核
kernel = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]])

mnist = datasets.MNIST(root='./data', train=True, download=True)
X = mnist.data.numpy()[0].astype(np.float32) / 255.0 
conv_result = conv2d_single(X, kernel)

# 显示原图和卷积后的图
plt.figure()

plt.subplot(1,2,1)
plt.imshow(X, cmap='gray')
plt.title('raw')

plt.subplot(1,2,2)
plt.imshow(conv_result, cmap='gray')
plt.title('produced')

plt.show()
