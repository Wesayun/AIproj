import numpy as np


def max_pooling(image, pool_size=2, stride=2):
    h, w = image.shape
    output_h = (h - pool_size) // stride + 1
    output_w = (w - pool_size) // stride + 1
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            region = image[i*stride:i*stride+pool_size, 
                          j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)
    return output


def mean_pooling(image, pool_size=2, stride=2):
    h, w = image.shape
    output_h = (h - pool_size) // stride + 1
    output_w = (w - pool_size) // stride + 1
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            region = image[i*stride:i*stride+pool_size, 
                          j*stride:j*stride+pool_size]
            output[i, j] = np.mean(region)
    return output