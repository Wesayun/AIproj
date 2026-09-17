import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from torchvision import datasets

def exam(net):
    mnist = datasets.MNIST(root='./data', train=False, download=True)
    X = mnist.data.numpy()[:].astype(np.float32) / 255.0    # 10000张图片
    y = mnist.targets[:].long()           # 10000个标签
    X_flat = X.reshape(X.shape[0], -1) 

    scores = net.forward(X_flat)
    pred = np.argmax(scores,axis=1) 
    y_np = y.numpy()
    accuracy = np.mean(pred == y_np)
    return [scores,pred,y_np,accuracy]