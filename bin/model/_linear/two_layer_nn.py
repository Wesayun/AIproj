import numpy as np
import fully_connected_nn as f_c

class two_layer_nn:
    def __init__(self):
        self.layer1 = f_c.LinearLayer(784,64)
        self.layer2 = f_c.LinearLayer(64,10)
        self.cache = {}
    def forward(self,x):
        h = self.layer1.forward(x)
        h_relu = np.maximum(0,h)
        scores = self.layer2.forward(h_relu)
        self.cache['x'] = x
        self.cache['h'] = h
        self.cache['h_relu'] = h_relu
        self.cache['scores'] = scores
        return(scores)

x = np.random.randn(1, 784)  # 模拟1张图片
a_net = two_layer_nn()
scores = a_net.forward(x)
print(scores.shape)  # 应该是 (1, 10)
