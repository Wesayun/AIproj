import numpy as np
class LinearLayer:
    def __init__(self, input_size, output_size):
        # 初始化权重
        # input_size 输入的特征数量
        # output_size 输出的类别数量
        self.W = (np.random.randn(input_size, output_size) * 0.001).astype(np.float32)
        self.b = np.zeros(output_size,dtype=np.float32)
    
    def forward(self, X):
        # X 是输入数组
        return X @ self.W + self.b
    
#layer = LinearLayer(784, 10)
#test_data = np.random.randn(5,784)
#print(layer.forward(test_data))
