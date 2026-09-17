import numpy as np
X = np.random.randn(5, 784)
W = np.random.randn(784, 10)
b = np.random.randn(10)
#得分大小为可信度的度量
scores = X @ W + b
print(scores.shape)