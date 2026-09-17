import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

i = int(input())
mnist = datasets.MNIST(root='./data', train=True, download=True)
X = mnist.data.numpy()[i].astype(np.float32) / 255.0 
y = mnist.targets[i].long() 

plt.imshow(X, cmap='gray')
plt.title(f"Label: {y}")
plt.show()
print(f"图片的形状是: {X.shape}")