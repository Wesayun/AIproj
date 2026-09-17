import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import two_layer_nn as tln
import loss
from torchvision import datasets

import backpropagation as back
import evaluate_accuracy as acc

mnist = datasets.MNIST(root='./data', train=True, download=True)
X = mnist.data.numpy()[:].astype(np.float32) / 255.0    # 60000张训练图片
y = mnist.targets[:].long()           # 60000个训练标签
X_flat = X.reshape(X.shape[0], -1) 

the_net = tln.two_layer_nn()

lr = 0.00001

try:
    epochs = int(input('请输入训练轮数：'))
except ValueError:
    print('输入的不是正整数')

losses = []

for epoch in range(epochs):
    
    scores = the_net.forward(X_flat)

    a_loss = np.mean(loss.compute_loss(scores,y))
    print(a_loss)
    losses.append(a_loss)

    grads = back.backward_pass(the_net,y)

    the_net.layer1.W -= lr * grads['dW1']
    the_net.layer1.b -= lr * grads['db1']
    the_net.layer2.W -= lr * grads['dW2']
    the_net.layer2.b -= lr * grads['db2']

    print(f"Epoch {epoch+1}/{epochs}, Loss: {a_loss:.4f}")

print("训练完成！")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(epochs),losses,label='loss_curve')
plt.show()

# 最后测试
exam_list = acc.exam(the_net)
print(f'测试集准确率为{exam_list[3]}')
ex_scores = exam_list[0]
ex_pred = exam_list[1]
ex_y_np = exam_list[2]

while True:
    try:
        test_pic = int(input('请输入你想查看准确率的照片,用0-9999的整数表示，输入-1终止'))
    except ValueError:
        break
    if test_pic < 0 or test_pic > 9999:
        break
    print(f'这张图片的真实类别是{ex_y_np[test_pic]}')
    print(f'模型判断该图片类别是{ex_pred[test_pic]}')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    plt.bar(range(10), ex_scores[test_pic])
    plt.xlabel('数字类别')
    plt.ylabel('得分')
    plt.title('模型对这张图片的预测得分')
    plt.show()