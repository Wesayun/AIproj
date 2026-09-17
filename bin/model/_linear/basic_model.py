import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import two_layer_nn as tln
import loss
from torchvision import datasets

import backpropagation as back
import evaluate_accuracy as acc


mnist = datasets.MNIST(root='./data', train=True, download=True)
X = mnist.data.numpy()[:].astype(np.float32) / np.float32(255.0)    # 60000张训练图片
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
exam2_list = acc.exam2(the_net)
print(f'测试集准确率为{exam2_list[3]}')
ex2_scores = exam2_list[0]
ex2_pred = exam2_list[1]
ex2_y_np = exam2_list[2]

#用户验证

while True:
    try:
        exam_type = int(input('请输入你想验证的类型：自己绘画数字请输入1，调用预设测试集图片请输入2,退出请输入3:'))
    except ValueError:
        print('只能输入1,2,3请重新输入')
        continue
    if exam_type not in (1,2,3):
        print('只能输入1,2,3请重新输入')
        continue
    if exam_type == 1:
        print('关闭窗口以返回')
        while True:
            exam1_list = acc.exam1(the_net)
            if exam1_list == None:
                print('已返回上层')
                break
            ex1_scores = exam1_list[0]
            ex1_pred = exam1_list[1]
            print(f'模型判断该图片类别是{ex1_pred[0]}')

            plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
            plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
            plt.bar(range(10), ex1_scores[0])
            plt.xlabel('数字类别')
            plt.ylabel('得分')
            plt.title('模型对这张图片的预测得分')
            plt.show()

    if exam_type == 2:
        while True:
            try:
                test_pic = int(input('请输入你想查看准确率的照片,用0-9999的整数表示，输入-1返回'))
            except ValueError:
                print('只能输入整数')
                continue
            if test_pic < 0 or test_pic > 9999:
                break
            print(f'这张图片的真实类别是{ex2_y_np[test_pic]}')
            print(f'模型判断该图片类别是{ex2_pred[test_pic]}')

            plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
            plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
            plt.bar(range(10), ex2_scores[test_pic])
            plt.xlabel('数字类别')
            plt.ylabel('得分')
            plt.title('模型对这张图片的预测得分')
            plt.show()
    
    if exam_type == 3:
        break

print('模型已关闭，感谢您的测试')