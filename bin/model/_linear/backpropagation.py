import numpy as np
import sys
sys.path.append(r'D:\code\src\nnscratch\scalar2matrix')
import two_layer_nn as tln

def backward_pass(net, y_true):
    """
    对给定的网络和真实标签，执行反向传播
    返回所有参数的梯度
    """
    # 从网络缓存中取出中间结果
    x = net.cache['x']
    h = net.cache['h']
    h_relu = net.cache['h_relu']
    scores = net.cache['scores']
    
    # 1. 损失对得分的梯度
    exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    N = scores.shape[0]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), y_true] = 1
    dL_dscores = probs - one_hot
    
    # 2. 第二层梯度
    # h_relu @ W2 + b2 = scores
    dL_dW2 = h_relu.T @ dL_dscores
    dL_db2 = np.sum(dL_dscores, axis=0)
    
    # 3. 回传到 h_relu
    dL_dh_relu = dL_dscores @ net.layer2.W.T
    
    # 4. 通过 ReLU 导数
    dL_dh = dL_dh_relu * (h > 0)
    
    # 5. 第一层梯度
    # x @ W1 + b1 = h
    dL_dW1 = x.T @ dL_dh
    dL_db1 = np.sum(dL_dh, axis=0)
    
    return {
        'dW1': dL_dW1, 'db1': dL_db1,
        'dW2': dL_dW2, 'db2': dL_db2
    }