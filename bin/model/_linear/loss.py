import numpy as np
def compute_loss(scores,y_true):
    scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
    #这一步是防止样本量过大，数据大小溢出，所以减去最大值
    p = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted), axis=1, keepdims=True)
    #注意这里是数组运算，keepdims是保持维度参数方便后续广播相除
    n = p.shape[0]  
    #p的行数
    loss = -np.log(p[np.arange(n),y_true]+ 1e-8)
    #这里使用高级索引，也就是数组逐元素配对，同时加上一个极小值，防止log0报错
    return loss
    #此处输出的是每张图片的损失，实际应用可取平均输出标量