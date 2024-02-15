import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import NessieDataset
from nets import NegativeBinomialNet
from Q_predicts import Q_predict_NegativeBinomial
from losses import NessieKLDivLoss, NessieHellingerDistance

## Demo

# 加载数据集
trainset = NessieDataset('data_train_example.json')

# 定义输入和输出的形状
input_size, output_size = trainset.get_shape()
output_size = 6

# 创建网络实例
net = NegativeBinomialNet(input_size, output_size)

# 选取拟合分布
Q_predict = Q_predict_NegativeBinomial
# 定义损失函数
loss_fn = NessieKLDivLoss(Q_predict)
# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.001)
# 定义偏差检验
inaccuracy_fn = NessieHellingerDistance(Q_predict)

# 设置训练参数
max_epochs = 20
batch_size = 10

# 数据生成器
training_generator = DataLoader(trainset, batch_size=batch_size ,shuffle=True)

# 训练网络
for epoch in range(max_epochs):

    for batch, (X, Y) in enumerate(training_generator):

        # 前向传播
        output = net(X)
        # 计算损失
        loss = loss_fn(output, Y)
            
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 打印训练状态
    
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, max_epochs, loss.item()))






        