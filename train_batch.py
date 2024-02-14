import torch
import torch.nn as nn
import torch.optim as optim
from net_Nessie_batch import *
from dataset import *

## Demo

# 加载数据集
trainset = NessieDataset('data.json')

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
max_epochs = 100
batch_size = 2

# 数据生成器
training_generator = DataLoader(trainset, batch_size=batch_size ,shuffle=True)

# 训练网络
for epoch in range(max_epochs):

    for batch, (X, Y) in enumerate(training_generator):
        print('X:',X)
        print('Y:',Y)

        # 前向传播
        output = net(X)
        print('model_out:',output)
        # 计算损失
        loss = loss_fn(output, Y)
        print('loss:',loss)
            
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 打印训练状态
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))

print('=====================================')

'''
wpr_result = net(input_data)
q = Q_predict(wpr_result)
test = q(torch.tensor([5,8,9,14]))
print('Result:', test)

print("HellingerDistance:", inaccuracy_fn(wpr_result, output_data))
'''




        