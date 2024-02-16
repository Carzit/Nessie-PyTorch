import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import NessieDataset, split_dataset
from nets import NegativeBinomialNet
from Q_predicts import Q_predict_NegativeBinomial
from losses import NessieKLDivLoss, NessieHellingerDistance

## Demo

'''
We use a training set of size 1k, a validation set of size 100 and a test set of size 500, sampled using a Sobol
sequence in the parameter region indicated in Table S1. For each datapoint, we take four snapshots at
times t = {5;10;25;100} and construct the corresponding histograms using the FSP. 

Our neural network consists of a single hidden layer with 128 neurons and outputs 4 negative binomial mixture components; 
we use a batch size of 64 for training. 

Learning Rate:we usually initialize lr=0.01 and decrease it astraining progress. 
Namely, we periodically monitor the loss function over the validation dataset and halve lr
if the loss has improved by less than 0.5% over the last 25 epochs on average. 

Stopping Criterion: The training procedure is terminated after lr has been decreased 5 times, which usually
indicates that optimization has stalled. 
'''

# 加载数据集
print('Loading Dataset...')
dataset = NessieDataset('data_train_example.json')
train_set, val_set, test_set = split_dataset(dataset, [0.7, 0.2, 0.1])
print('Successfully loaded!')

# 定义输入和输出的形状
x_shape, y_shape = dataset.get_shape()
input_size = x_shape
output_size = 4
hidden_size = 128

# 设置训练参数
max_epochs = 200
batch_size = 5 #10
learning_rate = 0.01
patience = 25

# 创建网络实例
net = NegativeBinomialNet(input_size=input_size, output_size=output_size, layers=2, nodes=[hidden_size])

# 选取拟合分布
Q_predict = Q_predict_NegativeBinomial
# 定义损失函数
loss_fn = NessieKLDivLoss(Q_predict)
# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# 定义偏差检验
inaccuracy_fn = NessieHellingerDistance(Q_predict)

# 数据生成器
train_generator = DataLoader(train_set, batch_size=batch_size ,shuffle=True)
val_generator = DataLoader(val_set, batch_size=batch_size ,shuffle=True)
test_generator = DataLoader(test_set, batch_size=batch_size ,shuffle=True)

# 训练网络
print(f'''Configs:
    Data: {len(dataset)}
    Epoch Number: {max_epochs}
    Batch Size: {batch_size}
    Input Feature Size: {input_size}
    Hidden Size(Neurons): {hidden_size}
    Output Size(Number of Components): {output_size}
    Initial Learning Rate: {learning_rate}
    ''')

print('Training Start!')
val_losses = []
lr_decrease_times = 0
for epoch in range(max_epochs):

    net.train()
    for batch, (X, Y) in enumerate(train_generator):
        # 前向传播
        output = net(X)
        # 计算损失
        train_loss = loss_fn(output, Y)
            
        # 反向传播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    net.eval()
    for batch, (X, Y) in enumerate(val_generator):
        # 前向传播
        output = net(X)
        # 计算损失
        val_loss = loss_fn(output, Y)
    
    if patience <= 0:
        avg_val_loss = sum(val_losses[-25:]) /25
        if val_loss.item() > 0.995 * avg_val_loss:

            print('decrease lr')

            learning_rate /= 2
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
            patience = 5
            lr_decrease_times += 1
            if lr_decrease_times == 5:
                break

    val_losses.append(val_loss.item())
    patience -= 1

    print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, max_epochs, train_loss.item(), val_loss.item()))

print('Merging Validation Training...')

for epoch in range(10):
    net.train()
    for batch, (X, Y) in enumerate(val_generator):
        # 前向传播
        output = net(X)
        # 计算损失
        train_loss = loss_fn(output, Y)
            
        # 反向传播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, max_epochs, train_loss.item()))

print('Test')

net.eval()
total_inaccuracy = 0
total_batch = 0
for batch, (X, Y) in enumerate(test_generator):
    # 前向传播
    output = net(X)
    # 计算损失
    test_inaccuracy = inaccuracy_fn(output, Y)
    total_inaccuracy += test_inaccuracy.item()
    total_batch += 1

print('Test Inaccuracy: {:.4f}'.format(total_inaccuracy/total_batch))

## 保存模型
PATH = "Nessie_V1.pt"
torch.save(net.state_dict(), PATH)









        