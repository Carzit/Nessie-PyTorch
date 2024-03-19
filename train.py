import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import NessieDataset, split_dataset
from nets import NegativeBinomialNet
from Q_predicts import Q_predict_NegativeBinomial
from losses import NessieKLDivLoss, NessieHellingerDistance
from training import train

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
input_size = 7
output_size = 4
hidden_size = 128

# 设置训练参数
max_epochs = 20
batch_size = 5 #10
learning_rate = 0.01
patience = 25

# 创建网络实例
net = NegativeBinomialNet(input_size=input_size, output_size=output_size, layers=2, nodes=[hidden_size])

# 选取拟合分布
Q_predict = Q_predict_NegativeBinomial
# 定义损失函数
loss_fn = NessieKLDivLoss(Q_predict, need_softmax=True)
# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# 定义偏差检验
inaccuracy_fn = NessieHellingerDistance(Q_predict, need_softmax=True)

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

model = train(epoches=20, 
                    optimizer=optimizer, 
                    model=net, 
                    loss_fn=loss_fn, 
                    train_generator=train_generator, 
                    val_generator=val_generator,
                    print_per_epoch=1,
                    save_per_epoch=1,
                    save_path=".\\save",
                    save_name="model", 
                    merge_val=True, 
                    merge_epoches=10,
                    device=torch.device('cuda'))











        