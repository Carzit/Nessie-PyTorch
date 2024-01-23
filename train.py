import torch
import torch.nn as nn
import torch.optim as optim
from net import *

## Demo

# 定义输入和输出的形状
n = 8
m = 3

input_size = n
output_size = m

# 示例输入和输出数据
input_data = torch.tensor([1.1,2,3,4,5,6,7,8])
output_data = torch.tensor([[1,0.5],[2,0.3],[3,0.1],[4,0.1]])


# 创建网络实例
net = NessieNet(input_size, output_size)
# 定义损失函数
loss_fn = NessieKLDivLoss()
# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)



# 训练网络
for epoch in range(100):
    # 前向传播
    output = net(input_data)
    
    # 计算损失
    loss = loss_fn(output, output_data)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 打印训练状态
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))

print('=====================================')

wpr_result = net(input_data)

q = Q_predict(wpr_result)
test = q(torch.tensor([3, 1, 2, 6]))
print('Result:', test)


dis = NessieHellingerDistance()
tt = dis(wpr_result, output_data)
print("HellingerDistance:", tt)




        