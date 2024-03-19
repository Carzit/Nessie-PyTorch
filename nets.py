import torch
import torch.nn as nn

def multiLinear(input_size:int, output_size:int, layers:int=1, nodes:list=None)->nn.Sequential:
    '''
    辅助快速定义模型线性层的功能函数
    n : 中间隐藏层的层数(原文为1)
    nodes : 自定义中间隐藏层的节点数
    '''
    if nodes is None:
        if layers == 1:
            return nn.Linear(input_size, output_size)
        else:
            raise RuntimeError('''Not Implement''')
    elif layers != len(nodes)+1:
        raise RuntimeError('''Wrong nodes imput''')
    else:
        if layers == 2:
            return nn.Sequential(nn.Linear(input_size, nodes[0]),
                                 nn.Linear(nodes[0], output_size))
        else:
            seq = [nn.Linear(input_size, nodes[0])]
            for i in range(len(nodes)):
                seq.append(nn.Linear(nodes[i], nodes[i+1]))
            seq.append(nn.Linear(nodes[-1], output_size))
            return nn.Sequential(*seq)

 
class NegativeBinomialNet(nn.Module):
    '''
    使用负二项分布拟合轨迹的神经网络,输出number of componets组负二项分布参数集
    '''

    def __init__(self, input_size, output_size, layers:int=1, nodes:list=None):
        '''
        output_size(k): 研究的这个过程中所有混合物成分的种类(number of componets)
        w : 每种成分所占比例
        p, r : 用于拟合负二项分布的参数
        '''
        super(NegativeBinomialNet, self).__init__()
        self.layer_w = multiLinear(input_size, output_size, layers, nodes)
        self.layer_p = multiLinear(input_size, output_size, layers, nodes)
        self.layer_r = multiLinear(input_size, output_size, layers, nodes)
        self.softmax_w = nn.Softmax(dim=1)
        self.sigmoid_p = nn.Sigmoid()
        self.relu_w = nn.ReLU()
        self.relu_p = nn.ReLU()
        self.relu_r = nn.ReLU()

    def forward(self, x):
        x = torch.log(x)
        w = self.relu_w(self.layer_w(x))
        w = torch.exp(w)
        w = self.softmax_w(w)
        p = self.relu_p(self.layer_p(x))
        p = self.sigmoid_p(p)
        r = self.relu_r(self.layer_r(x))
        r = torch.exp(r)
        out = torch.stack([w,p,r], dim=-1)

        return out
    
class PoissonNet(nn.Module):
    '''
    使用泊松分布拟合轨迹的神经网络,输出number of componets组负二项分布参数集
    '''
    def __init__(self, input_size, output_size, layers:int=1, nodes:list=None):
        '''
        output_size(k): 研究的这个过程中所有混合物成分的种类(number of componets)
        w : 每种成分所占比例
        lamda : 用于拟合负二项分布的参数
        '''
        super(PoissonNet, self).__init__()
        self.layer_w = multiLinear(input_size, output_size, layers, nodes)
        self.layer_lamda = multiLinear(input_size, output_size, layers, nodes)
        self.activation_w = nn.Softmax(dim=1)
        self.relu_w = nn.ReLU()
        self.relu_lamda = nn.ReLU()

    def forward(self, x):
        x = torch.log(x)
        w = self.relu_w(self.layer_w(x))
        w = torch.exp(w)
        w = self.activation_w(w)
        lamda = self.relu_lamda(self.layer_lamda(x))
        lamda = torch.exp(lamda)
        out = torch.stack([w,lamda], dim=-1)

        return out




