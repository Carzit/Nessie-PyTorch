import torch
import torch.nn as nn

from typing import List, Callable





def multiLinear(input_size:int, output_size:int, nodes:List[int]=None, activation:nn.Module=None)->nn.Sequential:
    '''
    辅助快速定义模型线性层的功能函数
    n : 中间隐藏层的层数(原文为1)
    nodes : 自定义中间隐藏层的节点数
    '''
    if nodes is None:
        return nn.Linear(input_size, output_size)
    else:
        if len(nodes) == 1:
            if activation is None:
                return nn.Sequential(nn.Linear(input_size, nodes[0]),
                                    nn.Linear(nodes[0], output_size))
            else:
                return nn.Sequential(nn.Linear(input_size, nodes[0]),
                                    activation(),
                                    nn.Linear(nodes[0], output_size),
                                    activation())
        else:
            if activation is None:
                seq = [nn.Linear(input_size, nodes[0])]
                for i in range(len(nodes)-1):
                    seq.append(nn.Linear(nodes[i], nodes[i+1]))
                seq.append(nn.Linear(nodes[-1], output_size))
            else:
                seq = [nn.Linear(input_size, nodes[0]), activation()]
                for i in range(len(nodes)-1):
                    seq.append(nn.Linear(nodes[i], nodes[i+1]))
                    seq.append(activation())
                seq.append(nn.Linear(nodes[-1], output_size))
                seq.append(activation())
            return nn.Sequential(*seq)
    
class Log(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Log, self).__init__(*args, **kwargs)
    def forward(self, x):
        return torch.log(x)

class Exp(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Exp, self).__init__(*args, **kwargs)
    def forward(self, x):
        return torch.exp(x)
 
class NegativeBinomialNet(nn.Module):
    '''
    使用负二项分布拟合轨迹的神经网络,输出number of componets组负二项分布参数集
    '''

    def __init__(self, input_size, output_size, nodes:List[int]=None):
        '''
        output_size(k): 研究的这个过程中所有混合物成分的种类(number of componets)
        w : 每种成分所占比例
        p, r : 用于拟合负二项分布的参数
        '''
        super(NegativeBinomialNet, self).__init__()
        self.layer_w = multiLinear(input_size, output_size, nodes, None)
        self.layer_p = multiLinear(input_size, output_size, nodes, nn.ReLU)
        self.layer_r = multiLinear(input_size, output_size, nodes, nn.ReLU)
        self.softmax_w = nn.Softmax(dim=1)
        self.sigmoid_p = nn.Sigmoid()
        self.trans_log = Log()
        self.trans_exp = Exp()
        

    def forward(self, x):
        x = self.trans_log(x)
        w = self.layer_w(x)
        w = self.trans_exp(w)
        w = self.softmax_w(w)
        p = self.layer_p(x)
        p = self.sigmoid_p(p) * 0.999
        r = self.layer_r(x)
        r = self.trans_exp(r)
        out = torch.stack([w,p,r], dim=-1)

        return out
    
    '''
    估计分布
    由神经网络预测的components_size个二元正态分布组合而成
    __init__输入形状:[batch_size, components_size, 6], 最后1维上的顺序为权重w, x1均值, x2均值, x1方差, x2方差, x1和x2的协方差
    __call__输入形状:[batch_size, copy_size], 其中 copy_size=copy_size_per_variate^variate_size
    '''

class MultivariateNormal2DNet(nn.Module):
    '''
    使用负二项分布拟合轨迹的神经网络,输出number of componets组负二项分布参数集
    '''

    def __init__(self, input_size, output_size, nodes:List[int]=None):
        '''
        output_size(k): 研究的这个过程中所有混合物成分的种类(number of componets)
        w : 每种成分所占比例
        x1均值, x2均值, x1方差, x2方差, x1和x2的协方差
        '''
        super(MultivariateNormal2DNet, self).__init__()
        self.layer_w = multiLinear(input_size, output_size, nodes, None)
        self.layer_mean1 = multiLinear(input_size, output_size, nodes, None)
        self.layer_mean2 = multiLinear(input_size, output_size, nodes, None)
        self.layer_sd1 = multiLinear(input_size, output_size, nodes, nn.ReLU)
        self.layer_sd2 = multiLinear(input_size, output_size, nodes, nn.ReLU)
        self.layer_corr = multiLinear(input_size, output_size, nodes, None)
        self.softmax_w = nn.Softmax(dim=1)
        self.tanh_corr = nn.Tanh()
        self.trans_log = Log()
        self.trans_exp = Exp()

    def forward(self, x):
        x_log = self.trans_log(x)

        w = self.layer_w(x_log)
        w = self.trans_exp(w)
        w = self.softmax_w(w)

        mean1 = self.layer_mean1(x)
        mean2 = self.layer_mean2(x)

        sd1 = self.layer_sd1(x_log)
        sd1 = self.trans_exp(sd1)
        sd2 = self.layer_sd2(x_log)
        sd2 = self.trans_exp(sd2)

        corr = self.layer_corr(x)
        corr = self.tanh_corr(corr)

        out = torch.stack([w, mean1, mean2, sd1, sd2, corr], dim=-1)

        return out


class PoissonNet(nn.Module):
    '''
    使用泊松分布拟合轨迹的神经网络,输出number of componets组负二项分布参数集
    '''
    def __init__(self, input_size, output_size, nodes:List[int]=None):
        '''
        output_size(k): 研究的这个过程中所有混合物成分的种类(number of componets)
        w : 每种成分所占比例
        lamda : 用于拟合负二项分布的参数
        '''
        super(PoissonNet, self).__init__()
        self.layer_w = multiLinear(input_size, output_size, nodes, None)
        self.layer_lamda = multiLinear(input_size, output_size, nodes, nn.ReLU)
        self.softmax_w = nn.Softmax(dim=1)
        self.trans_log = Log()
        self.trans_exp = Exp()

    def forward(self, x):
        x = self.trans_log(x)
        w = self.layer_w(x)
        w = self.trans_exp(w)
        w = self.softmax_w(w)
        lamda = self.layer_lamda(x)
        lamda = self.trans_exp(lamda)
        out = torch.stack([w,lamda], dim=-1)

        return out



