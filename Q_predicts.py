from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.poisson import Poisson
from torch.distributions.negative_binomial import NegativeBinomial


class Q_predict(nn.Module):
    '''
    估计分布的抽象类

    初始化时传入相应概率分布参数
    由神经网络预测的k个分布组合而成

    被调用时传入相应x值返回对应
    '''
    def __init__(self, model_out:torch.Tensor) -> None:# model_out shape:[batch_size, components_size, distribution_params_size]
        super(Q_predict, self).__init__()
        self.batch_size:int = model_out.shape[0]
        self.NBDistributionList = []
        self.wlist = []
        self.probslist = []
    
    def forward(self, X:torch.Tensor):#X shape=[batch_size, copy_size]
        X = X.unsqueeze(dim=-1)
        for i in range(self.batch_size):
            probs = (self.wlist[i] * torch.exp(self.NBDistributionList[i].log_prob(X[i]))) 
            #[[w1*NB1(x1), w2*NB2(x1), w3*NB3(x1), ...],
            # [w1*NB1(x2), w2*NB2(x2), w3*NB3(x2), ...],
            #  ...]

            #p1 = w1*NB1(x1) + w2*NB2(x1) + w3*NB3(x1) + ...
            #...
            #pn = w1*NB1(xn) + w2*NB2(xn) + w3*NB3(xn) + ...
            probs = probs.sum(dim=1) #probs shape=[copy_size] : [p1, p2, p3, ...]
            self.probslist.append(probs)
        return torch.stack(self.probslist, dim=0) #predict shape=[batch_size, copy_size]

class Q_predict_NegativeBinomial(Q_predict):
    '''
    估计分布
    由神经网络预测的k个负二项分布组合而成
    __init__输入形状:[batch_size, components_size, 3]
    __call__输入形状:[batch_size, copy_size]
    '''
    def __init__(self, model_out:torch.Tensor) -> None:
        super(Q_predict_NegativeBinomial, self).__init__(model_out)
        for wpr in model_out:# shape:[components_size, 3]
            w, p, r = wpr.transpose(0,1) # shape:[3]
            self.NBDistributionList.append(NegativeBinomial(r, p))
            self.wlist.append(w)

class Q_predict_Poisson(Q_predict):
    '''
    估计分布
    由神经网络预测的components_size个泊松分布组合而成
    __init__输入形状:[batch_size, components_size, 3]
    __call__输入形状:[batch_size, copy_size]
    '''
    def __init__(self, model_out:torch.Tensor) -> None:
        super(Q_predict_Poisson, self).__init__(model_out)
        for w_lamda in model_out:
            w, lamda = w_lamda.transpose(0,1) # shape:[2]
            self.NBDistributionList.append(Poisson(lamda))
            self.wlist.append(w)

class Q_predict_General(Q_predict):
    '''
    估计分布
    传入参数和对应分布
    适用于普遍分布类型
    '''
    def __init__(self, model_out:torch.Tensor, distribution:torch.distributions.distribution.Distribution) -> None:
        super(Q_predict_General, self).__init__(model_out)
        for w_others in model_out:
            w, *others = w_others.transpose(0,1)
            self.NBDistributionList.append(distribution(*others))
            self.wlist.append(w)