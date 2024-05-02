from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.poisson import Poisson
from torch.distributions.negative_binomial import NegativeBinomial


class Q_predict(nn.Module):
    '''
    估计分布的基类

    初始化时传入相应概率分布参数
    由神经网络预测的k个分布组合而成

    被调用时传入相应x值返回对应
    '''
    def __init__(self, model_out:torch.Tensor) -> None:# model_out shape:[batch_size, components_size, distribution_params_size]
        super(Q_predict, self).__init__()
        self.batch_size: int = model_out.shape[0]
        self.component_size: int = model_out.shape[1]
        self.copy_size: int
        self.w: torch.Tensor
        self.distribution: torch.distributions.distribution.Distribution
    
    def forward(self, X:torch.Tensor):#X shape=[batch_size, copy_size]
        self.copy_size = X.shape[1]
        self.distribution = self.distribution.expand([self.copy_size, self.batch_size, self.component_size])#[copy_size,batch_size, component_size]
        self.probs = torch.exp(self.distribution.log_prob(X.transpose(0,1).unsqueeze(-1))).permute(1,2,0)#[batch_size, component_size, copy_size]
        self.w = self.w.unsqueeze(-1).expand(self.batch_size, self.component_size, self.copy_size)
        return (self.w * self.probs).sum(1) #->[batch_size, copy_size]


class Q_predict_NegativeBinomial(Q_predict):
    '''
    估计分布
    由神经网络预测的k个负二项分布组合而成
    __init__输入形状:[batch_size, components_size, 3]
    __call__输入形状:[batch_size, copy_size]
    '''
    def __init__(self, model_out:torch.Tensor) -> None:
        super(Q_predict_NegativeBinomial, self).__init__(model_out)
        self.w, p, r = model_out.permute((2,0,1))
        self.distribution = NegativeBinomial(r, p)


class Q_predict_Poisson(Q_predict):
    '''
    估计分布
    由神经网络预测的components_size个泊松分布组合而成
    __init__输入形状:[batch_size, components_size, 2]
    __call__输入形状:[batch_size, copy_size]
    '''
    def __init__(self, model_out:torch.Tensor) -> None:
        super(Q_predict_Poisson, self).__init__(model_out)
        self.w, lamda = model_out.permute((2,0,1))
        self.distribution = Poisson(lamda)


class Q_predict_General(Q_predict):
    '''
    估计分布
    传入参数和对应分布
    适用于普遍分布类型
    '''
    def __init__(self, model_out:torch.Tensor, distribution:torch.distributions.distribution.Distribution) -> None:
        super(Q_predict_General, self).__init__(model_out)
        self.w, *others = model_out.permute((2,0,1))
        self.distribution = distribution