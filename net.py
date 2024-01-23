from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.negative_binomial import NegativeBinomial

class Q_predict:
    def __init__(self, wpr):
        w, p, r = wpr
        self.rs = r.unsqueeze(dim=0).view(-1,1)
        self.ps = p.unsqueeze(dim=0).view(-1,1)
        self.ws = w.unsqueeze(dim=0).view(-1,1)
        self.NBDistribution = NegativeBinomial(self.rs, self.ps)

    def __call__(self, x):
        probs = (self.ws * torch.exp(self.NBDistribution.log_prob(x))).sum(dim=0)
        return probs


class NessieNet(nn.Module):
    def __init__(self, input_size, output_size):
        '''
        output_size(k): 研究的这个过程中所有混合物成分的种类(number of componets)
        w : 每种成分所占比例
        p, r : 用于拟合负二项分布的参数
        '''
        super(NessieNet, self).__init__()
        self.layer_w = nn.Linear(input_size, output_size)
        self.layer_p = nn.Linear(input_size, output_size)
        self.layer_r = nn.Linear(input_size, output_size)
        self.activation_w = nn.Softmax(dim=0)
        self.activation_p = nn.Sigmoid()
        self.relu_w = nn.ReLU()
        self.relu_p = nn.ReLU()
        self.relu_r = nn.ReLU()

    def forward(self, x):
        x = torch.log(x)

        w = self.relu_w(self.layer_w(x))
        w = torch.exp(w)
        w = self.activation_w(w)

        p = self.relu_p(self.layer_p(x))
        p = self.activation_p(p)

        r = self.relu_r(self.layer_r(x))
        r = torch.exp(r)

        out = torch.stack([w,p,r])

        return out


    
class NessieKLDivLoss(nn.Module):
    def __init__(self):
        super(NessieKLDivLoss, self).__init__()

    def forward(self, predict, target):
        # target : tensor([[number_of_copy, probility],[...,...],...])
        target_copynumber, target_probility = target.transpose(0,1)
        q_predict = Q_predict(predict)
        loss = target_probility * (torch.log(target_probility) - torch.log(q_predict(target_copynumber)))
        loss = loss.sum()

        return loss

class NessieHellingerDistance(nn.Module):
    def __init__(self):
        super(NessieHellingerDistance, self).__init__()

    def forward(self, predict, target):
        # target : tensor([[number_of_copy, probility],[...,...],...])
        target_copynumber, target_probility = target.transpose(0,1)
        q_predict = Q_predict(predict)
        predict_probility = q_predict(target_copynumber)
        loss = torch.square(torch.sqrt(target_probility) - torch.sqrt(predict_probility))

        loss = torch.div(torch.sqrt(loss.sum()), torch.sqrt(torch.tensor(2.0)))

        return loss

