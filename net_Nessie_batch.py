from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.poisson import Poisson
from torch.distributions.negative_binomial import NegativeBinomial




class Q_predict_NegativeBinomial:
    '''
    估计分布
    由神经网络预测的k个负二项分布组合而成
    '''
    def __init__(self, model_out:torch.Tensor):
        '''
        model_out:[batch_size, components_size, 3]
        '''
        self.batch_size:int = model_out.shape[0]
        self.NBDistributionList = []
        self.wlist = []
        self.probslist = []

        for wpr in model_out:# shape:[components_size, 3]
            w, p, r = wpr.transpose(0,1) # shape:[3]
            self.NBDistributionList.append(NegativeBinomial(r, p))
            self.wlist.append(w)

    def __call__(self, X:torch.Tensor):#X shape=[batch_size, copy_size]
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

def multiLinear(input_size, output_size, n:int=1, nodes:list=None)->nn.Sequential:
    '''
    n : 中间隐藏层的层数(原文为1)
    nodes : 自定义中间隐藏层的节点数
    '''
    if nodes is None:
        if n == 1:
            return nn.Linear(input_size, output_size)
        elif n == 2:
            return nn.Sequential(nn.Linear(input_size, round((input_size + output_size) / 2)),
                                 nn.Linear(round((input_size + output_size) / 2), output_size))
        else:
            raise RuntimeError('''Not Implement''')
    elif n != len(nodes)+1:
        raise RuntimeError('''Wrong nodes imput''')
    else:
        seq = [nn.Linear(input_size, nodes[0])]
        for i in range(len(nodes)):
            seq.append(nn.Linear(nodes[i], nodes[i+1]))
        seq.append(nn.Linear(nodes[-1], output_size))
        return nn.Sequential(*seq)

 
class NegativeBinomialNet(nn.Module):

    def __init__(self, input_size, output_size, n:int=1, nodes:list=None):
        '''
        output_size(k): 研究的这个过程中所有混合物成分的种类(number of componets)
        w : 每种成分所占比例
        p, r : 用于拟合负二项分布的参数
        '''
        super(NegativeBinomialNet, self).__init__()
        self.layer_w = multiLinear(input_size, output_size, n, nodes)
        self.layer_p = multiLinear(input_size, output_size, n, nodes)
        self.layer_r = multiLinear(input_size, output_size, n, nodes)
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


class NessieKLDivLoss(nn.Module):

    def __init__(self, Q_predict):
        super(NessieKLDivLoss, self).__init__()
        self.Q_predict = Q_predict

    def forward(self, model_out:torch.Tensor, target:torch.Tensor):
        # predict : model outputs (args for distributions)
        # target : tensor([[number_of_copy, probility],[...,...],...])#target shape=[batch_size, copy_size, 2]
        q_predict = self.Q_predict(model_out)

        target_copynumber, target_probility = target.transpose(0,2).transpose(1,2)#[batch_size, copy_size]
        predict_probility = q_predict(target_copynumber)#shape=[batch_size, copy_size]

        loss = target_probility * (torch.log(target_probility) - torch.log(predict_probility))
        loss = loss.sum(dim=-1)#shape=[batch_size]
        loss = loss.sum()/q_predict.batch_size
        return loss

class NessieHellingerDistance(nn.Module):

    def __init__(self, Q_predict):
        super(NessieHellingerDistance, self).__init__()
        self.Q_predict = Q_predict

    def forward(self, model_out:torch.Tensor, target:torch.Tensor):
        # predict : model outputs (args for distributions)
        # Y target : tensor([[number_of_copy, probility],[...,...],...])
        # X [t, initial_values, other_variables...]
        q_predict = self.Q_predict(model_out)
        
        target_copynumber, target_probility = target.transpose(0,2).transpose(1,2)#[batch_size, copy_size]
        predict_probility = q_predict(target_copynumber)

        loss = torch.square(torch.sqrt(target_probility) - torch.sqrt(predict_probility))
        loss = torch.div(torch.sqrt(loss.sum(dim=-1)), torch.sqrt(torch.tensor(2.0)))
        loss = loss.sum()/q_predict.batch_size
        return loss

