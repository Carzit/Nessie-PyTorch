import torch
import torch.nn as nn
from Q_predicts import Q_predict

class NessieLoss(nn.Module):
    def __init__(self, Q_predict:Q_predict, need_relu:bool=False, need_softmax:bool=False):
        super(NessieLoss, self).__init__()
        self.Q_predict = Q_predict
        self.need_relu = need_relu
        self.need_softmax = need_softmax

    def forward(self, model_out:torch.Tensor, target:torch.Tensor):
        # predict : model outputs (args for distributions)
        # target : tensor([[number_of_copy, probility],[...,...],...])#target shape=[batch_size, copy_size, 2]
        self.q_predict = self.Q_predict(model_out)

        self.target_copynumber, self.target_probility = target.permute(2, 0, 1)#[batch_size, copy_size]

        self.predict_probility = self.q_predict(self.target_copynumber)#shape=[batch_size, copy_size]

        if self.need_relu:# relu the target probility (and add 1e-4) to process some kind of negative or zero values
            self.target_probility = nn.functional.relu(self.target_probility)+0.0001
            self.predict_probility = nn.functional.relu(self.predict_probility)+0.0001

        if self.need_softmax:# softmax the target probility to process some kind of negative values
            self.target_probility = nn.functional.softmax(self.target_probility, dim=1)
            self.predict_probility = nn.functional.softmax(self.predict_probility, dim=1)

class NessieKLDivLoss(NessieLoss):
    def __init__(self, Q_predict:Q_predict, need_relu:bool=False, need_softmax:bool=False):
        super(NessieKLDivLoss, self).__init__(Q_predict, need_relu, need_softmax)

    def forward(self, model_out:torch.Tensor, target:torch.Tensor):
        super(NessieKLDivLoss, self).forward(model_out, target)
        
        loss = self.target_probility * (torch.log(self.target_probility) - torch.log(self.predict_probility))
        loss = loss.sum(dim=-1)#shape=[batch_size]
        loss = loss.mean()

        return loss

class NessieHellingerDistance(NessieLoss):
    def __init__(self, Q_predict:Q_predict, need_relu:bool=False, need_softmax:bool=False):
        super(NessieHellingerDistance, self).__init__(Q_predict, need_relu, need_softmax)

    def forward(self, model_out:torch.Tensor, target:torch.Tensor):
        super(NessieHellingerDistance, self).forward(model_out, target)
        loss = torch.square(torch.sqrt(self.target_probility) - torch.sqrt(self.predict_probility))
        loss = torch.div(torch.sqrt(loss.sum(dim=-1)), torch.sqrt(torch.tensor(2.0)))
        loss = loss.sum(dim=-1)#shape=[batch_size]
        loss = loss.mean()
        return loss
    
class NessieLossGeneral(NessieLoss):
    def __init__(self, Q_predict:Q_predict, need_relu:bool=False, need_softmax:bool=False):
        super(NessieLossGeneral, self).__init__(Q_predict, need_relu, need_softmax)

    def forward(self, model_out:torch.Tensor, target:torch.Tensor, loss_func):
        super(NessieLossGeneral, self).forward(model_out, target)
        loss = loss_func(self.target_probility, self.predict_probility)
        loss = loss.sum(dim=-1)#shape=[batch_size]
        loss = loss.mean()
        return loss