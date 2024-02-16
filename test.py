import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from datasets import NessieDataset, split_dataset
from nets import NegativeBinomialNet
from Q_predicts import Q_predict_NegativeBinomial
from losses import NessieKLDivLoss, NessieHellingerDistance

## Demo

# 加载数据集
dataset = NessieDataset('data_train_example.json')
print(len(dataset))
train_set, val_set, test_set = split_dataset(dataset, [0.6, 0.2, 0.2])
print(train_set.get_shape())
