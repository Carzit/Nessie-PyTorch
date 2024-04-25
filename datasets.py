import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class NessieDataset(Dataset):

    def __init__(self, json_path):
        super(NessieDataset, self).__init__()

        with open(json_path, 'r') as file:
            data = json.load(file)
            self.Xs = data['data_X']
            self.Ys = data["data_Y"]


    def __len__(self):
        return min(len(self.Xs),len(self.Ys))

    def __getitem__(self, index):
        X = torch.tensor(self.Xs[index])
        Y = torch.tensor(self.Ys[index])
        return X, Y
    
    def get_shape(self):
        return torch.tensor(self.Xs[0]).shape, torch.tensor(self.Ys[0]).shape
    
    def split_dataset(self, lengths:list):
        total_length = len(self)
        split_lengths = list(map(lambda x:round(x / sum(lengths) * total_length), lengths))
        split_lengths[0] = total_length - sum(split_lengths[1:])
        return random_split(self, split_lengths)
    

def split_dataset(dataset:Dataset, lengths:list):
    total_length = len(dataset)
    split_lengths = list(map(lambda x:round(x / sum(lengths) * total_length), lengths))
    split_lengths[0] = total_length - sum(split_lengths[1:])
    return random_split(dataset, split_lengths)
 
