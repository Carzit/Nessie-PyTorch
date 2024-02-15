import json
import torch
from torch.utils.data import Dataset, DataLoader

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
        X_shape = torch.tensor(self.Xs[0]).shape
        Y_shape = torch.tensor(self.Ys[0]).shape

        return X_shape[0], Y_shape[0]
    
