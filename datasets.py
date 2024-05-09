import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from safetensors.torch import save_file, load_file

class NessieDataset(Dataset):

    def __init__(self, json_path, component_index=None):
        super(NessieDataset, self).__init__()
        self.component_index = component_index

        with open(json_path, 'r') as file:
            data = json.load(file)
            self.Xs = data['data_X']
            self.Ys = data["data_Y"]

    def __len__(self):
        return min(len(self.Xs),len(self.Ys))

    def __getitem__(self, index):
        X = torch.tensor(self.Xs[index])
        Y = torch.tensor(self.Ys[index])
        if not self.component_index is None:
            Y_split = torch.unbind(Y, dim=1)
            Y = torch.stack([Y_split[0], Y_split[self.component_index]], dim=1)
        return X, Y
    
    def info(self, index=0)->tuple:
        return len(self), self[index][0].shape, self[index][1].shape
    
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

def save_datasets(data_path:str, save_path:str, save_format:str=".pt", split:list=[0.7, 0.2, 0.1], component_index=None)->tuple:
    dataset = NessieDataset(data_path, component_index)
    train_set, val_set, test_set = split_dataset(dataset, split)
    save_sets = {"train_set":train_set,
                 "val_set":val_set,
                 "test_set":test_set}
    
    if save_format == ".pt":
        if not save_path.endswith(".pt"):
            save_path = save_path + ".pt"
        torch.save(save_sets, save_path)
    
    if save_format == ".safetensors":
        if not save_path.endswith(".safetensors"):
            save_path = save_path + ".safetensors"
        save_file(save_sets, save_path)
        
    return dataset.info(), save_path

def load_datasets(file_path:str)->dict:
    if file_path.endswith(".pt"):
        save_sets = torch.load(file_path)
        
    if file_path.endswith(".safetensors"):
        save_sets = load_file(file_path)

    return save_sets 
 
