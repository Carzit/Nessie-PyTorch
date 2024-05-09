import argparse
import os
import torch
from torch.utils.data import DataLoader

from datasets import NessieDataset, save_datasets, load_datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Split json Dataset and saved as pt File")

    parser.add_argument("--data", type=str, required=True, help="Dataset `.json` File Path")
    parser.add_argument("--save", type=str, required=True, help="Dataset `.pt` File path")
    parser.add_argument("--rate", type=int, nargs=3, default=[0.7, 0.2, 0.1], help="Train-Val-Test Split Rate")
    parser.add_argument("--split", type=bool, default=False, help="Split a Multi Component Dataset to Single Component Datasets")

    return parser.parse_args()

def main(data_path, save_path, rate, split):
    print(f"Load Dataset from \"{data_path}\"")    
    info, file_path = save_datasets(data_path=data_path, save_path=save_path, save_format=".pt", split=rate)
    total_len = info[0]
    input_size = info[1][0]
    copy_size = info[2][0]
    components = info[2][1]
    print(f"Total Length: {total_len}  \nSplit Rate: {rate}  \nSplit Length: {[total_len * i for i in rate]} \nInput Size: {input_size}")
    print(f"Dataset Saved to \"{file_path}\"")
    if split:
        save_path = os.path.splitext(save_path)[0]
        for i in range(1, components):
            _, file_path = save_datasets(data_path=data_path, save_path=save_path+f"_{i}", save_format=".pt", split=rate, component_index=i)
            print(f"Component {i} Dataset Saved to \"{file_path}\"")



if __name__ == "__main__":
    args = parse_args()
    main(data_path=args.data, save_path=args.save, rate=args.rate, split=args.split)

# python preprocess.py --data data\data_ssa.json --save data\data_ssa.pt
# python preprocess.py --data data\data_train_example.json --save data\example --rate 1 1 1
# python preprocess.py --data data\data_dssa.json --save data\data_dssa --rate 1 1 1 --split True