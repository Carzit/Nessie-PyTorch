import argparse
import torch
from torch.utils.data import DataLoader

from datasets import NessieDataset, save_datasets, load_datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Split json Dataset and saved as pt File")

    parser.add_argument("--data", type=str, required=True, help="Dataset `.json` File Path")
    parser.add_argument("--save", type=str, required=True, help="Dataset `.pt` File path")
    parser.add_argument("--split", type=int, nargs=3, default=[0.7, 0.2, 0.1], help="Train-Val-Test Split Rate")

    return parser.parse_args()

def main(data_path, save_path, split):
    print(f"Load Dataset from \"{data_path}\"")
    total_len, file_path = save_datasets(data_path=data_path, save_path=save_path, save_format=".pt", split=split)
    print(f"Total Length: {total_len}  \nSplit Rate: {split}  \nSplit Length: {[total_len * i for i in split]}")
    print(f"Successfully Saved to \"{file_path}\"")


if __name__ == "__main__":
    args = parse_args()
    main(data_path=args.data, save_path=args.save, split=args.split)

# python preprocess.py --data data\data_ssa.json --save data\data_ssa.pt
# python preprocess.py --data data\data_train_example.json --save data\example --split 1 1 1