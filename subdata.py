import argparse
import os
import json
from typing import List, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import NessieDataset, split_dataset
from nets import NegativeBinomialNet
from Q_predicts import Q_predict_NegativeBinomial
from losses import NessieKLDivLoss, NessieHellingerDistance

from utils import training_board
from utils import save_and_load
from utils.data_utils import get_filename

# TODO : will soon be merged into preprocess.py using subparsers.

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Nessie Model")

    parser.add_argument("--datafile", type=str, required=True, help="Dataset `.json` File Path")
    parser.add_argument("--savefile", type=str, required=True, help="Save Path")
    parser.add_argument("--index", type=int, default=0, help="X index")
    parser.add_argument("--value", type=float, default=None, help="X index")

    return parser.parse_args()


def split_list_by_index_value(data_list:List[list], X_index:int, mode="value"):
    split_dict = {}
    for index, sublist in enumerate(data_list):
        key = sublist[X_index]
        if key not in split_dict:
            split_dict[key] = []
        if mode == "index":
            split_dict[key].append(index)
        elif mode == "value":
            split_dict[key].append(sublist)
    return split_dict

def xindice_subdata(load_path:str, save_path:str=None, X_index:int=0, X_value=None):
    print(f"Loading data from {load_path}...")
    with open(load_path, 'r') as file:
        data = json.load(file)
    print(f"Succcessfully loaded.")

    Xs = data['data_X']
    Ys = data["data_Y"]
    split_dict = split_list_by_index_value(Xs, X_index, mode="index")
    print(f"All values in X[{X_index}]: {list(split_dict.keys())}")

    if save_path is None:
        save_path = os.path.join("data", f"{get_filename(load_path)}_sub_index{X_index}_{X_value}.json")

    if X_value is not None:
        print(f"Create subdataset file which satisfy X[{X_index}]=={X_value}...")
        indices = split_dict[X_value]
        Xs_sub = [Xs[indice] for indice in indices]
        Ys_sub = [Ys[indice] for indice in indices]

        with open(save_path, 'w', encoding='utf-8') as output_file:
            json.dump({"data_X":Xs_sub, "data_Y":Ys_sub}, output_file, ensure_ascii=False, indent=4)
        print(f"Subdata saved to `{save_path}`")

if __name__ == "__main__":
    args = parse_args()
    xindice_subdata(load_path=args.datafile, save_path=args.savefile, X_index=args.index, X_value=args.value)