import argparse
import os
import torch
from torch.utils.data import DataLoader

from datasets import NessieDataset, save_datasets, load_datasets
from utils.data_utils import *
from utils.parser_utils import str2bool

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessor. Load JSON file(s), do components split(in marginal distributions case), do train-val-test split and save datasets as .pt file(s).")
    parser.add_argument("--data", type=str, required=True, nargs="+", help="Dataset `.json` File Path. Can pass one or more JSON file paths or pass filefolder path which contains all JSON fils")
    parser.add_argument("--save", type=str, required=True, help="Dataset `.pt` File path")
    parser.add_argument("--rate", type=int, nargs=3, default=[0.7, 0.2, 0.1], help="Train-Val-Test Split Rate")
    parser.add_argument("--split", type=str2bool, default=False, help="Split a Multi Component Dataset to Single Component Datasets")

    return parser.parse_args()

def save_merged_data(json_files:list, save_path:str)->str:
    print('Merging Data...')
    merged_result = merge_json_files(json_files)
    merged_path = f"{os.path.splitext(save_path)[0]}_merged.json"
    with open(merged_path, 'w', encoding='utf-8') as output_file:
        json.dump(merged_result, output_file, ensure_ascii=False, indent=4)
    print(f"Data has been merged and saved to {merged_path}")
    return merged_path


def main(data_path, save_path, rate, split):
    if len(data_path) == 1:
        data_path = data_path[0]       
        path_flag = check_path(path=data_path, extension=".json")
        if path_flag == 0:
            raise ValueError(f"Unrecognizable path. Folder or file doesn't exist or extension doesn't match.")
        if path_flag == 1:
            pass
        if path_flag == 2:
            print(f'Folder path detected. All Json File in folder `{data_path}` will be loaded.')
            json_files = get_files_with_extensions(data_path, extension=".json")
            print(f"Total {len(json_files)} files: {json_files}")
            data_path = save_merged_data(json_files, save_path)

    elif len(data_path) >= 2:
        print(f'Multiple paths detected. All Json File input will be loaded.')
        json_files = list(data_path)
        print(f"Total {len(json_files)} files: {json_files}")
        data_path = save_merged_data(json_files, save_path)
        

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