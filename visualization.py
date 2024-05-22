import os
import argparse
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.parser_utils import str2bool
from utils.data_utils import get_filename
from utils.plot_utils import sample_distribution_plot, mean_var_plot

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Plot for Predict and Truth")

    parser.add_argument("--results", type=str, required=True, help="Results file (.pt) path")
    parser.add_argument("--index", type=int, nargs="+", default=None, help="Index or Index List. Default all.")
    parser.add_argument("--save_dir", type=str, default=None, help="File Folder to Save Plots")
    parser.add_argument("--mean_var", type=str2bool, default=True, help="Whether plot Mean-Var Match Plot")

    return parser.parse_args()

def main(results_path:str, index:list, save_dir:str, mvplot:bool):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
    # 玄学指令，解决OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    # Conda与Torch目录下均有libiomp5md.dll导致

    print("Loading Results...")
    results = torch.load(results_path)
    print("Successfully Loaded")

    if index is None:
        index = list(range(len(torch.load(results))))
    if save_dir is None:
        save_dir = os.path.join("infer", get_filename(results_path))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in tqdm(index):
        sample_distribution_plot(results, i, save_dir)
        
    if mvplot:
        mean_var_plot(results=results, save_dir=save_dir)

if __name__ == "__main__":
    args = parse_args()
    main(results_path=args.results, index=args.index, save_dir=args.save_dir, mvplot=args.mean_var)

# python plot.py --results "infer\out1_nohiddenlayer.pt" --index 1 2
