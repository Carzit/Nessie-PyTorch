import os
import argparse
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

def plot(results:list, index:int, save_dir:str):
    X, Y_true, Y_predict = results[index]
    X = X.tolist()
    Y_true = Y_true.tolist()
    Y_predict = Y_predict.tolist()

    plt.subplot(1, 2, 1)
    plt.bar(range(len(Y_true)), Y_true)
    plt.title("Truth")

    plt.subplot(1, 2, 2)
    plt.bar(range(len(Y_predict)), Y_predict)
    plt.title("Predict")

    plt.suptitle(f"X={X}")
    plt.savefig(os.path.join(save_dir, f"{index}{X}.png"))

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Plot for Predict and Truth")

    parser.add_argument("--results", type=str, required=True, help="Results file (.pt) path")
    parser.add_argument("--index", type=int, nargs="+", default=None, help="Index or Index List. Default all.")
    parser.add_argument("--save_dir", type=str, default=None, help="File Folder to Save Plots")

    return parser.parse_args()

def main(results_path:str, index:list, save_dir:str):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
    # 玄学指令，解决OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    # Conda与Torch目录下均有libiomp5md.dll导致

    print("Loading Results...")
    results = torch.load(results_path)
    print("Successfully Loaded")

    if index is None:
        index = list(range(len(torch.load(results))))
    if save_dir is None:
        save_dir = os.path.join("infer",os.path.splitext(os.path.basename(results_path))[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in tqdm(index):
        plot(results, i, save_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args.results, args.index, args.save_dir)

# python plot.py --results "infer\out1_nohiddenlayer.pt" --index 1 2
