import os
import argparse
import torch
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List, Tuple

def compute_mean_and_variance(prob_tensor:torch.Tensor)->Tuple[int]:
    n = len(prob_tensor)
    values = torch.arange(n).to(prob_tensor.device)
    mean = torch.sum(values * prob_tensor)
    variance = torch.sum((values - mean) ** 2 * prob_tensor)
    
    return mean.item(), variance.item()

def sample_distribution_plot(results:list, index:int, save_dir:str):
    X, Y_true, Y_predict = results[index]
    X = X.tolist()
    Y_true = Y_true.tolist()
    Y_predict = Y_predict.tolist()
    plt.subplot(1, 2, 1)
    plt.ylim(0, 0.3)
    plt.bar(range(len(Y_true)), Y_true)
    plt.title("Truth")

    plt.subplot(1, 2, 2)
    plt.ylim(0, 0.3)
    plt.bar(range(len(Y_predict)), Y_predict)
    plt.title("Predict")

    plt.suptitle(f"X={X}")
    plt.savefig(os.path.join(save_dir, f"{index}{X}.png"))

def mean_var_plot(results:List[Tuple[torch.Tensor]], save_dir:str, name:str="Mean_and_Var"):
    Y_true_means = []
    Y_true_vars = []
    Y_predict_means = []
    Y_predict_vars = []

    for index in tqdm(range(len(results))):
        _, Y_true, Y_predict = results[index]
        
        Y_true_mean, Y_true_var = compute_mean_and_variance(Y_true)
        Y_predict_mean, Y_predict_var = compute_mean_and_variance(Y_predict) 

        Y_true_means.append(Y_true_mean)
        Y_true_vars.append(Y_true_var)
        Y_predict_means.append(Y_predict_mean)
        Y_predict_vars.append(Y_predict_var)

    fig=plt.figure(figsize=(12,6))
    ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.8])
    ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.8])

    ax1.scatter(Y_predict_means, Y_true_means, color='blue', marker='o', s=0.5)  # s 参数用于设置点的大小
    ax1.set_title("Mean")
    ax1.set_xlabel("Nessie")
    ax1.set_ylabel("Simulation")
    ax1.set_xlim(0, 80)
    ax1.set_ylim(0, 80)
    
    ax2.scatter(Y_predict_vars, Y_true_vars, color='blue', marker='o', s=0.5)
    ax2.set_title("Var")
    ax2.set_xlabel("Nessie")
    ax2.set_ylabel("Simulation")
    ax2.set_xlim(0, 1600)
    ax2.set_ylim(0, 1600)

    plt.savefig(os.path.join(save_dir, f"{name}.png"))