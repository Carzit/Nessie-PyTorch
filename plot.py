import os
import argparse
import torch
from matplotlib import pyplot as plt

def plot(results_path:str, index:int):
    results = torch.load(results_path)

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
    plt.savefig(os.path.join("infer", f"X={X}.png"))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #玄学指令，解决OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
plot(r"infer\out1_nohiddenlayer.pt", 105)
