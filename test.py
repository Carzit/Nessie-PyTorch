import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import NessieDataset, save_datasets, load_datasets
from nets import NegativeBinomialNet
from Q_predicts import Q_predict_NegativeBinomial
from losses import NessieKLDivLoss, NessieHellingerDistance

from utils import training_board, save_and_load



def parse_args():
    parser = argparse.ArgumentParser(description="Train Nessie Model")

    parser.add_argument("--dataset", type=str, default=None, help="Dataset `.pt` File Path")
    parser.add_argument("--model", type=str, default=None, help="Model `.pt` File Path")

    parser.add_argument("--input_size", type=int, required=True, help="Input Size of Model")
    parser.add_argument("--output_size", type=int, required=True, help="Ouput Size of Model")
    parser.add_argument("--hidden_size", type=int, nargs="*", default=None, help="Hidden Size of Model. Specify none or one or more numbers.")

    parser.add_argument("--use_cuda", type=bool, default=torch.cuda.is_available(), help="Whether use CUDA. If CUDA is avaiable, default is True.")

    return parser.parse_args()

def main(data_path, input_size, output_size, hidden_size, model_path, use_cuda)->None:
    print(f"Loading Test Dataset from {data_path}")
    test_set = load_datasets(data_path)["test_set"]
    print('Successfully Loaded')
    
    # 创建网络实例
    net = NegativeBinomialNet(input_size=input_size, output_size=output_size, nodes=hidden_size if hidden_size is None else list(hidden_size))
    save_and_load.load(net, model_path, "pt")

    # 选取拟合分布
    Q_predict = Q_predict_NegativeBinomial
    # 定义损失函数
    loss_fn = NessieKLDivLoss(Q_predict, need_relu=True)

    # 数据生成器
    test_generator = DataLoader(test_set, batch_size=1, shuffle=True)

    # 设备
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    training_board.test(model=net, loss_fn=loss_fn, test_generator=test_generator, device=device)

if __name__ == "__main__":
    args = parse_args()
    main(data_path=args.dataset, input_size=args.input_size, output_size=args.output_size, hidden_size=args.hidden_size, model_path=args.model, use_cuda=args.use_cuda)

# python test.py --dataset "data\data_ssa.pt" --input_size 5 --output_size 4 --hidden_size --model "save\Model8_5-128-4-adam-reluloss_final.pt" --use_cuda True













        