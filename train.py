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

## Demo

'''
We use a training set of size 1k, a validation set of size 100 and a test set of size 500, sampled using a Sobol
sequence in the parameter region indicated in Table S1. For each datapoint, we take four snapshots at
times t = {5;10;25;100} and construct the corresponding histograms using the FSP. 

Our neural network consists of a single hidden layer with 128 neurons and outputs 4 negative binomial mixture components; 
we use a batch size of 64 for training. 

Learning Rate:we usually initialize lr=0.01 and decrease it astraining progress. 
Namely, we periodically monitor the loss function over the validation dataset and halve lr
if the loss has improved by less than 0.5% over the last 25 epochs on average. 

Stopping Criterion: The training procedure is terminated after lr has been decreased 5 times, which usually
indicates that optimization has stalled. 
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Train Nessie Model")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset `.pt` File Path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size of Dataloader")
    parser.add_argument("--shuffle", type=bool, default=True, help="Whether shuffle dataset; Default True")

    parser.add_argument("--input_size", type=int, required=True, help="Input Size of Model")
    parser.add_argument("--output_size", type=int, required=True, help="Ouput Size of Model")
    parser.add_argument("--hidden_size", type=int, nargs="*", default=None, help="Hidden Size of Model. Specify none or one or more numbers.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load Checkpoint Model")

    parser.add_argument("--max_epoches", type=int, required=True, help="Max Train Epoches")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial Learning Rate")
    parser.add_argument("--use_cuda", type=bool, default=torch.cuda.is_available(), help="Whether use CUDA. If CUDA is avaiable, default is True.")

    parser.add_argument("--sample_batch", type=int, default=0, help="Sample Every n Batch. Print Model Out and Batch Loss")

    parser.add_argument("--model_name", type=str, default="model", help="Model Name to Save; Default is `model`")

    return parser.parse_args()

def main(data_path, batch_size, shuffle, input_size, output_size, hidden_size, checkpoint, max_epoches, learning_rate, use_cuda, sample_batch, model_name)->None:
    # 加载数据集
    print(f"Loading Train and Validation Dataset from {data_path}")
    train_set = load_datasets(data_path)["train_set"]
    val_set = load_datasets(data_path)["val_set"]
    print('Successfully Loaded')

    # 创建网络实例
    net = NegativeBinomialNet(input_size=input_size, output_size=output_size, nodes=hidden_size if hidden_size is None else list(hidden_size))
    if not checkpoint is None:
        save_and_load.load(net, checkpoint, "pt")

    # 选取拟合分布
    Q_predict = Q_predict_NegativeBinomial
    # 定义损失函数
    loss_fn = NessieKLDivLoss(Q_predict, need_relu=True)
    # 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # 数据生成器
    train_generator = DataLoader(train_set, batch_size=batch_size ,shuffle=shuffle)
    val_generator = DataLoader(val_set, batch_size=batch_size ,shuffle=shuffle)

    # 设备
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # 训练网络
    print(f'''Configs:
Train Data Length: {len(train_set)}
Val Data Length: {len(val_set)}
Max Epoch: {max_epoches}
Batch Size: {batch_size}
Input Size: {input_size}
Hidden Size(Neurons): {hidden_size}
Output Size(Number of Components): {output_size}
Initial Learning Rate: {learning_rate}''')

    model = training_board.train(epoches=max_epoches,optimizer=optimizer,model=net,loss_fn=loss_fn,train_generator=train_generator, val_generator=val_generator, sample_per_batch=sample_batch, save_dir=r"save", save_name=model_name, save_format="pt" ,device=device)

if __name__ == "__main__":
    args = parse_args()
    main(data_path=args.dataset, 
         batch_size=args.batch_size, 
         shuffle=args.shuffle, 
         input_size=args.input_size, 
         output_size=args.output_size, 
         hidden_size=args.hidden_size, 
         checkpoint=args.checkpoint, 
         max_epoches=args.max_epoches, 
         learning_rate=args.learning_rate, 
         use_cuda=args.use_cuda, 
         sample_batch=args.sample_batch, 
         model_name=args.model_name)

# python train.py --dataset "data\data_ssa.pt" --batch_size 1 --shuffle True --input_size 5 --output_size 4 --hidden_size 128 --max_epoches 40 --learning_rate 0.01 --use_cuda True --sample_batch 5000 --model_name Model_5-128-4_nosoftmax

# python train.py --dataset "data\data_ssa.pt" --batch_size 1 --shuffle True --input_size 5 --output_size 4 --max_epoches 20 --learning_rate 0.01 --use_cuda True --model_name Model2

# python train.py --dataset "data\example.pt" --batch_size 4 --shuffle True --input_size 7 --output_size 4 --hidden_size 32 64 128 64 32 16 --max_epoches 20 --learning_rate 0.01 --use_cuda True --model_name Model_example

# python train.py --dataset "data\data_ssa.pt" --batch_size 1 --shuffle True --input_size 5 --output_size 16 --max_epoches 20 --learning_rate 0.001 --use_cuda True --model_name Model3_5-32-64-128-64-32-adam --sample_batch 5000 --hidden_size 32 64 128 64 32

# python train.py --dataset "data\data_ssa.pt" --batch_size 1 --shuffle True --input_size 5 --output_size 4 --max_epoches 20 --learning_rate 0.001 --use_cuda True --model_name Model4_5-4-adam --sample_batch 5000 

# python train.py --dataset "data\data_ssa.pt" --batch_size 1 --shuffle True --input_size 5 --output_size 4 --max_epoches 20 --learning_rate 0.0005 --use_cuda True --model_name Model5_5-4-adam --sample_batch 5000 --checkpoint save\Model4_5-4-adam_epoch9.pt

# python train.py --dataset "data\data_ssa.pt" --batch_size 1 --shuffle True --input_size 5 --output_size 4 --hidden_size 128 --max_epoches 20 --learning_rate 0.0005 --use_cuda True --model_name Model8_5-128-4-adam-reluloss --sample_batch 5000













        