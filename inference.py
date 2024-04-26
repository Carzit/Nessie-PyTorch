import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from datasets import NessieDataset, load_datasets
from nets import NegativeBinomialNet
from Q_predicts import Q_predict, Q_predict_NegativeBinomial
from losses import NessieLoss

from utils.save_and_load import save, load


class NessieInfer(NessieLoss):
    def __init__(self, Q_predict:Q_predict, need_softmax:bool=True):
        super(NessieInfer, self).__init__(Q_predict, need_softmax)

    def forward(self, model_out:torch.Tensor, target:torch.Tensor):
        super(NessieInfer, self).forward(model_out, target)
        return self.predict_probility, self.target_probility

@torch.no_grad()
def infer(dataset:Dataset, model:nn.Module, q_predict:nn.Module, device:torch.device=torch.device('cpu')):
    results = []
    dataloader = DataLoader(dataset, batch_size=1 ,shuffle=False)
    model = model.to(device=device)

    model.eval()
    for batch, (X, Y) in enumerate(tqdm(dataloader)):
        X = X.to(device=device)
        Y = Y.to(device=device)     
        model_out = model(X)
        infer = NessieInfer(q_predict, False)
        Y_predict, Y_true = infer(model_out, Y)

        results.append((X[0], Y_true[0], Y_predict[0]))

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Nessie Model Inference")

    parser.add_argument("--dataset", type=str, required=True, help="dataset file path")
    parser.add_argument("--model", type=str, required=True, help="model file path")

    parser.add_argument("--input_size", type=int, required=True, help="input size of model")
    parser.add_argument("--output_size", type=int, required=True, help="ouput size of modle")
    parser.add_argument("--hidden_size", type=int, nargs="*", default=None, help="hidden size of model")

    parser.add_argument("--save_path", type=str, default="model", help="results to save")
    parser.add_argument("--use_cuda", type=bool, default=torch.cuda.is_available(), help="Whether use CUDA. If CUDA is avaiable, default is True.")

    return parser.parse_args()


def main(data_path, model_path, input_size, output_size, hidden_size, save_path, use_cuda)->None:
    print('Loading Dataset...')
    test_set = load_datasets(data_path)["test_set"]
    print('Successfully loaded!')

    model = NegativeBinomialNet(input_size=input_size, output_size=output_size, nodes=hidden_size)
    load(model=model, path=model_path)

    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    results = infer(test_set, model, Q_predict_NegativeBinomial, device)

    torch.save(results, save_path)

if __name__ == "__main__":
    args = parse_args()
    main(data_path=args.dataset, model_path=args.model, input_size=args.input_size, output_size=args.output_size, hidden_size=args.hidden_size, save_path=args.save_path, use_cuda=args.use_cuda)

# python inference.py --dataset "data\data_ssa.pt" --model "save\model_final" --input_size 5 --output_size 4 --save_path "infer\out1_nohiddenlayer.pt" --use_cuda True












