import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from save_and_load import save_pt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(epoches:int, 
          optimizer:torch.optim.Optimizer, 
          model:torch.nn.Module, 
          loss_fn:torch.nn.Module, 
          train_generator:DataLoader, 
          val_generator:DataLoader,
          *,
          print_per_epoch:int=1,
          save_per_epoch:int=1,
          save_path:str=os.curdir,
          save_name:str="model",
          merge_val:bool=False,
          merge_epoches:int=None,
          device:torch.device=torch.device('cpu'))->torch.nn.Module:
    
    print('Training Start!')
    model = model.to(device=device)

    for epoch in range(epoches):
        
        model.train()
        for batch, (X, Y) in tqdm(enumerate(train_generator)):

            X = X.to(device=device)
            Y = Y.to(device=device)

            output = model(X)
            train_loss = loss_fn(output, Y)  

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch, (X, Y) in enumerate(val_generator):
                X = X.to(device=device)
                Y = Y.to(device=device)

                output = model(X)
                val_loss = loss_fn(output, Y)
                
        if (epoch+1) % print_per_epoch == 0:
            print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, epoches, train_loss.item(), val_loss.item()))
        if (epoch+1) % save_per_epoch == 0:
            model_name = f"{save_name}_epoch{epoch}"
            model_path = os.path.join(save_path, model_name)
            print(model_path)
            save_pt(model, model_path)


    if merge_val:
        if not merge_epoches:
            merge_epoches = epoches 

        print('Merging Validation Training...')

        for epoch in range(merge_epoches):
            model.train()
            for batch, (X, Y) in tqdm(enumerate(val_generator)):
                X = X.to(device=device)
                Y = Y.to(device=device)

                output = model(X)
                train_loss = loss_fn(output, Y)
                    
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            if (epoch+1) % print_per_epoch == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, merge_epoches, train_loss.item()))

    model_name = f"{save_name}_final"
    model_path = os.path.join(save_path, model_name)
    save_pt(model, model_path)
    
    return model

def test(model:torch.nn.Module, 
        loss_fn:torch.nn.Module, 
        test_generator:DataLoader,
        device:torch.device=torch.device('cpu'))->None:
    model = model.to(device=device)
    model.eval()

    total_inaccuracy = 0
    total_batch = 0
    for batch, (X, Y) in enumerate(test_generator):
        X = X.to(device=device)
        Y = Y.to(device=device)

        output = model(X)

        test_inaccuracy = loss_fn(output, Y)
        total_inaccuracy += test_inaccuracy.item()
        total_batch += 1
    print('Test Inaccuracy: {:.4f}'.format(total_inaccuracy/total_batch))
    
    
