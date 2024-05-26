import os
import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm
from .save_and_load import save

def train(
        epoches:int, 
        optimizer:torch.optim.Optimizer,
        model:torch.nn.Module, 
        loss_fn:torch.nn.Module, 
        train_generator:DataLoader, 
        val_generator:DataLoader,
        *,
        lr_scheduler:torch.optim.lr_scheduler.LRScheduler=None,
        hparams:dict=None,
        log_dir:str = r"log",
        sample_per_batch:int=0,
        sample_fn=lambda b, x, y, out, tloss, opt: print(f"Model Out: {out}\nLoss: {tloss}"),
        print_per_epoch:int=1,
        save_per_epoch:int=1,
        save_dir:str=os.curdir,
        save_name:str="model",
        save_format:str="pt",
        device:torch.device=torch.device('cpu'))->torch.nn.Module:
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    writer = SummaryWriter(os.path.join(log_dir, "TRAIN"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    model = model.to(device=device)

    # Train
    for epoch in range(epoches):
        
        # Train one Epoch
        model.train()
        for batch, (X, Y) in enumerate(tqdm(train_generator)):

            X = X.to(device=device)
            Y = Y.to(device=device)

            output = model(X)
            train_loss = loss_fn(output, Y)  

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            if sample_per_batch:
                if batch % sample_per_batch == 0:
                    sample_fn(batch, X, Y, output, train_loss, optimizer)

        # Record Train Loss Scalar
        writer.add_scalar("Train Loss", train_loss.item(), epoch)

        # If hyper parameters passed, record it in hparams(no val).
        if hparams and not val_generator:
            writer.add_hparams(hparam_dict=hparams, metric_dict={"hparam/TrainLoss":train_loss.item()})
        
        # If validation datasets exisit, calculate val loss without recording grad.
        if val_generator:
            model.eval() # set eval mode to frozen layers like dropout
            with torch.no_grad(): 
                for batch, (X, Y) in enumerate(val_generator):
                    X = X.to(device=device)
                    Y = Y.to(device=device)
                    output = model(X)
                    val_loss = loss_fn(output, Y)
                
                writer.add_scalar("Validation Loss", val_loss.item(), epoch)
                writer.add_scalars("Train-Val Loss", {"Train Loss": train_loss.item(), "Validation Loss": val_loss.item()}, epoch)

        # If hyper parameters passed, record it in hparams.
        if hparams and val_generator:
            writer.add_hparams(hparam_dict=hparams, metric_dict={"hparam/TrainLoss":train_loss.item(), "hparam/ValLoss":val_loss.item()})

        # If learning rate scheduler exisit, update learning rate per epoch.
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
        if lr_scheduler:
            lr_scheduler.step()
        
        # Flushes the event file to disk
        writer.flush()

        # Specify print_per_epoch = 0 to unable print training information.
        if print_per_epoch:
            if (epoch+1) % print_per_epoch == 0:
                print('Epoch [{}/{}], Train Loss: {:.6f}, Validation Loss: {:.6f}'.format(epoch+1, epoches, train_loss.item(), val_loss.item()))
        
        # Specify save_per_epoch = 0 to unable save model. Only the final model will be saved.
        if save_per_epoch:
            if (epoch+1) % save_per_epoch == 0:
                model_name = f"{save_name}_epoch{epoch}"
                model_path = os.path.join(save_dir, model_name)
                print(model_path)
                save(model, model_path, save_format)
        
        
    writer.close()
    model_name = f"{save_name}_final"
    model_path = os.path.join(save_dir, model_name)
    save(model, model_path, save_format)
    
    return model

@torch.no_grad()
def test(model:torch.nn.Module, 
        loss_fn:torch.nn.Module, 
        test_generator:DataLoader,
        *,
        log_dir:str = "log",
        device:torch.device=torch.device('cpu'))->None:
    
    writer = SummaryWriter(os.path.join(log_dir, "TEST"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    model = model.to(device=device)
    model.eval()

    total_inaccuracy = 0
    total_batch = 0
    for batch, (X, Y) in enumerate(tqdm(test_generator)):
        X = X.to(device=device)
        Y = Y.to(device=device)

        output = model(X)

        test_inaccuracy = loss_fn(output, Y)
        total_inaccuracy += test_inaccuracy.item()
        total_batch += 1

        writer.add_scalar("Criterion per Batch", test_inaccuracy.item(), batch)
        writer.add_scalar("Criterion Average", total_inaccuracy/total_batch, batch)

    print('Test Inaccuracy: {:.6f}'.format(total_inaccuracy/total_batch))