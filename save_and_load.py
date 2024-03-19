import torch

def save_pt(model:torch.nn.Module, path:str)->None:
    if not path.endswith('.pt'):
        path += '.pt'
    torch.save(model.state_dict(), path)

def load_pt(model:torch.nn.Module, path:str)->torch.nn.Module:
    if not path.endswith('.pt'):
        path += '.pt'
    model.load_state_dict(torch.load(path))
    return model
