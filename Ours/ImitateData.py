import torch

def load_ft(data_dir, feature_name):
    return torch.randn(28,1024)*3.0,torch.randn(500,1),0,0