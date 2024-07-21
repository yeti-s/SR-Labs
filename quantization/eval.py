import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from util import calc_psnr
from dataset import TestDataset

@torch.no_grad()
def evaluate(model:nn.Module, data_loader:DataLoader, scale:int, upscale:int, steps:int=0) -> float:
    """_summary_

    Args:
        model (nn.Module): _description_
        data_loader (DataLoader): _description_
        scale (int): _description_
        upscale (int): _description_
        steps (int, optional): _description_. Defaults to 0.

    Returns:
        float: _description_
    """
    model.eval()
    total_psnr = 0
    device = next(model.parameters()).device
    steps = len(data_loader) if steps == 0 else steps
    
    for step, inputs in enumerate(data_loader):
        if step >= steps:
            break
        
        hr = inputs[0].to(device)
        lr = inputs[1].to(device)
        sr = model(lr, scale, upscale)
        psnr = calc_psnr(sr, hr, scale, 1, benchmark=True)
        total_psnr += psnr

    return total_psnr / steps