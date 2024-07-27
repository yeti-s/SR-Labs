# implementation of https://arxiv.org/abs/2406.06649

import os
import sys
import random
from tqdm import tqdm
from functools import partial
from datetime import datetime


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
# from torch._prims_common import DeviceLikeType
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR



from OverNet import LDGs
# from dataset import TrainDataset
from quantization.ops import fake_quantize
from quantization.eval import evaluate, infer, TestDataset

DEFAULT_QUANT_BIT = 8
DEFAULT_BATCH_SIZE = 4

class Quant2d(nn.Module):
    def __init__(self, conv2d:nn.Conv2d, boundaries:dict, n_bits:int) -> None:
        super().__init__()
        
        self.register_buffer('weight', conv2d.weight.data.cpu())
        self.register_buffer('bias', conv2d.bias.data.cpu())

        if boundaries['is_zero_base']:
            # no lower boundary update when activation minimum is zero
            self.register_buffer('l_a', boundaries['l_a'])
        else:
            self.l_a = nn.Parameter(boundaries['l_a'])
        self.l_w = nn.Parameter(boundaries['l_w'])
        self.u_w = nn.Parameter(boundaries['u_w'])
        self.l_a = nn.Parameter(boundaries['l_a'])
        self.u_a = nn.Parameter(boundaries['u_a'])
        
        self.n_bits = n_bits
        self.stride = conv2d.stride
        self.padding = conv2d.padding
        self.dilation = conv2d.dilation
        self.groups = conv2d.groups
    
    def forward(self, x:Tensor) -> Tensor:
        q_w = fake_quantize(self.weight, self.l_w, self.u_w, self.n_bits)
        q_x = fake_quantize(x, self.l_a, self.u_a, self.n_bits)
        return F.conv2d(q_x, q_w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def calibrate_dobi(tensor:Tensor, K:int, n_bits:int) -> tuple[Tensor, Tensor, bool]:
    """Distribution-Oriented Bound Initialization

    Args:
        tensor (Tensor): tensor to find lower, upper bound
        K (int): the number of search point
        n_bits (int, optional): the number of quantization bits. Defaults is 8.

    Returns:
        tuple[Tensor, Tensor, bool]: per channel lower, upper bound tensors, and is zero base
    """
    tensor = torch.flatten(tensor, 1)
    l = torch.min(tensor, dim=1).values.unsqueeze(-1)
    u = torch.max(tensor, dim=1).values.unsqueeze(-1)
    is_zero_base = l.min().item() == 0. # is tensor symmetric distribution?
    delta = (u - l) / (2 * K)
    
    min_mse = sys.maxsize
    best_l = l
    best_u = u
    
    for i in range(K):
        if is_zero_base:
            l_i = l
        else: 
            l_i = l + i * delta
        u_i = u + i * delta
        q_w = fake_quantize(tensor, l_i, u_i, n_bits)
        mse = F.mse_loss(tensor, q_w).item()

        if mse < min_mse:
            best_l = l_i
            best_u = u_i
            min_mse = mse
    
    return best_l, best_u, is_zero_base

def get_boundaries(model:nn.Module, calib_loader:DataLoader, scale:int, upscale:int, n_bits:int, K:int=100) -> dict:
    """Get boundary using calibration dataset

    Args:
        model (nn.Module): target model
        calib_loader (str): calibration dataset loader
        scale (int): scale
        upscale (int): upscale
        n_bits (int): n bits quantization
        K (int, optional): the number of search point. default is 100
        
    Returns:
        dict: {
            'l_w': lower boundary for weight (per channel),
            'u_w': upper boundary for weight (per channel),
            'l_a': lower bounader for activation (per channel),
            'u_a': upper boundary for activation (per channel),
            'is_zero_base': minimum of tensor is zero
        }
    """
    activations = {}
    boundaries = {}
    hooks = []
    
    def func_hook(name, module, x, y):
        if name not in activations:
            activations[name] = []
        activations[name].append(x[0].detach())
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            l_w, u_w, _ = calibrate_dobi(module.weight.data, K, n_bits)
            l_w = l_w.view(-1, 1, 1, 1).cpu()
            u_w = u_w.view(-1, 1, 1, 1).cpu()
            boundaries[name] = {
                'l_w': l_w,
                'u_w': u_w
            }
            module.register_forward_hook(partial(func_hook, name))
        
    infer(model, calib_loader, 1)
        
    for name, tensors in activations.items():
        tensor = torch.cat(tensors).transpose(0, 1)
        l_a, u_a, is_zero_base = calibrate_dobi(tensor, K, n_bits)
        l_a = l_a.view(1, -1, 1, 1)
        u_a = u_a.view(1, -1, 1, 1)
        boundaries[name]['l_a'] = l_a.cpu()
        boundaries[name]['u_a'] = u_a.cpu()
        boundaries[name]['is_zero_base'] = is_zero_base
        
        activations[name] = []
        torch.cuda.empty_cache()
    
    for hook in hooks:
        hook.remove()
        
    return boundaries

def quantize_dobi(model:nn.Module, boundaries:dict, name:str, n_bits:int) -> None:
    """Quantize every Conv2d model into Quant2d

    Args:
        model (nn.Module): target model to quantize
        boundaries (dict): boundary info of weights and activations
        name (str): module name
        n_bits (int): n bits quantization
    """
    for module_name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(model, module_name, Quant2d(module, boundaries['.'.join([name, module_name])[1:]], n_bits))
        else:
            quantize_dobi(module, boundaries, '.'.join([name, module_name]), n_bits)
            

def distilate(
    teacher:nn.Module, 
    student:nn.Module, 
    calib_loader:DataLoader,
    val_loader:DataLoader,
    coeff=10000, 
    epochs=3000
) -> None:
    """Fine-tune cliping range

    Args:
        teacher (nn.Module): original model
        student (nn.Module): dobi quantized model
        calib_loader (DataLoader): calibration data loader
        val_loader (DataLoader): validation data loader
        coeff (int, optional): co-efficient of output and feature loss. defaults to 5.
        epochs (int, optional): training iterations. defaults to 3000.
    """
    teacher.eval()
    student.eval()
    
    hooks = []
    caches = {}
    params = None
    
    def hook_for_teacher(name, module, x, y):
        tensor = y.detach()
        tensor = tensor / torch.norm(tensor)
        caches[name] = tensor
    
    def hook_for_student(name, module, x, y):
        if name not in caches:
            return
        tensor = y
        tensor = tensor / torch.norm(tensor)
        caches[name] = coeff * F.mse_loss(tensor, caches[name])
        
    for param in teacher.parameters():
        param.requires_grad = False
    
    for name, module in teacher.named_modules():
        if isinstance(module, LDGs):
            hooks.append(module.register_forward_hook(partial(hook_for_teacher, name)))
            
    for name, module in student.named_modules():
        if isinstance(module, LDGs):
            hooks.append(module.register_forward_hook(partial(hook_for_student, name)))
        if isinstance(module, Quant2d):
            if params is None:
                params = list(module.parameters())
            else:
                params = params + list(module.parameters())
    
    optimizer = Adam(params)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    checkpoint_dir = os.path.join('data', 'quantization', f'{datetime.now().strftime("%Y:%m:%d_%H:%M:%S")}')
    print(f'create {checkpoint_dir} directory')
    os.makedirs(checkpoint_dir, exist_ok=True)
    def save_checkpoint(filename):
        torch.save({
            'state_dict': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, os.path.join(checkpoint_dir, filename))
    
    device = next(student.parameters()).device
    steps = len(calib_loader)
    best_psnr = 0
    
    for epoch in tqdm(range(epochs)):
        total_out_loss = 0
        total_gdg_loss = 0
        
        for _, inputs in enumerate(calib_loader):
            optimizer.zero_grad()
            
            sc_idx = random.randint(0, len(inputs)-1)
            hr, lr = inputs[sc_idx][0], inputs[sc_idx][1]
            
            scale1, scale2 = hr.size(2) / lr.size(2), hr.size(3) / lr.size(3)
            teacher.set_scale(scale1, scale2)
            student.set_scale(scale1, scale2)
            
            lr = lr.to(device)
            output_teacher = teacher(lr, scale1, scale2)
            output_student = student(lr, scale1, scale2)
            
            out_loss = F.l1_loss(output_student, output_teacher)
            gdg_loss = 0
            for _, items in caches.items():
                gdg_loss = gdg_loss + items
            loss = out_loss + gdg_loss
            total_out_loss += out_loss.item()
            total_gdg_loss += gdg_loss.item()
            
            loss.backward()
            optimizer.step()
            
            caches.clear()
            torch.cuda.empty_cache()
        
        scheduler.step()
        
        print(f"{epoch+1}th training is finished.\
            average loss is {(total_gdg_loss + total_out_loss) / steps} \
            [GDG Layer: {total_gdg_loss / steps}, OUTPUT: {total_out_loss / steps}]")
        save_checkpoint(f'calib_{epoch+1}.pt')
        
        psnr = evaluate(student, val_loader)
        print(f"PSNR score is {psnr}")
        if psnr > best_psnr:
            best_psnr = psnr
            save_checkpoint(f'best.pt')
        
    for hook in hooks:
        hook.remove()
