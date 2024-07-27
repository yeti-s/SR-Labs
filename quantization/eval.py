import random
import h5py
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from util import calc_psnr_legacy

class TestDataset(Dataset):
    def __init__(self, path, scale):
        super(TestDataset, self).__init__()

        h5f = h5py.File(path, "r")

        self.hr = [v[:] for v in h5f["HR"].values()]
        # perform multi-scale training
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]

        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        item = [(self.hr[index], self.lr[i][index]) for i, _ in enumerate(self.lr)]
        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]

    def __len__(self):
        return len(self.hr)


@torch.no_grad()
def evaluate(model:nn.Module, data_loader:DataLoader, steps:int=0) -> float:
    """_summary_

    Args:
        model (nn.Module): _description_
        data_loader (DataLoader): _description_
        steps (int, optional): _description_. Defaults to 0.

    Returns:
        float: _description_
    """
    
    random.seed(42) # reset random seed
    
    model.eval()
    total_psnr = 0
    device = next(model.parameters()).device
    steps = len(data_loader) if steps == 0 else steps
    
    for step, inputs in enumerate(data_loader):
        if step >= steps:
            break
        
        scale_idx = random.randint(0, len(inputs)-1)
        hr = inputs[scale_idx][0].to(device)
        lr = inputs[scale_idx][1].to(device)
        scale, upscale = hr.size(2) // lr.size(2), hr.size(3) // lr.size(3)
            
        model.set_scale(scale, upscale)
        sr = model(lr, scale, upscale)
        total_psnr += calc_psnr_legacy(sr, hr, scale, 1)

    return total_psnr / steps

@torch.no_grad()
def infer(model:nn.Module, data_loader:DataLoader, steps:int=0) -> None:
    """_summary_

    Args:
        model (nn.Module): _description_
        data_loader (DataLoader): _description_
        steps (int, optional): _description_. Defaults to 0.

    Returns:
        float: _description_
    """
    
    random.seed(42) # reset random seed
    
    model.eval()
    device = next(model.parameters()).device
    steps = len(data_loader) if steps == 0 else steps
    
    for step, inputs in enumerate(data_loader):
        if step >= steps:
            break
        
        scale_idx = random.randint(0, len(inputs)-1)
        hr = inputs[scale_idx][0].to(device)
        lr = inputs[scale_idx][1].to(device)
        scale, upscale = hr.size(2) / lr.size(2), hr.size(3) / lr.size(3)
            
        model.set_scale(scale, upscale)
        model(lr, scale, upscale)

@torch.no_grad()
def evaluate_random_scale(model, test_data):
    
    random.seed(42) # reset random seed
    
    model.eval()
    device = next(model.parameters()).device
    mean_psnr = 0

    test_data = TestDataset(test_data, 0)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=1)

    for _, inputs in enumerate(test_loader):
        hr = inputs[0]
        lr = inputs[1]

        lr = lr.to(device)
        hr = hr.to(device)

        scale = hr.size(2) / lr.size(2)
        upscale = hr.size(3) / lr.size(3)
        model.set_scale(scale, upscale)
        sr = model(lr, scale, upscale)

        psnr = calc_psnr_legacy(sr, hr, scale, 1, benchmark=True)
        mean_psnr += psnr / len(test_data)

    return mean_psnr