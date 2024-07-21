import h5py
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from util import calc_psnr

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
        
        hr = inputs[0][0].to(device)
        lr = inputs[0][1].to(device)
        sr = model(lr, scale, upscale)
        total_psnr += calc_psnr(sr, hr, scale, 1)

    return total_psnr / steps