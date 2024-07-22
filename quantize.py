import argparse
import torch
from dataset import TrainDataset
from torch.utils.data import DataLoader

from OverNet import Network
from quantization.quant2d import get_boundaries, quantize_dobi, distilate
from quantization.eval import TestDataset

scale, upscale = 4, 4

def load_model(checkpoint):
    state_dict = torch.load(checkpoint, map_location='cpu')['model_state_dict']
    teacher, student = Network(), Network()
    teacher.load_state_dict(state_dict)
    student.load_state_dict(state_dict)
    return teacher, student

def quantize(model, calib_data, n_bits, device):
    dobi_calib_dataset = TrainDataset(calib_data, 64, scale)
    dobi_calib_loader = DataLoader(dobi_calib_dataset, 32, True)
    boundaries = get_boundaries(model, dobi_calib_loader, scale, upscale, n_bits)
    
    model.cpu()
    quantize_dobi(model, boundaries, '', n_bits)
    model.to(device)

def calibrate(teacher, student, calib_data, val_data):
    calib_dataset = TrainDataset(calib_data, 64, scale)
    calib_loader = DataLoader(calib_dataset, 16, True)
    val_dataset = TestDataset(val_data, scale)
    val_loader = DataLoader(val_dataset, 1, False)
    distilate(teacher, student, calib_loader, val_loader, scale, upscale)

def main(args):
    calib_data, val_data = args.calib_data, args.val_data
    scale, upscale = args.scale, args.upscale
    n_bits, device = args.bits, args.device
    
    teacher, student = load_model(args.checkpoint)
    teacher.set_scale(scale, upscale)
    teacher.to(device)
    student.set_scale(scale, upscale)
    student.to(device)
    
    quantize(student, calib_data, n_bits, device)
    calibrate(teacher, student, calib_data, val_data)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='model parameters')
    parser.add_argument('--calib_data', type=str, help='calibration dataset')
    parser.add_argument('--val_data', type=str, help='validation dataset')
    parser.add_argument('--scale', type=int, help='scale for model inference')
    parser.add_argument('--upscale', type=int, help='upscale for model inference')
    parser.add_argument('--device', type=str, default='cpu', help='device for model inference and training')
    parser.add_argument('--bits', type=int, default=8, help='quantization bits')
    main(parser.parse_args())