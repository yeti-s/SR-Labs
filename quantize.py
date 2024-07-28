import argparse
import torch
from dataset import TrainDataset
from torch.utils.data import DataLoader

from OverNet import Network
from quantization.quant2d import get_boundaries, quantize_dobi, distilate, get_model_size
from quantization.eval import TestDataset

def load_model(checkpoint):
    state_dict = torch.load(checkpoint, map_location='cpu')['model_state_dict']
    teacher, student = Network(), Network()
    teacher.load_state_dict(state_dict)
    student.load_state_dict(state_dict)
    return teacher, student

def quantize(model, calib_data, n_bits, device):
    dobi_calib_dataset = TrainDataset(calib_data, 64, 4)
    dobi_calib_loader = DataLoader(dobi_calib_dataset, 32, True)
    boundaries = get_boundaries(model, dobi_calib_loader, 4, 4, n_bits)
    
    model.cpu()
    quantize_dobi(model, boundaries, '', n_bits)
    model.to(device)

def calibrate(teacher, student, calib_data, val_data, batch_size, coeff):
    calib_dataset = TrainDataset(calib_data, 64, 0)
    calib_loader = DataLoader(calib_dataset, batch_size, True)
    val_dataset = TestDataset(val_data, 0)
    val_loader = DataLoader(val_dataset, 1, False)
    distilate(teacher, student, calib_loader, val_loader, coeff=coeff)

def main(args):
    calib_data, val_data = args.calib_data, args.val_data
    n_bits, device = args.bits, args.device
    batch_size, coeff = args.batch_size, args.coeff
    
    teacher, student = load_model(args.checkpoint)
    print(f'original model size is {get_model_size(teacher)} KB')
    teacher.to(device)
    student.to(device)
    
    quantize(student, calib_data, n_bits, device)
    print(f'model size after quantization is {get_model_size(student)} KB')
    
    calibrate(teacher, student, calib_data, val_data, batch_size, coeff)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='model parameters')
    parser.add_argument('--calib_data', type=str, help='calibration dataset')
    parser.add_argument('--val_data', type=str, help='validation dataset')
    parser.add_argument('--device', type=str, default='cpu', help='device for model inference and training')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for distilation')
    parser.add_argument('--coeff', type=int, default=10000, help='coefficient of layer and output loss')
    parser.add_argument('--bits', type=int, default=8, help='quantization bits')
    main(parser.parse_args())