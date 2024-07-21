import argparse
import torch

from OverNet import Network
from quantization.quant2d import quantize

def main(args):
    model = Network()
    scale, upscale = args.scale, args.upscale
    model.set_scale(scale, upscale)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    quantize(model, args.calib_data, args.val_data, scale, upscale, args.device, n_bits=args.bits)
    
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