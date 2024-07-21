import torch
import torch.nn as nn
import torch.autograd as autograd
from torch import Tensor

EPS = 1e-8

# custom round function to calculate backpropagation
class RoundSTEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class RoundSTE(nn.Module):
    def forward(self, input):
        return RoundSTEFunction.apply(input)
    
round_ste = RoundSTE()

def fake_quantize(tensor:Tensor, l:Tensor, u:Tensor, n_bits:int) -> Tensor:
    """Simulate quantization for calculating quantization error

    Args:
        tensor (Tensor): tensor to quantize
        l (Tensor): lower bound for quantization
        u (Tensor): upper bound for quantization
        n_bits (int): the number of quantization bits.

    Returns:
        Tensor: dequantized rounded tensor
    """
    c_tensor = torch.clip(tensor, l, u)
    tensor_range, bits_range = (u - l) + EPS, 2**n_bits - 1
    
    r_tensor = bits_range / tensor_range * (c_tensor - l)
    r_tensor = round_ste(r_tensor)
    return r_tensor * tensor_range / bits_range + l

