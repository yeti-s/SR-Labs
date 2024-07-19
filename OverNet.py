import torch.nn as nn
from ops import *
import torch
import torch.nn.functional as F
import numpy as np


class SA_upsample(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super(SA_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale, scale2):
        b, c, h, w = x.size()

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)


        # (2) predict filters and offsets
        embedding = self.body(input)
        ## offsets
        offset = self.offset(embedding)

        ## filters
        routing_weights = self.routing(embedding)
        routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale2*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, round(scale*h), round(scale2*w), self.channels//8, self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, round(scale*h), round(scale2*w), self.channels, self.channels//8)

        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale, scale2)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1

        ## spatially varying filtering
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0

def grid_sample(x, offset, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(scale2*w)), range(round(scale*h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1),1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros')

    return output

#Local Dense Groups (LDGs)
class LDGs(nn.Module):
    def __init__(self,
                 in_channels, out_channels, wn,
                 group=1):
        super(LDGs, self).__init__()

        self.RB1 = ResidualBlock(wn, in_channels, out_channels)
        self.RB2 = ResidualBlock(wn, in_channels, out_channels)
        self.RB3 = ResidualBlock(wn, in_channels, out_channels)

        self.reduction1 = BasicConv2d(wn, in_channels*2, out_channels, 1, 1, 0)
        self.reduction2 = BasicConv2d(wn, in_channels*3, out_channels, 1, 1, 0)
        self.reduction3 = BasicConv2d(wn, in_channels*4, out_channels, 1, 1, 0)

        self.sa_adapt1 = SA_adapt(64)
        self.sa_adapt2 = SA_adapt(64)
        self.sa_adapt3 = SA_adapt(64)

    def forward(self, x, scale1, scale2):
        c0 = o0 = x

        RB1 = self.RB1(o0)
        concat1 = torch.cat([c0, RB1], dim=1)
        out1_ = self.reduction1(concat1)
        out1 = self.sa_adapt1(out1_, scale1, scale2)

        RB2 = self.RB2(out1)
        concat2 = torch.cat([concat1, RB2], dim=1)
        out2_ = self.reduction2(concat2)
        out2 = self.sa_adapt2(out2_, scale1, scale2)

        RB3 = self.RB3(out2)
        concat3 = torch.cat([concat2, RB3], dim=1)
        out3_ = self.reduction3(concat3)
        out3 = self.sa_adapt3(out3_, scale1, scale2)

        return out3

class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
        super(SA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale, scale2):
        # generate routing weights
        scale = torch.ones(1, 1).to(x.device) / scale
        scale2 = torch.ones(1, 1).to(x.device) / scale2
        routing_weights = self.routing(torch.cat((scale, scale2), 1)).view(self.num_experts, 1, 1)

        # fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.channels_in, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        # convolution
        out = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)

        return out
    
class SA_adapt(nn.Module):
    def __init__(self, channels):
        super(SA_adapt, self).__init__()
        self.mask1 = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        
        self.mask2 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_conv(channels, channels, 3, 1, 1)

    def forward(self, x, scale, scale2):
        #mask = self.mask(x)
        m1 = self.mask1(x)
        m2 = F.interpolate(m1, size=x.shape[2:], mode='bilinear', align_corners=False)
        mask = self.mask2(m2)
        adapted = self.adapt(x, scale, scale2)

        return x + adapted * mask

class Network(nn.Module):

    def __init__(self, **kwargs):
        super(Network, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)
        #upscale = kwargs.get("upscale")
        #scale = kwargs.get("scale")
        group = kwargs.get("group", 4)

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))


        self.GDG1 = LDGs(64, 64, wn=wn)
        self.GDG2 = LDGs(64, 64, wn=wn)
        self.GDG3 = LDGs(64, 64, wn=wn)

        self.reduction1 = BasicConv2d(wn, 64*2, 64, 1, 1, 0)
        self.reduction2 = BasicConv2d(wn, 64*3, 64, 1, 1, 0)
        self.reduction3 = BasicConv2d(wn, 64*4, 64, 1, 1, 0)

        self.reduction = BasicConv2d(wn, 64*3, 64, 1, 1, 0)

        self.Global_skip = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(64, 64, 1, 1, 0), nn.ReLU(inplace=True))

        #self.upsample = UpsampleBlock(64, upscale=upscale,  wn=wn, group=group)

        self.exit1 = wn(nn.Conv2d(64, 3, 3, 1, 1))

        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

        self.sa_upsample = SA_upsample(64)

    def set_scale(self, scale1, scale2):
        self.scale1 = scale1
        self.scale2 = scale2

    def forward(self, x, scale1, scale2):
        x = self.sub_mean(x)
        skip = x

        x = self.entry_1(x)

        c0 = o0 = x

        GDG1 = self.GDG1(o0, self.scale1, self.scale2)
        concat1 = torch.cat([c0, GDG1], dim=1)
        out1 = self.reduction1(concat1)

        GDG2 = self.GDG2(out1, self.scale1, self.scale2)
        concat2 = torch.cat([concat1, GDG2], dim=1)
        out2 = self.reduction2(concat2)

        GDG3 = self.GDG3(out2, self.scale1, self.scale2)
        concat3 = torch.cat([concat2, GDG3], dim=1)
        out3 = self.reduction3(concat3)


        output = self.reduction(torch.cat((out1, out2, out3),1))
        output = self.res_scale(output) + self.x_scale(self.Global_skip(x))

        #output = self.upsample(output, upscale=upscale)
        #output = F.interpolate(output, (x.size(-2) * scale, x.size(-1) * scale), mode='bicubic', align_corners=False)
        
        output = self.sa_upsample(output, self.scale1, self.scale2)
        output_size = (output.size()[2], output.size()[3])
        skip = F.interpolate(skip, size=output_size, mode='bicubic', align_corners=False)

        output = self.exit1(output) + skip
        output = self.add_mean(output)

        return output
