# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from turtle import left
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.convgru import BasicUpdateBlock
from models.networks.ops import *


class GACRMBlock(nn.Module):
    def __init__(self, opt, hidden_dim=32):
        super().__init__()
        self.temperature = opt.temperature
        self.iters = opt.iteration_count
        norm = nn.InstanceNorm2d(hidden_dim, affine=False)
        relu = nn.ReLU(inplace=True)
        """
        concat left and right features
        """
        self.initial_layer = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1, stride=1),
            norm,
            relu,
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            norm,
            relu,
        )
        self.refine_net = BasicUpdateBlock()

        self.filter = nn.Sequential(nn.Conv2d(2, 32, kernel_size=5, padding=2),
                       norm,
                       relu,
                       nn.Conv2d(32, 2, kernel_size=5, padding=2))
        self.filter2 = nn.Sequential(nn.Conv2d(2, 32, kernel_size=5, padding=2),
                       norm,
                       relu,
                       nn.Conv2d(32, 2, kernel_size=5, padding=2))

        self.corr_layer = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=7, padding=3, stride=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, stride=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, left_features, right_features, right_input, initial_offset_x, 
                initial_offset_y, warp_mask=None):
        batch_size, channel, height, width = left_features.size()
        initial_input = torch.cat((left_features, right_features), dim=1)
        hidden = self.initial_layer(initial_input)
        offset_x, offset_y = initial_offset_x, initial_offset_y

        tmp_flow = apply_offset(torch.cat([offset_y, offset_x], dim=1))
        right_features = F.grid_sample(right_features, tmp_flow, mode='bilinear')

        corr = self.corr_layer(torch.cat([left_features, right_features], dim=1))

        for it in range(self.iters):
            """GRU refinement"""
            flow = torch.cat((offset_y, offset_x), dim=1)
            hidden, delta_offset_y, delta_offset_x = self.refine_net(hidden, corr, flow)

            offset_x = offset_x + delta_offset_x 
            offset_y = offset_y + delta_offset_y

            if it == 0:
                out_offset = self.filter(torch.cat([offset_x, offset_y], dim=1))
                offset_x, offset_y = torch.split(out_offset, [1, 1], dim=1)
            else:
                out_offset = self.filter2(torch.cat([offset_x, offset_y], dim=1))
                offset_x, offset_y = torch.split(out_offset, [1, 1], dim=1)
        _warp = torch.zeros_like(right_input)
        offset_max = torch.stack((offset_y[:, 0, :, :], offset_x[:, 0, :, :]), dim=1) 
        flow_max = apply_offset(offset_max)

        _warp = F.grid_sample(right_input, flow_max, align_corners=False) 

        if not warp_mask is None:
            warp_mask = F.interpolate(warp_mask, size=(height, width), mode='bilinear')
            _warp = _warp * warp_mask + (1 - warp_mask)

        split_offset_x = torch.split(offset_x, 1, dim=1)
        split_offset_y = torch.split(offset_y, 1, dim=1)
        flows = []
        for sub_offset_x, sub_offset_y in zip(split_offset_x, split_offset_y):
            flows.append(apply_offset(torch.cat((sub_offset_y, sub_offset_x), dim=1)))

        offsets = torch.cat((offset_y, offset_x), dim=1)
        return offsets, _warp, torch.cat(flows, dim=3)

    