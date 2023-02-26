# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_1d_to_2d(index, base=64):
    x = index // base
    y = index % base
    return x,y


def convert_2d_to_1d(x, y, base=64):
    return x*base+y


def batch_meshgrid(shape, device):
    batch_size, _, height, width = shape
    x_range = torch.arange(0.0, width, device=device)
    y_range = torch.arange(0.0, height, device=device)
    x_coordinate, y_coordinate = torch.meshgrid(x_range, y_range)
    x_coordinate = x_coordinate.expand(batch_size, -1, -1).unsqueeze(1)
    y_coordinate = y_coordinate.expand(batch_size, -1, -1).unsqueeze(1)
    return x_coordinate, y_coordinate


def inds_to_offset(inds):
    """
    inds: b x number x h x w
    """
    shape = inds.size()
    device = inds.device
    x_coordinate, y_coordinate = batch_meshgrid(shape, device)
    batch_size, _, height, width = shape
    x = inds.div(width)
    y = inds - (inds.detach() // width) * width
    return x - x_coordinate, y - y_coordinate
    # x = inds // width
    # y = inds % width
    # return x - x_coordinate, y - y_coordinate


def offset_to_inds(offset_x, offset_y):
    shape = offset_x.size()
    device = offset_x.device
    x_coordinate, y_coordinate = batch_meshgrid(shape, device)
    h, w = offset_x.size()[2:]
    x = torch.clamp(x_coordinate + offset_x, 0, h-1)
    y = torch.clamp(y_coordinate + offset_y, 0, w-1)
    return x * offset_x.size()[3] + y


def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list)]
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))] 

    return torch.stack(grid_list, dim=-1)

# new offset implementation
# def apply_offset(offset):
#     sizes = list(offset.size()[2:])
#     grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
#     grid_list = reversed(grid_list)
#     # apply offset
#     grid_list = [(grid.float().unsqueeze(0)  / (size - 1.0)) * 2 - 1.0
#                         for size, grid in zip(reversed(sizes), grid_list)]
#     offset = torch.tanh(offset)
#     # normalize
#     grid_list = [grid + offset[:, dim, ...] for dim, grid in enumerate(grid_list)] 

#     return torch.stack(grid_list, dim=-1)

# def apply_offset(offset):
#     '''
#         convert offset grid to location grid
#         offset: [N, 2, H, W] for 2D or [N, 3, D, H, W] for 3D
#         output: [N, 2, H, W] for 2D or [N, 3, D, H, W] for 3D
#     '''
#     sizes = list(offset.size()[2:])
#     grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
#     grid_list = reversed(grid_list)
#     # apply offset
#     grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
#                  for dim, grid in enumerate(grid_list)]

#     grid_list = [torch.stack([torch.clamp(grid[j], 0, sizes[1-i]-1) for j in range(grid.shape[0])]) 
#                  for i, grid in enumerate(grid_list)]

#     grid_list = [torch.round(grid) for grid in grid_list]
#     # normalize
#     grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
#                  for grid, size in zip(grid_list, reversed(sizes))]

#     return torch.stack(grid_list, dim=-1)


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


def depth_to_space(x, block_size):
    return F.pixel_shuffle(x, upscale_factor=block_size)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k
