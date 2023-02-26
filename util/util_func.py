import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
import torch
import random, numbers, math
import cv2
from random import randint
# import flow_vis
import torch.nn.functional as F


def save_im_tensor(x, name):
    x = x[0].permute(1,2,0)
    x = x.detach().cpu().numpy()
    x = x *0.5 + 0.5
    x = np.clip((x * 255).astype(np.uint8), 0, 255)
    x = x[:,:,[2,1,0]]
    cv2.imwrite('./MM_submission/{}.jpg'.format(name), x)


def visualize_img(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    num = len(imgs)
    row = int(math.sqrt(num))
    col = (num + row - 1) // row
    for i, img in enumerate(imgs):
        plt.subplot(row, col, i+1)
        if len(img.shape) == 4:
            img = img[0]
        # if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if len(img.shape) > 2:                      # three channels
            if img.shape[0] < img.shape[2]:         # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 3:                   # RGB image
                img = img * 0.5 + 0.5
            else:
                img = img[:, :, 0]                  # gray scale
        plt.imshow(img, cmap='gray')
    #plt.show()
    plt.savefig('./debug.png')


# def visualize_flow(flow, gt_flow, res):
#     resolution = eval(res)
#     grid_list = torch.meshgrid([torch.arange(size, device=flow.device) for size in [resolution,resolution]])
#     grid_list = reversed(grid_list)
#     grid_list = [grid / ((size - 1.0) / 2.0) - 1.0 for grid, size in zip(grid_list, reversed([resolution,resolution]))] 
#     grid_list = [flow[0,:,:,dim] - grid.float().unsqueeze(0) for dim, grid in enumerate(grid_list)]
#     flow_numpy = torch.stack(grid_list, dim=-1).detach().cpu().numpy()
#     smpl_flow = (gt_flow.permute(0, 3, 1, 2)).detach().cpu().numpy()
#     vis_flow = flow_vis.flow_to_color(flow_numpy[0], convert_to_bgr=False)
#     vis_gt_flow = flow_vis.flow_to_color(smpl_flow[0].transpose(1, 2, 0), convert_to_bgr=False)
#     cv2.imwrite('./{}.jpg'.format('flow_{}'.format(res)), vis_flow[:,:,[2,1,0]])
#     cv2.imwrite('./{}.jpg'.format('gt_flow_{}'.format(res)), vis_gt_flow[:,:,[2,1,0]])

#     a = 10



def tensor_dilate(bin_img, ksize=3):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    dilated, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
    return dilated
