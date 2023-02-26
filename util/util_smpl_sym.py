import numpy as np
import tqdm
import imageio
import cv2
import argparse
import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.cpp_extension import load
from torch.autograd import Function

smpl_flow_sym = load(name='smpl_flow_symmetric', sources=['./models/networks/cal_correspondence_symmetric/cal_corr.cpp', 
                                            './models/networks/cal_correspondence_symmetric/cal_corr_cuda.cu'])

class VisMapPaddingFunction(Function):
    @staticmethod
    def forward(ctx, vis, face_num):
        vis_faces = smpl_flow_sym.vis_2_onehot(vis, face_num)
        return vis_faces[0]
    
    @staticmethod
    def backward(ctx, grad_vis):
        return None


class CorrCalculateFunction(Function):
    @staticmethod
    def forward(ctx, vis_1, faces, verts_1, verts_2, visible_faces_2, vis_sym_mask_1, symmetric_table):
        corr_map, corr_mask = smpl_flow_sym.cal_corr(vis_1, faces, verts_1, verts_2, visible_faces_2, 
                                                     vis_sym_mask_1, symmetric_table)
        return corr_map, corr_mask
    
    @staticmethod 
    def backward(ctx, grad_corr_map, grad_corr_mask):
        return None, None, None, None, None, None, None


class FlowCalculator(nn.Module):
    def __init__(self):
        super(FlowCalculator, self).__init__()
        symmetric_table = np.load('./smpl/vertices_symm_idxs.npy')
        symmetric_table = np.concatenate((symmetric_table, np.array([-1], dtype=np.int))).astype(np.int32)
        self.symmetric_table = torch.from_numpy(symmetric_table)

    def forward(self, verts_1, vis_1, verts_2, vis_2, face, vis_sym_mask_1):
        flow_s2t, vis_t_mask = self.get_correspondence_map(verts_1, vis_1, verts_2, vis_2, face, vis_sym_mask_1)
        flow_s2t = torch.stack([self.corr_to_flow(corr, order='HWC') for corr in flow_s2t])
        flow_s2t = self.fix_flow(flow_s2t.permute(0, 3, 1, 2))
        return flow_s2t, vis_t_mask

    def get_correspondence_map(self, verts_1, vis_1, verts_2, vis_2, faces, vis_sym_mask_1):
        '''
        Compute for each pixel (x,y) in img_1 the corresponding pixel (u,v) in img_2.
        Input:
            pred_1: HMR prediction for image 1. see calc_correspondence_from_smpl.
            pred_2: HMR prediction for image 2
            faces: list, each element is a triangle face represented by 3 vertex indices (i_a, i_b, i_c)
        Output:
            corr_map: (img_size, img_size, 2). corr_map[y,x] = (u,v)
            corr_mask: (img_size, img_size). The value is one of:
                0: human pixel with correspondence in img_1
                1: human pixel without correspondece in img_1
                2: background pixel
        '''
        torch.cuda.set_device(vis_1.device)
        h, w = vis_1.shape[1], vis_1.shape[2]
        pad_size = abs(h - w) // 2

        for i in range(vis_1.shape[0]):
            valid_index_A = (vis_1[i] == 4294967295)
            vis_1[i][valid_index_A] = -1
            valid_index_B = (vis_2[i] == 4294967295)
            vis_2[i][valid_index_B] = -1
            
        vis_1 = vis_1.to(torch.float32)
        vis_2 = vis_2.to(torch.float32)

        faces = torch.from_numpy(faces.astype(np.float32)).to(vis_1.device)
        vis_sym_mask_1 = vis_sym_mask_1.to(torch.float32)
        symmetric_table = self.symmetric_table.to(torch.float32).to(vis_1.device)

        verts_1, verts_2 = verts_1[:, :, :2].contiguous(), verts_2[:, :, :2].contiguous()
        invisible = -1 # 4294967295

        face_num = faces.shape[0]
        visible_faces_2 = VisMapPaddingFunction.apply(vis_2, face_num).detach()
        corr_maps, corr_masks = CorrCalculateFunction.apply(vis_1, faces, verts_1, verts_2, visible_faces_2, 
                                                            vis_sym_mask_1, symmetric_table)
        corr_maps = corr_maps.detach()
        corr_masks = corr_masks.detach()

        corr_maps[:, :, :, 0] += pad_size
        corr_maps = torch.nn.functional.pad(corr_maps,(0,0, pad_size,pad_size, 0,0),'constant',value=0)
        corr_masks = torch.nn.functional.pad(corr_masks.unsqueeze(3),(0,0,pad_size,pad_size),'constant',value=2)

        return corr_maps, corr_masks

    def corr_to_flow(self, corr, vis=None, order='NCHW'):
        '''
        order should be one of {'NCHW', 'HWC'}
        '''
        if order == 'NCHW':
            if isinstance(corr, torch.Tensor):
                flow = corr.clone()
                flow[:,0,:,:] -= torch.arange(flow.shape[3], dtype=flow.dtype, device=flow.device) # x-axis
                flow[:,1,:,:] -= torch.arange(flow.shape[2], dtype=flow.dtype, device=flow.device).view(-1,1) #  y-axis
            elif isinstance(corr, np.ndarray):
                flow = corr.copy()
                flow[:,0,:,:] -= np.arange(flow.shape[3])
                flow[:,1,:,:] -= np.arange(flow.shape[2]).reshape(-1,1)
        elif order == 'HWC':
            if isinstance(corr, torch.Tensor):
                flow = corr.clone()
                flow[:,:,0] -= torch.arange(flow.shape[1], dtype=flow.dtype, device=flow.device)
                flow[:,:,1] -= torch.arange(flow.shape[0], dtype=flow.dtype, device=flow.device).view(-1,1)
            elif isinstance(corr, np.ndarray):
                flow = corr.copy()
                flow[:,:,0] -= np.arange(flow.shape[1]).reshape(-1,)
                flow[:,:,1] -= np.arange(flow.shape[0]).reshape(-1,1)
        if vis is not None:
            if isinstance(vis, torch.Tensor):
                vis = (vis<2).float()
            elif isinstance(vis, np.ndarray):
                vis = (vis<2).astype(np.float32)
            flow *= vis
        return flow

    def fix_flow(self, flow, mode='bilinear', mask=None, mask_value=-1):
        '''
        warp an image/tensor according to given flow.
        Input:
            x: (bsz, c, h, w)
            flow: (bsz, c, h, w)
            mask: (bsz, 1, h, w). 1 for valid region and 0 for invalid region. invalid region will be fill with "mask_value" in the output images.
        Output:
            y: (bsz, c, h, w)
        '''
        bsz, c, h, w = flow.size()
        # mesh grid
        xx = flow.new_tensor(range(w)).view(1,-1).repeat(h,1)
        yy = flow.new_tensor(range(h)).view(-1,1).repeat(1,w)
        xx = xx.view(1,1,h,w).repeat(bsz,1,1,1)
        yy = yy.view(1,1,h,w).repeat(bsz,1,1,1)
        grid = torch.cat((xx,yy), dim=1).float()
        grid = grid + flow
        # scale to [-1, 1]
        grid[:,0,:,:] = 2.0*grid[:,0,:,:]/max(w-1,1) - 1.0
        grid[:,1,:,:] = 2.0*grid[:,1,:,:]/max(h-1,1) - 1.0

        grid = grid.permute(0,2,3,1)
        return grid