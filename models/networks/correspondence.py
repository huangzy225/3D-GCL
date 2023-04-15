# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tkinter import X
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import util.util as util
from models.networks.base_network import BaseNetwork
from models.networks.architecture import ResidualBlock
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.gacrms import GACRMBlock
from models.networks.ops import *


def pono_c(feature, eps=1e-10):
    b, c, h, w = feature.size()
    feature = feature.view(b, c, -1)
    dim_mean = 1 
    feature = feature - feature.mean(dim=dim_mean, keepdim=True)
    feature_norm = torch.norm(feature, 2, 1, keepdim=True) + eps
    feature = torch.div(feature, feature_norm)
    return feature.view(b, -1, h, w)


class FeatureEncoder(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt, input_nc=20):
        super().__init__()
        self.opt = opt
        kw = opt.featEnc_kernel
        pw = int((kw-1)//2)
        nf = opt.nef
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = nn.Sequential(norm_layer(nn.Conv2d(input_nc, nf*2, 3, stride=1, padding=pw)),
                                   nn.LeakyReLU(0.2))

        self.layer2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 2, nf * 2),
        )
        self.layer3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 4, nf * 4),
        )
        self.layer4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 8, kw, stride=2, padding=pw)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 8, nf * 8),
        )
        self.layer5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 8, kw, stride=2, padding=pw)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 8, nf * 8),
        )
        # self.layer6 = nn.Sequential(
        #     norm_layer(nn.Conv2d(nf * 4, nf * 4, kw, stride=2, padding=pw)),
        #     ResidualBlock(nf * 4, nf * 4),
        # )

    def forward(self, input):
        x1 = self.layer1(input)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        # x6 = self.layer6(self.actvn(x5))
        return [x5]

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class FeatureEncoder_v2(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt, input_nc=20):
        super().__init__()
        self.opt = opt
        nf = opt.nef
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = nn.Sequential(norm_layer(nn.Conv2d(input_nc, nf*2, 3, stride=2, padding=1)),
                                    nn.LeakyReLU(0.2))

        self.layer2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 2, nf * 2),
        )
        self.layer3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 4, nf * 4),
        )
        self.layer4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 8, 5, stride=1, padding=2)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 8, nf * 8),
        )
        self.layer5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 4, 5, stride=1, padding=2)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 4, nf * 4),
        )

    def forward(self, input):
        x1 = self.layer1(input)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return [x5]

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class FeatureEncoderHD(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt, input_nc=20):
        super().__init__()
        self.opt = opt
        nf = opt.nef
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = nn.Sequential(norm_layer(nn.Conv2d(input_nc, nf*2, 3, stride=2, padding=1)),
                                    nn.LeakyReLU(0.2))

        self.layer2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 2, nf * 2),
        )
        self.layer3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 4, nf * 4),
        )
        self.layer4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 4, 5, stride=1, padding=2)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 4, nf * 4),
        )
        self.layer5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 4, 5, stride=1, padding=2)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 4, nf * 4),
        )

        # upsample
        self.merge1 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 4, 5, stride=1, padding=2)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 4, nf * 4),
        )
        self.merge2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 4, 5, stride=1, padding=2)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 4, nf * 4),
        )
        self.up1 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * (4+2), nf * 4, 5, stride=1, padding=2)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 4, nf * 4),
        )
        self.up2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * (4+2), nf * 4, 5, stride=1, padding=2)),
            nn.LeakyReLU(0.2),
            ResidualBlock(nf * 4, nf * 4),
        )

    def forward(self, input):
        x1 = self.layer1(input)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        cat1 = self.merge1(torch.cat([x5, x4], dim=1))
        cat2 = self.merge2(torch.cat([cat1, x3], dim=1))
        up1 = self.up1(torch.cat([nn.Upsample(scale_factor=2)(cat2), x2], dim=1))
        up2 = self.up2(torch.cat([nn.Upsample(scale_factor=2)(up1), x1], dim=1))

        return [x5, up1, up2]

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class GlobalHDCorrespondence(BaseNetwork):
    def __init__(self, opt):
        self.opt = opt
        super().__init__()

        self.content_encoder = FeatureEncoderHD(opt, input_nc=3)
        self.refernce_encoder = FeatureEncoderHD(opt, input_nc=3+3)

        feature_channel = opt.nef
        self.phi_1 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.theta_1 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)

        self.phi_2 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.theta_2 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)

        self.phi_3 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.theta_3 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)

        self.GACRM = GACRMBlock(opt, hidden_dim=32)  
        self.GACRM2 = GACRMBlock(opt, hidden_dim=32)  

        # filter_x = [[0, 0, 0],
        #             [1, -2, 1],
        #             [0, 0, 0]]
        # filter_y = [[0, 1, 0],
        #             [0, -2, 0],
        #             [0, 1, 0]]
        # filter_diag1 = [[1, 0, 0],
        #                 [0, -2, 0],
        #                 [0, 0, 1]]
        # filter_diag2 = [[0, 0, 1],
        #                 [0, -2, 0],
        #                 [1, 0, 0]]
        # weight_array = np.ones([3, 3, 1, 4])
        # weight_array[:, :, 0, 0] = filter_x
        # weight_array[:, :, 0, 1] = filter_y
        # weight_array[:, :, 0, 2] = filter_diag1
        # weight_array[:, :, 0, 3] = filter_diag2

        # weight_array = torch.cuda.FloatTensor(weight_array).permute(3,2,0,1)
        # self.weight = nn.Parameter(data=weight_array, requires_grad=False)

    def tensor_dilate(self, bin_img, ksize=3):
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        dilated, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
        return dilated

    def soft_argmax(self, x):
        x = F.softmax(x / 0.001, dim=-1)
        scale = 64
        batch_size, c = x.shape[0], x.shape[1]
        index = torch.arange(c, dtype=x.dtype, device=x.device).contiguous()
        x = x.permute(0,2,1).view(batch_size, c, scale, scale)
        index = index.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        inds = torch.sum(index * x, dim=1, keepdim=True)
        offset_x, offset_y = inds_to_offset(inds)
        return offset_x, offset_y

    """512x512"""
    def multi_scale_warping(self, f1, f2, ref, hierarchical_scale, pre=None):
        if hierarchical_scale == 0:
            scale = 64
            batch_size, channel, feature_height, feature_width = f1.size()
            ref = F.interpolate(ref, scale_factor=0.5 ** 3, mode='bilinear')
            ref_flatten = ref.view(batch_size, 3, scale * scale)
            f1 = f1.view(batch_size, channel, scale * scale)
            f2 = f2.view(batch_size, channel, scale * scale)
            matmul_result = torch.matmul(f1.permute(0, 2, 1), f2)/self.opt.temperature
            mat = F.softmax(matmul_result, dim=-1)
            y_flatten = torch.matmul(mat, ref_flatten.permute(0, 2, 1))
            y = y_flatten.permute(0, 2, 1).view(batch_size, 3, scale, scale)

            # matmul_cycle = torch.matmul(f2.permute(0, 2, 1), f1)/self.opt.temperature
            # mat_cycle = F.softmax(matmul_result, dim=-1)
            # y_cycle = torch.matmul(mat_cycle, y_flatten)
            # y_cycle = y_cycle.permute(0, 2, 1).view(batch_size, 3, scale, scale)

            scale_cloth_mask = F.interpolate(self.cloth_mask, scale_factor=0.5 ** 3, mode='bilinear')
            scale_cloth_mask = scale_cloth_mask.view(batch_size, 1, scale * scale)
            warp_mask = torch.matmul(mat.detach(), scale_cloth_mask.permute(0, 2, 1))
            warp_mask = warp_mask.permute(0, 2, 1).view(batch_size, 1, scale, scale)
            self.warp_mask = self.tensor_dilate(warp_mask)

            # offset_x, offset_y = self.soft_argmax(matmul_result)
            # offset_approx = torch.cat((offset_y, offset_x), dim=1)

            return mat, y, self.warp_mask

        if hierarchical_scale == 1:
            scale = 128
            # with torch.no_grad():
            batch_size, channel, feature_height, feature_width = f1.size()
            topk_num = 1
            topk_inds = torch.topk(pre.detach(), topk_num, dim=-1)[-1]
            inds = topk_inds.permute(0, 2, 1).view(batch_size, topk_num, (scale//2), (scale//2)).float()
            offset_x, offset_y = inds_to_offset(inds)
            offset_x_up = F.interpolate((2 * offset_x), scale_factor=2, mode='nearest')
            offset_y_up = F.interpolate((2 * offset_y), scale_factor=2, mode='nearest')
            scale_ref = F.avg_pool2d(ref, 4, stride=4)
            offsets, y, flows = self.GACRM(f1, f2, scale_ref, offset_x_up, offset_y_up, self.warp_mask)
            return offsets, y, flows
        if hierarchical_scale == 2:
            offset_y, offset_x = torch.split(pre, 1, dim=1)
            offset_x_up = F.interpolate((2 * offset_x), scale_factor=2, mode='nearest')
            offset_y_up = F.interpolate((2 * offset_y), scale_factor=2, mode='nearest')
            scale_ref = F.avg_pool2d(ref, 2, stride=2)
            offsets, y, flows = self.GACRM2(f1, f2, scale_ref, offset_x_up, offset_y_up, self.warp_mask)
            return offsets, y, flows

    def get_tv_loss(self, x, mask):
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]) * mask[:,:,1:,:]
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]) * mask[:,:,:,1:]
        return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))

    def forward(self, data_unpack):
        corr_out = {}
        cur_cloth_ref = data_unpack['cloth_ref']
        cont_input = data_unpack['tar_uv']
        ref_input = torch.cat((data_unpack['src_uv'], cur_cloth_ref), dim=1)
        
        cont_features = self.content_encoder(cont_input)
        ref_features = self.refernce_encoder(ref_input)

        theta = []
        phi = []
        """512x512"""
        theta.append(pono_c(self.theta_1(cont_features[0])))
        phi.append(pono_c(self.phi_1(ref_features[0])))
          
        theta.append(pono_c(self.theta_2(cont_features[1])))
        phi.append(pono_c(self.phi_2(ref_features[1])))

        theta.append(pono_c(self.theta_3(cont_features[2])))
        phi.append(pono_c(self.phi_3(ref_features[2])))

        ref = cur_cloth_ref
        self.cloth_mask = data_unpack['cloth_ref_mask']

        ys = []
        m = None
        m, y, warp_mask = self.multi_scale_warping(theta[0], phi[0], ref, 0, pre=m)
        ys.append(y)
        offset_128, y, flow128 = self.multi_scale_warping(theta[1], phi[1], ref, 1, pre=m)
        ys.append(y)
        offset_256, y, flow256 = self.multi_scale_warping(theta[2], phi[2], ref, 2, pre=offset_128)
        ys.append(y)
        
        corr_out['warp_out'] = ys
        corr_out['warp_mask'] = warp_mask
        corr_out['corr_m'] = m
        corr_out['flow128'] = flow128
        corr_out['flow256'] = flow256

        return corr_out

