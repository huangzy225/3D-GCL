import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.networks.stylegan2_architecture import *
from models.networks.base_network import BaseNetwork


class PoseEncoder(nn.Module):
    def __init__(self, ngf=64, blur_kernel=[1, 3, 3, 1], size=256):
        super().__init__()
        self.size = size
        convs = [ConvLayer(3, ngf, 1)]
        convs.append(ResBlock(ngf, ngf*2, blur_kernel))
        convs.append(ResBlock(ngf*2, ngf*4, blur_kernel))
        convs.append(ResBlock(ngf*4, ngf*8, blur_kernel))
        convs.append(ResBlock(ngf*8, ngf*8, blur_kernel))
        if self.size == 512:
            convs.append(ResBlock(ngf*8, ngf*8, blur_kernel))
        if self.size == 1024:
            convs.append(ResBlock(ngf*8, ngf*8, blur_kernel))
            convs.append(ResBlock(ngf*8, ngf*8, blur_kernel))

        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        out = self.convs(input)
        return out


class GarmentTransferSpatialAppearanceEncoder0308(nn.Module):
    def __init__(self, ngf=64, blur_kernel=[1, 3, 3, 1], size=256, part='upper_body'):
        super().__init__()
        self.size = size
        self.part = part

        input_nc = 3 # source RGB and sil

        self.conv1 = ConvLayer(input_nc, ngf, 1)                # ngf 256 256
        self.conv2 = ResBlock(ngf, ngf*2, blur_kernel)          # 2ngf 128 128
        self.conv3 = ResBlock(ngf*2, ngf*4, blur_kernel)        # 4ngf 64  64
        self.conv4 = ResBlock(ngf*4, ngf*8, blur_kernel)        # 8ngf 32  32
        self.conv5 = ResBlock(ngf*8, ngf*8, blur_kernel)        # 8ngf 16  16
        self.conv6 = ResBlock(ngf*8, ngf*8, blur_kernel)        # 8ngf 16  16 - starting from ngf 512 512

        
        self.conv_f1 = ConvLayer(input_nc, ngf, 1)              # ngf 256 256
        self.conv_f2 = ResBlock(ngf, ngf, blur_kernel)          # 2ngf 128 128
        self.conv_f3 = ResBlock(ngf, ngf, blur_kernel)          # 4ngf 64  64
        self.conv_f4 = ResBlock(ngf, ngf, blur_kernel)          # 8ngf 32  32
        self.conv_f5 = ResBlock(ngf, ngf, blur_kernel)          # 8ngf 16  16
        self.conv_f6 = ResBlock(ngf, ngf, blur_kernel)          # 8ngf 16  16 - starting from ngf 512 512

        mult = 8
        self.conv11 = EqualConv2d(ngf+ngf, ngf*mult, 1)
        self.conv21 = EqualConv2d(ngf*2+ngf, ngf*mult, 1)
        self.conv31 = EqualConv2d(ngf*4+ngf, ngf*mult, 1)
        self.conv41 = EqualConv2d(ngf*8+ngf, ngf*mult, 1)
        self.conv51 = EqualConv2d(ngf*8+ngf, ngf*mult, 1)
        self.conv61 = EqualConv2d(ngf*8+ngf, ngf*mult, 1)

        self.conv13 = EqualConv2d(ngf*mult, ngf*1, 3, padding=1)
        self.conv23 = EqualConv2d(ngf*mult, ngf*2, 3, padding=1)
        self.conv33 = EqualConv2d(ngf*mult, ngf*4, 3, padding=1)
        self.conv43 = EqualConv2d(ngf*mult, ngf*8, 3, padding=1)
        self.conv53 = EqualConv2d(ngf*mult, ngf*8, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, data_pack, flow_up):
        # input
        input_cloth = data_pack['cloth_ref']
        x1 = self.conv1(input_cloth)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        # warp- resize flow
        f1 = torch.nn.functional.interpolate(flow_up.permute(0,3,1,2), size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=True)
        f2 = torch.nn.functional.interpolate(flow_up.permute(0,3,1,2), size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=True)
        f3 = torch.nn.functional.interpolate(flow_up.permute(0,3,1,2), size=(x3.shape[2], x3.shape[3]), mode='bilinear', align_corners=True)
        f4 = torch.nn.functional.interpolate(flow_up.permute(0,3,1,2), size=(x4.shape[2], x4.shape[3]), mode='bilinear', align_corners=True)
        f5 = torch.nn.functional.interpolate(flow_up.permute(0,3,1,2), size=(x5.shape[2], x5.shape[3]), mode='bilinear', align_corners=True)
        f6 = torch.nn.functional.interpolate(flow_up.permute(0,3,1,2), size=(x6.shape[2], x6.shape[3]), mode='bilinear', align_corners=True)

        # warp- now warp
        upper_x1 = torch.nn.functional.grid_sample(x1, f1.permute(0,2,3,1))
        upper_x2 = torch.nn.functional.grid_sample(x2, f2.permute(0,2,3,1))
        upper_x3 = torch.nn.functional.grid_sample(x3, f3.permute(0,2,3,1))
        upper_x4 = torch.nn.functional.grid_sample(x4, f4.permute(0,2,3,1))
        upper_x5 = torch.nn.functional.grid_sample(x5, f5.permute(0,2,3,1))
        upper_x6 = torch.nn.functional.grid_sample(x6, f6.permute(0,2,3,1))

        upper_ref_mask = data_pack['cloth_ref_mask']
        upper_warp_mask = F.grid_sample(upper_ref_mask, flow_up, mode='bilinear')
        part_mask1 = torch.nn.functional.interpolate(upper_warp_mask, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=True)
        part_mask2 = torch.nn.functional.interpolate(upper_warp_mask, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=True)
        part_mask3 = torch.nn.functional.interpolate(upper_warp_mask, size=(x3.shape[2], x3.shape[3]), mode='bilinear', align_corners=True)
        part_mask4 = torch.nn.functional.interpolate(upper_warp_mask, size=(x4.shape[2], x4.shape[3]), mode='bilinear', align_corners=True)
        part_mask5 = torch.nn.functional.interpolate(upper_warp_mask, size=(x5.shape[2], x5.shape[3]), mode='bilinear', align_corners=True)
        part_mask6 = torch.nn.functional.interpolate(upper_warp_mask, size=(x6.shape[2], x6.shape[3]), mode='bilinear', align_corners=True)

        # target
        # scale_warp_mask = F.interpolate(upper_warp_mask, scale_factor=2, mode='nearest')
        # overlap_mask = (1 - scale_warp_mask) * data_pack['lower_mask']
        # retain = data_pack['head'] * (1 - overlap_mask) + overlap_mask
        retain = data_pack['head']
        
        face_x1 = self.conv_f1(retain)
        face_x2 = self.conv_f2(face_x1)
        face_x3 = self.conv_f3(face_x2)
        face_x4 = self.conv_f4(face_x3)
        face_x5 = self.conv_f5(face_x4)
        face_x6 = self.conv_f6(face_x5)

        x1 = upper_x1 * part_mask1 
        x2 = upper_x2 * part_mask2 
        x3 = upper_x3 * part_mask3
        x4 = upper_x4 * part_mask4 
        x5 = upper_x5 * part_mask5 
        x6 = upper_x6 * part_mask6 

        F6 = self.conv61(torch.cat([x6,face_x6], 1))
        f5 = self.up(F6)+self.conv51(torch.cat([x5,face_x5], 1))
        F5 = self.conv53(f5)
        f4 = self.up(f5)+self.conv41(torch.cat([x4,face_x4], 1))
        F4 = self.conv43(f4)
        f3 = self.up(f4)+self.conv31(torch.cat([x3,face_x3], 1))
        F3 = self.conv33(f3)
        f2 = self.up(f3)+self.conv21(torch.cat([x2,face_x2], 1))
        F2 = self.conv23(f2)
        f1 = self.up(f2)+self.conv11(torch.cat([x1,face_x1], 1))
        F1 = self.conv13(f1)

        return [F6, F5, F4, F3, F2, F1]



class Stylegan2Generator(BaseNetwork):
    def __init__(
        self,
        opt,
        size=512,
        style_dim=2048,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        garment_transfer=False,
        part='upper_body',
    ):
        super().__init__()

        self.garment_transfer = garment_transfer
        self.size = size
        self.style_dim = style_dim

        self.appearance_encoder = GarmentTransferSpatialAppearanceEncoder0308(size=size, part=part)
        self.pose_encoder = PoseEncoder(size=size)

        # StyleGAN
        self.channels = {
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.conv1 = StyledConv(
            self.channels[16], self.channels[16], 3, style_dim, blur_kernel=blur_kernel, spatial=True
        )
        self.to_rgb1 = ToRGB(self.channels[16], style_dim, upsample=False, spatial=True)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 4) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[16]

        for i in range(5, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    spatial=True,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, spatial=True,
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, spatial=True))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2


    def make_noise(self):
        device = self.input.input.device
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises


    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        data_pack, 
        styles=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        flow_up = data_pack['flow256']
        styles = self.appearance_encoder(data_pack, flow_up)

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        latent = [styles[0], styles[0]]
        if self.size == 1024:
            length = 6
        elif self.size == 512:
            length = 5
        else:
            length = 4
        for i in range(length):
            latent += [styles[i+1],styles[i+1]]

        out = self.pose_encoder(data_pack['tar_uv'])
        out = self.conv1(out, latent[0], noise=noise[0])
        skip = self.to_rgb1(out, latent[1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb  in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs#, self.to_silhouettes
        ):
            out = conv1(out, latent[i], noise=noise1)
            out = conv2(out, latent[i + 1], noise=noise2)
            skip = to_rgb(out, latent[i + 2], skip)
            i += 2

        image = skip

        return image

