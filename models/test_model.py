# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import models.networks as networks
from util.util_func import tensor_dilate


class TestModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt, rank):
        super().__init__()
        self.opt = opt
        self.rank = rank
        self.device = torch.device('cuda', rank)
        self.initialize_networks(opt)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.SMPL_Faces = np.load('./smpl/smpl_faces.npy')

    def initialize_networks(self, opt):
        self.netCorr = networks.define_Corr(opt, self.rank)
        self.netG = networks.define_G(opt, self.rank)

        netCorr_weights = torch.load('./net_Corr.pth')
        self.netCorr.load_state_dict(netCorr_weights)

        netG_weights = torch.load('./latest_net_G_ema_copy.pth')
        self.netG.load_state_dict(netG_weights)
        return

         
    def preprocess_input(self, data):
        parsing = data['parsing_array'].to(self.device)
        parsing_ref = data['parsing_ref_array'].to(self.device)

        cloth_ref_mask = (parsing_ref==5).to(torch.uint8) + (parsing_ref==6).to(torch.uint8) + \
                          (parsing_ref==7).to(torch.uint8) 
        head_mask = (parsing==1).to(torch.uint8) + (parsing==2).to(torch.uint8) + (parsing==13).to(torch.uint8) 
        lower_mask = (parsing==8).to(torch.uint8) + (parsing==9).to(torch.uint8) + (parsing==12).to(torch.uint8) +\
                     (parsing==16).to(torch.uint8) + (parsing==17).to(torch.uint8) +\
                     (parsing==18).to(torch.uint8) + (parsing==19).to(torch.uint8)
        head_mask = head_mask + lower_mask
        
        cloth_mask = (parsing==5).to(torch.uint8) + (parsing==6).to(torch.uint8) + \
                     (parsing==7).to(torch.uint8) 
        sil = 1 - (parsing==0).to(torch.uint8)
        cloth_mask = cloth_mask.float()
        cloth_ref_mask = cloth_ref_mask.float()
        sil = sil.float()

        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(self.device)

        image = data['image']
        ref = data['ref']

        cloth_ref_mask = cloth_ref_mask.unsqueeze(1)
        cloth_mask = cloth_mask.unsqueeze(1)
        head_mask = head_mask.unsqueeze(1) 
        sil = sil.unsqueeze(1)

        src_dense_uv = data['source_densepose'].to(torch.float32).to(self.device)
        tar_dense_uv = data['target_densepose'].to(torch.float32).to(self.device)

        palm_mask = (tar_dense_uv[:,:1,:,:] == 30).to(torch.uint8) +\
                    (tar_dense_uv[:,:1,:,:] == 40).to(torch.uint8)
        palm_mask = tensor_dilate(palm_mask)
        hand_mask = (parsing==14).to(torch.uint8) + (parsing==15).to(torch.uint8) 
        palm_mask = hand_mask.unsqueeze(1) * palm_mask

        head_mask = head_mask + palm_mask

        head = head_mask * image  + (1 - head_mask)
        cloth_ref = cloth_ref_mask * ref + (1 - cloth_ref_mask)

        sil = torch.zeros((sil.shape)).float().to(self.device)
        for b in range(sil.shape[0]):
            w = sil.shape[3]
            pad_width = (512-320) // 2
            sil[b][:, :, pad_width:w-pad_width] = 1 # mask out the padding

        image = image * sil

        data_unpack = {'real': image, 'ref': ref, 'cloth_ref': cloth_ref, 
                       'cloth_ref_mask': cloth_ref_mask, 'head': head, 
                       'src_uv': src_dense_uv, 'tar_uv':tar_dense_uv, 
                       'sil': sil, 'lower_mask': lower_mask.unsqueeze(1)}

        return data_unpack

    def forward(self, data, test_flow=False):
        data_unpack = self.preprocess_input(data, )
        out = {}
        with torch.no_grad():
            out = self.generate_fake(data_unpack, test_flow)
        return out

    def generate_fake(self, data_unpack, test_flow=False):
        img_size = 512
        generate_out = {}
        with torch.no_grad():
            corr_out = self.netCorr(data_unpack)
        
        if test_flow:
            _, _, height, width = data_unpack['cloth_ref'].shape
            flow_up = F.interpolate(corr_out['flow256'].permute(0,3,1,2), scale_factor=2, mode='bilinear').permute(0,2,3,1)
            warp_mask = F.interpolate(corr_out['warp_mask'], size=(height, width), mode='bilinear')
            warp_cloth = F.grid_sample(data_unpack['cloth_ref'], flow_up, align_corners=False) 
            warp_cloth = warp_cloth * warp_mask + (1 - warp_mask)
            # warp_cloth = F.interpolate(corr_out['warp_out'][-1], scale_factor=2, mode='bilinear')
            generate_out['fake_image'] = warp_cloth
            return generate_out

        data_unpack = {**data_unpack, **corr_out}
        fake_image = self.netG(data_unpack) * data_unpack['sil']

        generate_out['fake_image'] = F.interpolate(fake_image, size=(img_size, img_size), mode='bilinear')
        generate_out['real'] = F.interpolate(data_unpack['real'], size=(img_size, img_size), mode='bilinear')# F.grid_sample(data_unpack['cloth_ref'], upscale_flows[0], mode='bilinear', padding_mode='border')
        generate_out['ref'] = F.interpolate(data_unpack['ref'], size=(img_size, img_size), mode='bilinear') # warp_mask.repeat(1,3,1,1)
        generate_out['tar_uv'] = F.interpolate(data_unpack['tar_uv'], size=(img_size, img_size), mode='bilinear') # comp_cloth
        generate_out = {**generate_out}
        return generate_out

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake, real

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def forward_visualize(self, data):
        with torch.no_grad():
            data_unpack = self.preprocess_input(data, )
            generated_out = self.generate_fake(data_unpack, sync=True)   
            out = {}
            out['fake_image'] = generated_out['fake_image']
            out['input_semantics'] = data_unpack['input_semantics']
            out['fake_full'] = generated_out['fake_full']
            out['warp_gt'] = generated_out['warp_gt']
            out['vis_pose'] = generated_out['vis_pose']   # .repeat(1, 3, 1, 1)
        return out

