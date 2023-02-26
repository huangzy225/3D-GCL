# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch.nn as nn
import util.util as util
import torch.nn.functional as F
import models.networks as networks
from util.util_smpl_sym import FlowCalculator


class GlobalCorrespondenceModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt, rank):
        super().__init__()
        self.opt = opt
        self.rank = rank
        self.device = torch.device('cuda', rank)
        self.net = torch.nn.ModuleDict(self.initialize_networks(opt))
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        if opt.isTrain:
            self.criterionVGG = networks.VGGLoss(device=self.device)
            self.criterionSMPL = networks.SMPLLossPixel()
            self.contextual_forward_loss = networks.ContextualLoss_forward(opt)
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.MSE_loss = torch.nn.MSELoss()
            self.criterionL1 = nn.L1Loss()
            if opt.which_perceptual == '5_2':
                self.perceptual_layer = -1
            elif opt.which_perceptual == '4_2':
                self.perceptual_layer = -2
        self.SMPL_Faces = np.load('./smpl/smpl_faces.npy')
        self.SMPLFlowPredictor = FlowCalculator()

    def create_optimizers(self, opt):
        beta1, beta2 = opt.beta1, opt.beta2
        G_lr = opt.lr
        optimizer_Corr = torch.optim.Adam(self.net['netCorr'].parameters(), lr=G_lr, betas=(beta1, beta2))
        return optimizer_Corr

    def save(self, epoch, rank=None):
        util.save_network(self.net['netCorr'].module, 'Corr', epoch, self.opt, rank)

    def initialize_networks(self, opt):
        net = {}
        net['netCorr'] = networks.define_Corr(opt, self.rank)
        if not opt.isTrain or (opt.continue_train and self.rank == 0):
            net['netCorr'] = util.load_network(net['netCorr'], 'Corr', opt.which_epoch, opt, self.rank)
        return net

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = torch.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = torch.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-3], 2), F.avg_pool2d(target[-3].detach(), 2))) * 2
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def get_tv_loss(self, x):
        tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))
        
    def preprocess_input(self, data):
        parsing = data['parsing_array'].to(self.device)
        parsing_ref = data['parsing_ref_array'].to(self.device)

        cloth_ref_mask = (parsing_ref==5).to(torch.uint8) + (parsing_ref==6).to(torch.uint8) + \
                          (parsing_ref==7).to(torch.uint8) 
        head_mask = (parsing==1).to(torch.uint8) + (parsing==2).to(torch.uint8) + (parsing==13).to(torch.uint8)      
        cloth_mask = (parsing==5).to(torch.uint8) + (parsing==6).to(torch.uint8) + \
                     (parsing==7).to(torch.uint8) 

        cloth_mask = cloth_mask.float()
        cloth_ref_mask = cloth_ref_mask.float()

        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(self.device)

        # label = data['label'][:,:3,:,:].float()
        # input_semantics = data['label'].float()
        image = data['image']

        # ref_label = data['ref_label'][:,:3,:,:].float()
        # ref_semantics = data['ref_label'].float()
        ref = data['ref']

        cloth_ref_mask = cloth_ref_mask.unsqueeze(1)
        cloth_mask = cloth_mask.unsqueeze(1)
        head_mask = head_mask.unsqueeze(1) 

        cloth = cloth_mask * image + (1 - cloth_mask)
        head = head_mask * image  + (1 - head_mask)

        src_verts = data['source_vertices_img'].to(self.device)
        src_vis = data['source_visibility'].to(self.device)
        tar_verts = data['target_vertices_img'].to(self.device)
        tar_vis = data['target_visibility'].to(self.device)

        flow_A2B, vis_mask_B = self.SMPLFlowPredictor(tar_verts, tar_vis, src_verts, 
                                               src_vis, self.SMPL_Faces, data['symmetric_mask'])
        vis_mask_B = (vis_mask_B.squeeze(3) == 0).to(torch.float32)
        vis_mask_B = vis_mask_B.unsqueeze(1)
        vis_mask_B = torch.floor(F.grid_sample(cloth_ref_mask, flow_A2B, mode='bilinear')) * vis_mask_B

        cloth_ref = cloth_ref_mask * ref + (1 - cloth_ref_mask)
        symmetric_mask = data['symmetric_mask'].to(self.device).unsqueeze(1)
        src_dense_uv = data['source_densepose'].to(torch.float32).to(self.device)
        tar_dense_uv = data['target_densepose'].to(torch.float32).to(self.device)

        data_unpack = {'real_image': image, 'ref': ref, 'cloth_ref': cloth_ref, 'cloth_ref_mask': cloth_ref_mask, 'cloth': cloth,
                        'cloth_mask': cloth_mask, 'head': head, 'flow_A2B': flow_A2B, 'vis_mask_B': vis_mask_B, 
                        'src_uv': src_dense_uv, 'tar_uv':tar_dense_uv, 'symmetric_mask':symmetric_mask,
                        'parsing': parsing}

        return data_unpack

    def forward(self, data):
        data_unpack = self.preprocess_input(data)
        generated_out = {}
        g_loss, generated_out = self.compute_generator_loss(data_unpack)
        out = {}
        out['fake_image'] = generated_out['fake_image']
        out['fake128'] = generated_out['fake128']
        out['fake256'] = generated_out['fake256']
        out['fake512'] = generated_out['fake512']
        return g_loss, out

    def compute_generator_loss(self, data_unpack):
        G_losses = {}
        generate_out = self.generate_fake(data_unpack)   

        loss_l1_64 = 0
        loss_vgg_64 = 0
        loss_smpl_64 = 0

        loss_l1 = 0
        loss_vgg = 0
        loss_smpl = 0

        # 64 resolution constrain
        max_num = 3
        num = 0
        scale_vis_mask = F.interpolate(data_unpack['vis_mask_B'], scale_factor=0.5 ** (max_num - num), mode='bilinear')
        scale_cloth = F.interpolate(data_unpack['cloth'], scale_factor=0.5 ** (max_num - num), mode='bilinear')
        scale_cloth_mask = F.interpolate(data_unpack['cloth_mask'], scale_factor=0.5 ** (max_num - num), mode='bilinear')
        scale_cloth_ref = F.interpolate(data_unpack['cloth_ref'], scale_factor=0.5 ** (max_num - num), mode='bilinear')
        warp_cloth = generate_out['warp_out'][num] 

        loss_l1_64 += self.criterionL1(warp_cloth, scale_cloth)
        loss_vgg_64 = self.criterionVGG(warp_cloth * scale_cloth_mask, scale_cloth * scale_cloth_mask, cx=False)[0]
        loss_vgg_64 = loss_vgg_64 * 0.2
        
        scale_flow = F.interpolate(data_unpack['flow_A2B'].permute(0, 3, 1, 2), scale_factor=0.5 ** (max_num - num), mode='bilinear')
        warp_smpl, warp_corr = self.criterionSMPL(generate_out['corr_m'], scale_flow, scale_vis_mask, scale_cloth_ref)
        loss_smpl_64 = self.criterionL1(warp_smpl * scale_vis_mask, warp_corr * scale_vis_mask) + \
                       self.criterionVGG(warp_smpl * scale_vis_mask, warp_corr * scale_vis_mask, cx=False)[0] * 0.2

        G_losses['smpl_64'] = loss_smpl_64 * 10
        G_losses['l1_64'] = loss_l1_64 * 1
        G_losses['vgg_64'] = loss_vgg_64 * 1
        # G_losses['mse_corr'] = mse_corr * 200

        # scale_cloth_ref_mask = F.interpolate(data_unpack['cloth_ref_mask'], scale_factor=0.5 ** (max_num - num), mode='bilinear')
        # warp_cycle = generate_out['y_cycle']
        # loss_l1_64_cyc = self.criterionL1(warp_cycle, scale_cloth_ref)
        # loss_vgg_64_cyc, _ = self.criterionVGG(warp_cycle * scale_cloth_ref_mask, scale_cloth_ref * scale_cloth_ref_mask, cx=False)
        # loss_cyc = loss_l1_64_cyc + loss_vgg_64_cyc * 0.2
        # G_losses['loss_cyc'] = loss_cyc

        # 128 resolution constrain
        max_num = 2
        num = 0
        scale_vis_mask = F.interpolate(data_unpack['vis_mask_B'], scale_factor=0.5 ** (max_num - num), mode='bilinear')
        cur_person_clothes = F.interpolate(data_unpack['cloth'], scale_factor=0.5 ** (max_num - num), mode='bilinear')

        scale_cloth_ref = F.interpolate(data_unpack['cloth_ref'], scale_factor=0.5 ** (max_num - num), mode='bilinear')
        warp_cloth = generate_out['warp_out'][1] 
        scale_cloth_mask = F.interpolate(data_unpack['cloth_mask'], scale_factor=0.5 ** (max_num - num), mode='bilinear')

        loss_l1 += nn.L1Loss()(warp_cloth, cur_person_clothes) 
        vgg_loss, _ = self.criterionVGG(warp_cloth * scale_cloth_mask, cur_person_clothes * scale_cloth_mask, cx=False)
        loss_vgg += vgg_loss* 0.2
        # loss_full_cx += cx_loss

        scale_flow = F.interpolate(data_unpack['flow_A2B'].permute(0, 3, 1, 2), scale_factor=0.5 ** (max_num - num), mode='bilinear')
        loss_smpl += nn.L1Loss()(generate_out['flow128'].permute(0,3,1,2) * scale_vis_mask, 
                                 scale_flow.repeat(1,1,1,1) * scale_vis_mask)

        # 256 resolution constrain
        max_num = 1
        num = 0
        scale_vis_mask = F.interpolate(data_unpack['vis_mask_B'], scale_factor=0.5 ** (max_num - num), mode='bilinear')
        cur_person_clothes = F.interpolate(data_unpack['cloth'], scale_factor=0.5 ** (max_num - num), mode='bilinear')

        scale_cloth_ref = F.interpolate(data_unpack['cloth_ref'], scale_factor=0.5 ** (max_num - num), mode='bilinear')
        warp_cloth = generate_out['warp_out'][2] 
        scale_cloth_mask = F.interpolate(data_unpack['cloth_mask'], scale_factor=0.5 ** (max_num - num), mode='bilinear')

        loss_l1 += nn.L1Loss()(warp_cloth, cur_person_clothes) 
        vgg_loss, _ = self.criterionVGG(warp_cloth * scale_cloth_mask, cur_person_clothes * scale_cloth_mask, cx=False)
        loss_vgg += vgg_loss * 0.2

        scale_flow = F.interpolate(data_unpack['flow_A2B'].permute(0, 3, 1, 2), scale_factor=0.5 ** (max_num - num), mode='bilinear')
        loss_smpl += nn.L1Loss()(generate_out['flow256'].permute(0,3,1,2) * scale_vis_mask, 
                                 scale_flow.repeat(1,1,1,1) * scale_vis_mask) 

        G_losses['smpl'] = loss_smpl * 30
        G_losses['full_l1'] = loss_l1 * 5
        G_losses['full_vgg'] = loss_vgg * 5
        
        # G_losses['full_cx'] = loss_full_cx * 0.01
        # G_losses['smooth'] = generate_out['smooth'] * 0.01
        # G_losses['2nd_smooth'] = generate_out['2nd_smooth'] * 0.05
        return G_losses, generate_out   

    def generate_fake(self, data_unpack):
        img_size = 512
        generate_out = {}
        corr_out = self.net['netCorr'](data_unpack)
        generate_out['fake_image'] = F.interpolate(corr_out['warp_out'][0], size=(img_size, img_size), mode='bilinear')
        generate_out['fake128'] = F.interpolate(corr_out['warp_out'][1], size=(img_size, img_size), mode='bilinear')# F.grid_sample(data_unpack['cloth_ref'], upscale_flows[0], mode='bilinear', padding_mode='border')
        generate_out['fake256'] = F.interpolate(corr_out['warp_out'][2], size=(img_size, img_size), mode='bilinear') # warp_mask.repeat(1,3,1,1)
        generate_out['fake512'] = F.interpolate(corr_out['warp_out'][2], size=(img_size, img_size), mode='bilinear') # comp_cloth
        generate_out = {**generate_out, **corr_out}
        return generate_out

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def forward_visualize(self, data):
        with torch.no_grad():
            data_unpack = self.preprocess_input(data, )
            self.net['netCorr'].zero_grad()
            generated_out = self.generate_fake(data_unpack)   
            out = {}
            out['fake_image'] = generated_out['fake_image']
            out['fake128'] = generated_out['fake128']
            out['fake256'] = generated_out['fake256']
            out['fake512'] = generated_out['fake512'] 
            out['gt'] = data_unpack['real_image']
        return out

