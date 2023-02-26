# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch.nn as nn
import util.util as util
import torch.nn.functional as F
import models.networks as networks
from util.util_smpl_sym import FlowCalculator
from models.networks.correspondence import GlobalHDCorrespondence

from torch import autograd
from models.networks.op import conv2d_gradfix
from util.util_func import tensor_dilate

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class StyleGAN2TryonModel(torch.nn.Module):
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
        
        self.net_corr = GlobalHDCorrespondence(opt).to(self.device)
        # load correspondence model
        ckpt_corr = torch.load('./net_Corr.pth', map_location=self.device)
        self.net_corr.load_state_dict(ckpt_corr)

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
        optimizer_G = torch.optim.Adam(self.net['netG'].parameters(), lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(self.net['netD'].parameters(), lr=G_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch, rank=None):
        util.save_network(self.net['netG'].module, 'G', epoch, self.opt, rank)
        util.save_network(self.net['netD'].module, 'D', epoch, self.opt, rank)
        util.save_network(self.net['netG_ema'], 'G_ema', epoch, self.opt, rank)

    def initialize_networks(self, opt):
        net = {}
        net['netG'] = networks.define_G(opt, self.rank)
        net['netD'] = networks.define_D(opt, self.rank)
        if not opt.isTrain or (opt.continue_train and self.rank == 0):
            # net['netG'] = util.load_network(net['netG'], 'G', opt.which_epoch, opt, self.rank)
            # net['netD'] = util.load_network(net['netD'], 'D', opt.which_epoch, opt, self.rank)
    
            # net['netG'].load_state_dict(torch.load('./latest_net_G.pth', map_location=lambda storage, loc: storage))
            # net['netD'].load_state_dict(torch.load('./latest_net_D.pth', map_location=lambda storage, loc: storage))
            
            ckpt = torch.load('./epoch_15_iter_5000.pt', map_location=lambda storage, loc: storage)
            net['netG'].load_state_dict(ckpt["g"])
            net['netD'].load_state_dict(ckpt["d"])
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

    def forward(self, data, mode, opt=None):
        data_unpack = self.preprocess_input(data)
        generated_out = {}
        if mode == 'generate':
            loss, generated_out = self.compute_generator_loss(data_unpack)
            # out = {}
            # out['fake_image'] = generated_out['fake_image']
            # out['real'] = generated_out['real']
            # out['ref'] = generated_out['ref']
            # out['tar_uv'] = generated_out['tar_uv']
        elif mode == 'discriminate':
            loss, generated_out = self.compute_discriminator_loss(data_unpack)
        elif mode == 'r1_reg':
            return self.compute_r1_loss(data_unpack, opt)
        else:
            raise Exception("Mode should be generate or discriminate!")

        out = {}
        out['fake_image'] = generated_out['fake_image']
        out['real'] = generated_out['real']
        out['ref'] = generated_out['ref']
        out['tar_uv'] = generated_out['tar_uv']
        return loss, out

    def compute_generator_loss(self, data_unpack):
        requires_grad(self.net['netG'], True)
        requires_grad(self.net['netD'], False)
        
        G_losses = {}
        generate_out = self.generate_fake(data_unpack)   

        fake_img = generate_out['fake_image']

        fake_pred = self.net['netD'](fake_img, pose=generate_out['tar_uv'])
        g_loss = F.softplus(-fake_pred).mean()

        G_losses["g"] = g_loss
        g_l1 = self.criterionL1(fake_img, data_unpack['real'])
        G_losses['l1'] = g_l1 
        g_vgg, _ = self.criterionVGG(fake_img, data_unpack['real'])
        G_losses['vgg'] = g_vgg

        # g_bg = self.criterionL1(fake_img_nomask * (1-data_unpack['sil']), (1-data_unpack['sil']))
        # G_losses['bg'] = g_bg * 0.1
        return G_losses, generate_out   

    def generate_fake(self, data_unpack, test=False):
        img_size = 512
        generate_out = {}
        with torch.no_grad():
            corr_out = self.net_corr(data_unpack)
        data_unpack = {**data_unpack, **corr_out}

        if test:
            fake_image = self.net['netG_ema'](data_unpack)
        else:
            fake_image = self.net['netG'](data_unpack) * data_unpack['sil']

        generate_out['fake_image'] = F.interpolate(fake_image, size=(img_size, img_size), mode='bilinear')
        generate_out['real'] = F.interpolate(data_unpack['real'], size=(img_size, img_size), mode='bilinear')# F.grid_sample(data_unpack['cloth_ref'], upscale_flows[0], mode='bilinear', padding_mode='border')
        generate_out['ref'] = F.interpolate(data_unpack['ref'], size=(img_size, img_size), mode='bilinear') # warp_mask.repeat(1,3,1,1)
        generate_out['tar_uv'] = F.interpolate(data_unpack['tar_uv'], size=(img_size, img_size), mode='bilinear') # comp_cloth
        generate_out = {**generate_out}
        return generate_out

    def compute_discriminator_loss(self, data_unpack):
        requires_grad(self.net['netG'], False)
        requires_grad(self.net['netD'], True)

        generate_out = self.generate_fake(data_unpack)   

        D_losses = {}

        fake_pred = self.net['netD'](generate_out['fake_image'], pose=generate_out['tar_uv'])
        real_pred = self.net['netD'](data_unpack['real'], pose=generate_out['tar_uv'])
        
        # bz = fake_img.shape[0]
        # fake_real = torch.cat([fake_img, data_unpack['real']], dim=0)
        # pose = torch.cat([generate_out['tar_uv'], generate_out['tar_uv']], dim=0)
        # fake_real_pred = self.net['netD'](fake_real, pose=pose)
        # fake_pred, real_pred = fake_real_pred.split(bz, dim=0)

        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        d_loss = fake_loss.mean() + real_loss.mean() 
        D_losses["d"] = d_loss

        return D_losses, generate_out   


    def d_r1_loss(self, real_pred, real_img):
        with conv2d_gradfix.no_weight_gradients():
            grad_real, = autograd.grad(
                outputs=real_pred.sum(), inputs=real_img, create_graph=True
            )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty

    def compute_r1_loss(self, data_unpack, opt):
        D_losses = {}
        real_img = data_unpack['real']
        pose = data_unpack['tar_uv']

        real_img.requires_grad = True
        real_pred = self.net['netD'](real_img, pose=pose)
        r1_loss = self.d_r1_loss(real_pred, real_img)

        D_losses["r1"] = opt.r1 / 2 * r1_loss * opt.d_reg_every + 0 * real_pred[0]
        return D_losses

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def split_data_unpack(self, data_unpack, index):
        split_data = {}
        for key in data_unpack.keys():
            split_data[key] = data_unpack[key][index:index+1, ...]
        return split_data

    def forward_visualize(self, data):
        with torch.no_grad():
            data_unpack = self.preprocess_input(data, )
            fake_test_images = []
            for i in range(data_unpack['real'].shape[0]):
                split_data = self.split_data_unpack(data_unpack, i)
                fake_image = self.generate_fake(split_data, test=True)['fake_image']    
                fake_test_images.append(fake_image)
            fake_test_images = torch.cat(fake_test_images, dim=0)
            out = {}
            out['fake_image'] = fake_test_images
            out['real'] = data_unpack['real']
            out['ref'] = data_unpack['ref']
            out['tar_uv'] = data_unpack['tar_uv']
            out['src_uv'] = data_unpack['src_uv']
        return out

