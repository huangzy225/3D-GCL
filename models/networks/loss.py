# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input).type_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def feature_normalize(feature_in, eps=1e-10):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + eps
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm

class ContextualLoss_forward(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self, opt):
        super(ContextualLoss_forward, self).__init__()
        self.opt = opt
        return None

    def forward(self, X_features, Y_features, h=0.1, feature_centering=True):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # to normalized feature vectors
        if feature_centering:
            # if self.opt.PONO:
            X_features = X_features - Y_features.mean(dim=1).unsqueeze(dim=1)
            Y_features = Y_features - Y_features.mean(dim=1).unsqueeze(dim=1)
            # else:
            #     X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            #     Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)

        # X_features = X_features - Y_features.mean(dim=1).unsqueeze(dim=1)
        # Y_features = Y_features - Y_features.mean(dim=1).unsqueeze(dim=1)

        X_features = feature_normalize(X_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size * feature_size
        Y_features = feature_normalize(Y_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size * feature_size

        # X_features = F.unfold(
        #     X_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2
        # Y_features = F.unfold(
        #     Y_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        # d_norm = d
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-3)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)
        loss = -torch.log(CX)

        # contextual loss per batch
        # loss = torch.mean(loss)
        return loss


class VGGLoss(nn.Module):
    def __init__(self, device, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids
        self.contextual_forward_loss = ContextualLoss_forward(None)

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = torch.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = torch.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-3], 2), F.avg_pool2d(target[-3].detach(), 2))) * 2
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def forward(self, x, y, cx=False):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        cx_loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        if cx:
            cx_loss = self.get_ctx_loss([x_vgg[2], x_vgg[3], x_vgg[4]], [y_vgg[2], y_vgg[3], y_vgg[4]])
        return loss, cx_loss

class CorrectnessLoss(nn.Module):
    def __init__(self, device, layids = None):
        super(CorrectnessLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y, flow):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            scale_flow = F.interpolate(flow.permute(0,3,1,2), size=x_vgg[i].shape[2:], mode='bilinear')
            loss += self.weights[i] * self.criterion(F.grid_sample(x_vgg[i].detach(), scale_flow.permute(0,2,3,1), mode='bilinear', padding_mode='border'), 
                                                      y_vgg[i].detach())
            # loss += self.weights[i] * self.criterion(F.grid_sample(x_vgg[i].detach(), flow[-(i+1)], mode='bicubic', padding_mode='border'), 
            #                                          y_vgg[i].detach())
        return loss


class SMPLLoss(nn.Module):
    def __init__(self):
        super(SMPLLoss, self).__init__()

    def offset_to_inds(self, x, y):
        return (x * y.size()[3] + y).to(torch.long)
        
    def forward(self, corr_m, gt_flow, vis_mask, scale_ref=None):
        batch_size, c, h, w = gt_flow.shape
        gt_grid = torch.cat([(gt_flow[:,i:i+1,:,:] + 1.0) * (size - 1.0) / 2  for i, size in enumerate([w,h])], dim=1)
        floor_grid = torch.floor(gt_grid)
        offset_x = [0,0,1,1]
        offset_y = [0,1,0,1]
        filtered_corrs = []
        filtered_gts = torch.zeros(size=(batch_size, 4, h, w), device=corr_m.device)

        # compute weight
        filtered_gts[:,0:1,:,:] = (floor_grid[:,1:2,:,:] + 1 - gt_grid[:,1:2,:,:]) * (floor_grid[:,0:1,:,:] + 1 - gt_grid[:,0:1,:,:])
        filtered_gts[:,1:2,:,:] = (floor_grid[:,1:2,:,:] + 1 - gt_grid[:,1:2,:,:]) * (gt_grid[:,0:1,:,:] - floor_grid[:,0:1,:,:])
        filtered_gts[:,2:3,:,:] = (gt_grid[:,1:2,:,:] - floor_grid[:,1:2,:,:]) * (floor_grid[:,0:1,:,:] + 1 - gt_grid[:,0:1,:,:])
        filtered_gts[:,3:4,:,:] = (gt_grid[:,1:2,:,:] - floor_grid[:,1:2,:,:]) * (gt_grid[:,0:1,:,:] - floor_grid[:,0:1,:,:])

        scale = 64
        for i in range(4):
            x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[i], 0, h-1) 
            y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[i], 0, w-1)  
            idx = self.offset_to_inds(x, y)
            filtered_corr = torch.gather(corr_m.permute(0, 2, 1).view(batch_size, scale*scale, scale, scale), 1, idx)
            filtered_corrs.append(filtered_corr)
        filterd_corr_out = torch.cat(filtered_corrs, dim=1)

        # filtered_gt_out = torch.cat(filtered_gts, dim=1)
        return nn.L1Loss()(filterd_corr_out * vis_mask, filtered_gts * vis_mask)


class SMPLLossPixel(nn.Module):
    def __init__(self):
        super(SMPLLossPixel, self).__init__()

    def offset_to_inds(self, x, y):
        return (x * x.size()[3] + y).to(torch.long)
        
    def forward(self, corr_m, gt_flow, vis_mask, scale_ref):
        batch_size, c, h, w = gt_flow.shape
        gt_grid = torch.cat([(gt_flow[:,i:i+1,:,:] + 1.0) * (size - 1.0) / 2  for i, size in enumerate([w,h])], dim=1)
        floor_grid = torch.floor(gt_grid)
        offset_x = [0,0,1,1]
        offset_y = [0,1,0,1]
        filtered_corrs = []
        scale = 64

        # filtered correlation matrix
        for i in range(4):
            x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[i], 0, h-1) 
            y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[i], 0, w-1)  
            idx = self.offset_to_inds(x, y)
            filtered_corr = torch.gather(corr_m.permute(0, 2, 1).view(batch_size, scale*scale, scale, scale), 1, idx)
            filtered_corrs.append(filtered_corr)
        filterd_corr_out = torch.cat(filtered_corrs, dim=1)
        filtered_corr_m = torch.zeros(size=(batch_size, scale*scale, h, w), device=corr_m.device)
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[0], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[0], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_corr_m = filtered_corr_m.scatter_(dim=1, index=idx, src=filterd_corr_out[:,0:1,:,:])
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[1], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[1], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_corr_m = filtered_corr_m.scatter_(dim=1, index=idx, src=filterd_corr_out[:,1:2,:,:])
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[2], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[2], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_corr_m = filtered_corr_m.scatter_(dim=1, index=idx, src=filterd_corr_out[:,2:3,:,:])
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[3], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[3], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_corr_m = filtered_corr_m.scatter_(dim=1, index=idx, src=filterd_corr_out[:,3:4,:,:])
        filtered_corr_m = filtered_corr_m.view(batch_size, scale*scale, scale*scale).permute(0, 2, 1)

        # max index
        # topk_inds = torch.topk(corr_m, 1, dim=-1)[-1]
        # idx = topk_inds.permute(0, 2, 1).view(batch_size, 1, scale, scale)
        # # filterd_corr_out = torch.gather(corr_m.permute(0, 2, 1).view(batch_size, scale*scale, scale, scale), 1, idx)
        # filtered_corr_m = torch.zeros(size=(batch_size, 4096, h, w), device=corr_m.device)
        # filtered_corr_m = filtered_corr_m.scatter_(dim=1, index=idx, src=torch.ones(size=(batch_size, 1, scale, scale), device=corr_m.device))
        # filtered_corr_m = filtered_corr_m.view(batch_size, scale*scale, scale*scale).permute(0, 2, 1)

        # gt
        filtered_gts = torch.zeros(size=(batch_size, scale*scale, h, w), device=corr_m.device)
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[0], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[0], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_gts = filtered_gts.scatter_(dim=1, index=idx, src=(floor_grid[:,1:2,:,:] + 1 - gt_grid[:,1:2,:,:]) * (floor_grid[:,0:1,:,:] + 1 - gt_grid[:,0:1,:,:]))
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[1], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[1], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_gts = filtered_gts.scatter_(dim=1, index=idx, src=(floor_grid[:,1:2,:,:] + 1 - gt_grid[:,1:2,:,:]) * (gt_grid[:,0:1,:,:] - floor_grid[:,0:1,:,:]))
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[2], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[2], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_gts = filtered_gts.scatter_(dim=1, index=idx, src=(gt_grid[:,1:2,:,:] - floor_grid[:,1:2,:,:]) * (floor_grid[:,0:1,:,:] + 1 - gt_grid[:,0:1,:,:]))
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[3], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[3], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_gts = filtered_gts.scatter_(dim=1, index=idx, src=(gt_grid[:,1:2,:,:] - floor_grid[:,1:2,:,:]) * (gt_grid[:,0:1,:,:] - floor_grid[:,0:1,:,:]))
        filtered_gt_corrs = filtered_gts.view(batch_size, scale*scale, scale*scale).permute(0, 2, 1)

        # scale = 64
        # for i in range(4):
        #     x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[i], 0, h-1) 
        #     y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[i], 0, w-1)  
        #     idx = self.offset_to_inds(x, y)
        #     filtered_corr = torch.gather(filtered_gt_corrs.permute(0, 2, 1).view(batch_size, scale*scale, scale, scale), 1, idx)
        #     filtered_corrs.append(filtered_corr)
        # filterd_corr_out = torch.cat(filtered_corrs, dim=1)

        warp_smpl = torch.matmul(filtered_gt_corrs, scale_ref.view(batch_size, 3, scale * scale).permute(0, 2, 1)).permute(0, 2, 1).view(batch_size, 3, scale, scale)
        warp_corr = torch.matmul(filtered_corr_m, scale_ref.view(batch_size, 3, scale * scale).permute(0, 2, 1)).permute(0, 2, 1).view(batch_size, 3, scale, scale)
        
        # mse_corr = nn.MSELoss()(filtered_gt_corrs, corr_m)

        return warp_smpl, warp_corr


class TestSMPLLoss(nn.Module):
    def __init__(self):
        super(TestSMPLLoss, self).__init__()

    def offset_to_inds(self, x, y):
        return (x * x.size()[3] + y).to(torch.long)
        
    def forward(self, corr_m, gt_flow, vis_mask, scale_ref):
        batch_size, c, h, w = gt_flow.shape
        gt_grid = torch.cat([(gt_flow[:,i:i+1,:,:] + 1.0) * (size - 1.0) / 2  for i, size in enumerate([w,h])], dim=1)
        floor_grid = torch.floor(gt_grid)
        offset_x = [0,0,1,1]
        offset_y = [0,1,0,1]
        filtered_corrs = []
        filtered_gts = torch.zeros(size=(batch_size, 4096, h, w), device=corr_m.device)

        scale = 64
        for i in range(4):
            x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[i], 0, h-1) 
            y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[i], 0, w-1)  
            idx = self.offset_to_inds(x, y)
            filtered_corr = torch.gather(corr_m.permute(0, 2, 1).view(batch_size, scale*scale, scale, scale), 1, idx)
            filtered_corrs.append(filtered_corr)
        filterd_corr_out = torch.cat(filtered_corrs, dim=1)


        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[0], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[0], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_gts = filtered_gts.scatter_(dim=1, index=idx, src=(floor_grid[:,1:2,:,:] + 1 - gt_grid[:,1:2,:,:]) * (floor_grid[:,0:1,:,:] + 1 - gt_grid[:,0:1,:,:]))
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[1], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[1], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_gts = filtered_gts.scatter_(dim=1, index=idx, src=(floor_grid[:,1:2,:,:] + 1 - gt_grid[:,1:2,:,:]) * (gt_grid[:,0:1,:,:] - floor_grid[:,0:1,:,:]))
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[2], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[2], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_gts = filtered_gts.scatter_(dim=1, index=idx, src=(gt_grid[:,1:2,:,:] - floor_grid[:,1:2,:,:]) * (floor_grid[:,0:1,:,:] + 1 - gt_grid[:,0:1,:,:]))
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[3], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[3], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_gts = filtered_gts.scatter_(dim=1, index=idx, src=(gt_grid[:,1:2,:,:] - floor_grid[:,1:2,:,:]) * (gt_grid[:,0:1,:,:] - floor_grid[:,0:1,:,:]))

        scale = 64
        filtered_gt_corrs = filtered_gts.view(batch_size, scale*scale, scale*scale).permute(0, 2, 1)

        # scale = 64
        # for i in range(4):
        #     x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[i], 0, h-1) 
        #     y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[i], 0, w-1)  
        #     idx = self.offset_to_inds(x, y)
        #     filtered_corr = torch.gather(filtered_gt_corrs.permute(0, 2, 1).view(batch_size, scale*scale, scale, scale), 1, idx)
        #     filtered_corrs.append(filtered_corr)
        # filterd_corr_out = torch.cat(filtered_corrs, dim=1)

        filtered_cyc_gts = torch.zeros(size=(batch_size, 4096, h, w), device=corr_m.device)
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[0], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[0], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_cyc_gts = filtered_cyc_gts.scatter_(dim=1, index=idx, src=filterd_corr_out[:,0:1,:,:])
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[1], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[1], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_cyc_gts = filtered_cyc_gts.scatter_(dim=1, index=idx, src=filterd_corr_out[:,1:2,:,:])
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[2], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[2], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_cyc_gts = filtered_cyc_gts.scatter_(dim=1, index=idx, src=filterd_corr_out[:,2:3,:,:])
        x = torch.clamp(floor_grid[:,1:2,:,:] + offset_x[3], 0, h-1) 
        y = torch.clamp(floor_grid[:,0:1,:,:] + offset_y[3], 0, w-1)  
        idx = self.offset_to_inds(x, y)
        filtered_cyc_gts = filtered_cyc_gts.scatter_(dim=1, index=idx, src=filterd_corr_out[:,3:4,:,:])

        filtered_cyc_gt_corrs = filtered_cyc_gts.view(batch_size, scale*scale, scale*scale).permute(0, 2, 1)

        warp_smpl = torch.matmul(filtered_gt_corrs, scale_ref.view(batch_size, 3, scale * scale).permute(0, 2, 1)).permute(0, 2, 1).view(batch_size, 3, scale, scale)
        warp_corr = torch.matmul(filtered_cyc_gt_corrs, scale_ref.view(batch_size, 3, scale * scale).permute(0, 2, 1)).permute(0, 2, 1).view(batch_size, 3, scale, scale)

        from util_func import visualize_img
        visualize_img([warp_smpl * vis_mask, warp_corr * vis_mask, scale_ref])
        
        # filtered_gt_out = torch.cat(filtered_gts, dim=1)
        return nn.L1Loss()(vis_mask, vis_mask)
        # loss