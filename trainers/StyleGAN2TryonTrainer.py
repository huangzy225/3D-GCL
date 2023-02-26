import torch
import copy
import os
import util.util as util
from models.StyleGAN2Tryon_model import StyleGAN2TryonModel
from torch.nn.parallel import DistributedDataParallel as DDP


class Stylegan2Trainer():
    def __init__(self, opt, rank):
        self.opt = opt
        self.rank = rank
        self.device = torch.device('cuda', rank)
        self.model = StyleGAN2TryonModel(opt, rank)

        self.model.net['netG'] = self.model.net['netG'].to(self.device)
        self.model.net['netD'] = self.model.net['netD'].to(self.device)
        self.model.net['netG_ema'] = copy.deepcopy(self.model.net['netG']).eval()
        self.optimizer_G, self.optimizer_D = self.model.create_optimizers(opt)

        if not opt.isTrain or (opt.continue_train):
            # pass
            
            ckpt = torch.load('./epoch_15_iter_5000.pt', map_location=lambda storage, loc: storage)
            self.model.net['netG_ema'].load_state_dict(ckpt["g_ema"])
            self.optimizer_G.load_state_dict(ckpt["g_optim"])
            self.optimizer_D.load_state_dict(ckpt["d_optim"])

            # self.model.net['netG_ema'].load_state_dict(torch.load('./latest_net_G_ema.pth', map_location=lambda storage, loc: storage))
            # self.optimizer_G.load_state_dict(torch.load('./latest_optim_G.pth', map_location=lambda storage, loc: storage)['optim_state_dict'])
            # self.optimizer_D.load_state_dict(torch.load('./latest_optim_D.pth', map_location=lambda storage, loc: storage)['optim_state_dict'])
            
            # self.model.net['netG_ema'] = util.load_network(self.model.net['netG_ema'], 'G_ema', opt.which_epoch, opt, self.rank)
            # optm_state_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_optim_G.pth' % (opt.which_epoch))
            # optim_state = torch.load(optm_state_path, map_location=torch.device('cuda', rank))
            # self.optimizer_G.load_state_dict(optim_state['optim_state_dict'])
            # optm_state_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_optim_D.pth' % (opt.which_epoch))
            # optim_state = torch.load(optm_state_path, map_location=torch.device('cuda', rank))
            # self.optimizer_D.load_state_dict(optim_state['optim_state_dict'])

        self.old_lr = opt.lr

        if (opt.num_gpus > 1):
            if rank == 0:
                print(f'Distributing across {opt.num_gpus} GPUs...')
            self.model.net['netG'] = DDP(self.model.net['netG'], device_ids=[rank], 
                                            output_device=rank, broadcast_buffers=False)
            self.model.net['netD'] = DDP(self.model.net['netD'], device_ids=[rank], 
                                            output_device=rank, broadcast_buffers=False)

        self.g_losses = {}
        self.d_losses = {}

    def save_optim(self, opt, epoch):
        save_filename = '%s_optim_G.pth' % (epoch)
        save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
        torch.save({'optim_state_dict': self.optimizer_G.state_dict()}, save_path)

        save_filename = '%s_optim_D.pth' % (epoch)
        save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
        torch.save({'optim_state_dict': self.optimizer_D.state_dict()}, save_path)

    def run_generator_one_step(self, data):
        g_losses, out = self.model(data, mode='generate')
        g_loss = sum(g_losses.values())
        self.model.net['netG'].zero_grad()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.out = out

    def run_discriminator_one_step(self, data):
        d_losses, out = self.model(data, mode='discriminate')
        d_loss = sum(d_losses.values())
        self.model.net['netD'].zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses
        self.out = out
    
    def run_discriminator_regularize(self, data, opt):
        d_losses = self.model(data, mode='r1_reg', opt=opt)
        d_loss = sum(d_losses.values())
        self.model.net['netD'].zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def run_forward_visualize(self, data):
        out = self.model.forward_visualize(data)
        self.out = out

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.out['fake_image']

    def update_learning_rate(self, epoch):
        pass
    
    def save(self, epoch):
        self.model.save(epoch, self.rank)

    def accumulate(self, decay=0.999):
        par1 = dict(self.model.net['netG_ema'].named_parameters())
        par2 = dict(self.model.net['netG'].module.named_parameters())
        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
            
    # def update_learning_rate(self, epoch):
    #     if epoch > self.opt.niter:
    #         lrd = self.opt.lr / self.opt.niter_decay
    #         new_lr = self.old_lr - lrd
    #     else:
    #         new_lr = self.old_lr
    #     if new_lr != self.old_lr:
    #         new_lr_Corr = new_lr
    #     else:
    #         new_lr_Corr = self.old_lr
    #     for param_group in self.optimizer_Corr.param_groups:
    #         param_group['lr'] = new_lr_Corr
    #     if self.rank == 0:
    #         print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
    #     self.old_lr = new_lr
