import torch
from models.GlobalCorrespondence_model_stage import GlobalCorrespondenceModel
from torch.nn.parallel import DistributedDataParallel as DDP


class GlobalCorrespondenceTrainer():
    def __init__(self, opt, rank, resume_epoch=0):
        self.opt = opt
        self.rank = rank
        self.device = torch.device('cuda', rank)

        self.pix2pix_model = GlobalCorrespondenceModel(opt, rank)
        self.pix2pix_model.to(self.device)

        self.optimizer_Corr = self.pix2pix_model.create_optimizers(opt)
        self.old_lr = opt.lr

        if (opt.num_gpus > 1):
            print(f'Distributing across {opt.num_gpus} GPUs...')
            self.pix2pix_model.net['netCorr'] = DDP(self.pix2pix_model.net['netCorr'], 
                                                    device_ids=[rank], find_unused_parameters=True)

        self.g_losses = {}
        self.d_losses = {}

    def run_generator_one_step(self, data):
        self.optimizer_Corr.zero_grad()
        g_losses, out = self.pix2pix_model(data)
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_Corr.step()
        self.g_losses = g_losses
        self.out = out

    def run_forward_visualize(self, data):
        out = self.pix2pix_model.forward_visualize(data)
        self.out = out

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.out['fake_image']

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model.save(epoch, self.rank)

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr
        if new_lr != self.old_lr:
            new_lr_Corr = new_lr
        else:
            new_lr_Corr = self.old_lr
        for param_group in self.optimizer_Corr.param_groups:
            param_group['lr'] = new_lr_Corr
        if self.rank == 0:
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
