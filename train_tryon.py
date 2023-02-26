import os
import cv2
import time 
import dataset
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard

from util.util import print_current_errors, log_errors
from options.train_options import TrainOptions
from util.iter_counter import IterationCounter
from trainers.StyleGAN2TryonTrainer import Stylegan2Trainer


def save_test(imgs, save_name, nrow=8, padding=0, normalize=True):
    imgs = imgs.detach().cpu().numpy() * 0.5 + 0.5
    gw, gh = 10,5
    _, C, H, W = imgs.shape
    imgs = np.reshape(imgs, (gh, gw, C, H, W))
    imgs = imgs.transpose(0, 3, 1, 4, 2)
    imgs = imgs.reshape(gh * H, gw * W, C)
    imgs = (np.clip(imgs, 0., 1.)*255).astype(np.uint8)
    cv2.imwrite(save_name, imgs[:, :, ::-1])
    

def training_loop(rank, opt):
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    dataloader, len_dataloader, visualize_data = dataset.create_ddp_dataloader(rank, opt)        
    iter_counter = IterationCounter(opt, len_dataloader, rank)
    trainer = Stylegan2Trainer(opt, rank)
    save_root = os.path.join('checkpoints', opt.name, 'train')
    if not os.path.exists(save_root) and rank == 0:
        os.makedirs(save_root)
    if rank == 0:
        writer_dir = os.path.join('checkpoints', opt.name)
        writer = tensorboard.SummaryWriter(writer_dir)
    start_time = time.time()

    random.seed(0)

    for epoch in iter_counter.training_epochs():
        opt.epoch = epoch
        iter_counter.record_epoch_start(epoch)
        step = 0
        for i, data_i in enumerate(dataloader):        
            iter_counter.record_one_iteration()

            trainer.run_discriminator_one_step(data_i)
            d_regularize = i % opt.d_reg_every == 0
            if d_regularize:
                trainer.run_discriminator_regularize(data_i, opt)

            trainer.run_generator_one_step(data_i)
            trainer.accumulate()
            
            if rank == 0:
                losses = trainer.get_latest_losses()
                log_errors(writer, losses, start_time, iter_counter.total_steps_so_far)
                if iter_counter.needs_printing():
                    print_current_errors(opt, writer, epoch, iter_counter.epoch_iter, iter_counter.epoch_iter_num, 
                                            losses, iter_counter.time_per_iter, start_time, iter_counter.total_steps_so_far)

                if iter_counter.needs_displaying():
                    imgs_num = visualize_data['label'].shape[0]
                    trainer.run_forward_visualize(visualize_data)
                    imgs = torch.cat((visualize_data['ref'].cpu(), 
                                    trainer.get_latest_generated().data.cpu(), 
                                    trainer.out['src_uv'].data.cpu(),
                                    trainer.out['tar_uv'].data.cpu(),
                                    visualize_data['image'].cpu()), 0)
                                
                    save_name = '%08d_%08d.png' % (epoch, iter_counter.total_steps_so_far)
                    save_name = os.path.join(save_root, save_name)
                    save_test(imgs, save_name, nrow=imgs_num, padding=0, normalize=True)

                if iter_counter.needs_saving():     
                    print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, iter_counter.total_steps_so_far))
                    trainer.save('latest')  
                    trainer.save_optim(opt, 'latest')                      
                
                    iter_counter.record_current_iter()

            step += 1

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if rank == 0 and epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, iter_counter.total_steps_so_far))
            trainer.save(epoch)
            trainer.save_optim(opt, epoch)                      

    if rank == 0:
        print('Training was successfully finished.')
        writer.close()


def subprocess_fn(rank, opt):
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=opt.num_gpus)
    training_loop(rank=rank, opt=opt)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.num_gpus = len(opt.gpu_ids)
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    if opt.num_gpus == 1:
        subprocess_fn(rank=0, opt=opt)
    else:
        torch.multiprocessing.spawn(fn=subprocess_fn, args=(opt,), nprocs=opt.num_gpus)
