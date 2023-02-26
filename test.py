# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import dataset
from util.util import mkdir
from models.test_model import TestModel
from torchvision.utils import save_image
from options.test_options import TestOptions


if __name__ == '__main__':
    opt = TestOptions().parse()
    dataloader = dataset.create_dataloader(opt)
    model = TestModel(opt, rank=opt.gpu_ids[0])
    model.to(opt.gpu_ids[0])
    model.eval()
    save_root = os.path.join(opt.results_dir, opt.name)
    if not os.path.exists(save_root):
        mkdir(save_root)
    for i, data_i in enumerate(dataloader):
        print('{} / {}'.format(i, len(dataloader)))
        out = model(data_i, test_flow=True)
        imgs_num = data_i['image'].shape[0]
        for it in range(imgs_num):
            save_name = os.path.join(save_root, '%08d_%04d.png' % (i, it))
            save_image(out['fake_image'][it:it+1], save_name, padding=0, normalize=True)
