# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=2000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')
        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        # training loss weights
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--which_perceptual', type=str, default='4_2', help='relu5_2 or relu4_2')
        parser.add_argument('--weight_perceptual', type=float, default=0.001)
        parser.add_argument('--weight_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--weight_contextual', type=float, default=1.0, help='ctx loss weight')
        parser.add_argument('--weight_fm_ratio', type=float, default=1.0, help='vgg fm loss weight comp with ctx loss')
        
        parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        parser.add_argument(
            "--d_reg_every",
            type=int,
            default=16,
            help="interval of the applying r1 regularization",
        )
        parser.add_argument(
                "--g_reg_every",
                type=int,
                default=4,
                help="interval of the applying path length regularization",
        )
        self.isTrain = True
        return parser
