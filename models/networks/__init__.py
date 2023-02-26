# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
from models.networks.generator import *
from models.networks.ContextualLoss import *
from models.networks.correspondence import *
from models.networks.ops import *
import util.util as util


def find_network_using_name(target_network_name, filename, add=True):
    target_class_name = target_network_name + filename if add else target_network_name
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)
    assert issubclass(network, BaseNetwork), \
       "Class %s should be a subclass of BaseNetwork" % network
    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()
    netG_cls = find_network_using_name(opt.netG, 'generator')
    parser = netG_cls.modify_commandline_options(parser, is_train)
    return parser


def create_network(cls, opt, rank=None):
    net = cls(opt)
    net.print_network(rank)
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(torch.device('cuda', rank))
    return net


def define_G(opt, rank=None):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt, rank)


def define_D(opt, rank=None):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt, rank)

def define_Corr(opt, rank=None):
    netCoor_cls = find_network_using_name(opt.netCorr, 'correspondence')
    return create_network(netCoor_cls, opt, rank)
