from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
import functools
import torch
import os
import torch.nn as nn
from torch.nn import init
import numpy as np
import pix2pix


if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False


# Assuming init_type = xavier
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class TestModel(nn.Module):
    def __init__(self,opt):
        super(TestModel, self).__init__()
        self.opt = opt

        assert(not opt.isTrain)
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if use_gpu else torch.Tensor

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)


	# Assuming norm_type = batch
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        # model  of Generator Net is unet_256
        self.GeneratorNet = pix2pix.Generator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer,use_dropout = not opt.no_dropout)
        if use_gpu:
             self.GeneratorNet.cuda()
        self.GeneratorNet.apply(init_weights)

        which_epoch = opt.which_epoch
        self.load_network(self.GeneratorNet, 'Generator', which_epoch)

        print('<========= NETWORKS INITIATED ======>')
        pix2pix.print_net(self.GeneratorNet)
        print('<===================================>')

    def save_network(self, network, network_label, epoch_label):
        save_path = "./saved_models/%s_net_%s.pth" % (epoch_label,        network_label)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda()

    def load_network(self, network, network_label, epoch_label):
        save_path = "./saved_models/%s_net_%s.pth" % (epoch_label,        network_label)
        # torch.save(network.cpu().state_dict(), save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        for scheduler in self.MySchedulers:
            scheduler.step()
        lr = self.MyOptimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_input(self,input):
        input_A = input['B']
        input_B = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.generated_B = self.GeneratorNet.forward(self.real_A)

    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.generated_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B) ])
