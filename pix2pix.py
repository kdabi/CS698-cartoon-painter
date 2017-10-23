from __future__ import division, print_function, unicode_literals
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from torch.optim import lr_scheduler
import functools
from torch.nn import init

if torch.cuda.is_available():
    use_gpu = True

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

def print_net(net):
	params =0
	for param in net.parameters():
		params += param.numel()
	print(net)
	print('Total number of parameters in this network is %d' % params)

class Pix2Pix(nn.Module):
    def __init__(self, opt):
        super(Pix2Pix, self).__init__()
        self.opt = opt

        self.input_A = self.Tensor()
        self.input_B = self.Tensor()

        # Assuming norm_type = batch
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        self.GeneraterNet = Generator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer,not opt.no_dropout)
        if use_gpu:
            self.GeneraterNet.cuda()
        self.GeneraterNet.apply(init_weights)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.DiscriminatorNet = Discriminator(opt.input_nc+ opt.output_nc, opt.ndf, 3, norm_layer, use_sigmoid = use_sigmoid)
            if use_gpu:
                self.DiscriminatorNet.cuda()
            self.DiscriminatorNet.apply(init_weights)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.GeneraterNet, 'Generater', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.DiscriminatorNet, 'Discriminator', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.learning_rate = opt.lr
            # defining loss functions
            self.criterionGAN = GANLoss(use_lsgan = not opt.no_lsgan, tensor=self.Tensor)
            self.criterianL1 = torch.nn.L1Loss()

            self.MySchedulers = []  # initialising schedulers
            self.MyOptimizers = []  # initialising optimizers
            self.generater_optimizer = torch.optim.Adam(self.GeneraterNet.parameters(), lr=self.learning_rate, betas = (opt.beta1, 0.999))
            self.discriminator_optimizer = torch.optim.Adam(self.DiscriminatorNet.parameters(), lr=self.learning_rate, betas = (opt.beta1, 0.999))
            self.MyOptimizers.append(self.generator_optimizer)
            self.MyOptimizers.append(self.discriminator_optimizer)
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - opt.niter)/float(opt.niter_decay+1)
            for optimizer in self.MyOptimizers:
                self.MySchedulers.append(lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda_rule))
                # assuming opt.lr_policy == 'lambda'


        print('<============ NETWORKS INITIATED ============>')
        print_net(self.GeneraterNet)
        if self.isTrain:
            print_net(self.DiscriminatorNet)
        print('<=============================================>')





    def save_network(self, network, network_label, epoch_label):
        save_path = "./saved_models/%s_net_%s.pth" % (epoch_label, network_label)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda()

    def load_network(self, network, network_label, epoch_label):
        save_path = "./saved_models/%s_net_%s.pth" % (epoch_label, network_label)
        # torch.save(network.cpu().state_dict(), save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        pass



class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # constructing the unet generator structure
        generator_block = UnetBlock(ngf*8, ngf*8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_down -5):
            generator_block = UnetBlock(ngf*8, ngf*8, input_nc=None, submodule=generator_block, norm_layer=norm_layer, use_drop=use_dropout)

        generator_block = UnetBlock(ngf*4, ngf*8, input_nc=None,submodule=generator_block, norm_layer=norm_layer)
        generator_block = UnetBlock(ngf*2, ngf*4, input_nc=None,submodule=generator_block, norm_layer=norm_layer)
        generator_block = UnetBlock(ngf, ngf*2, input_nc=None,submodule=generator_block, norm_layer=norm_layer)
        generator_block = UnetBlock(output_nc, ngf, input_nc=input_nc,submodule=generator_block, outermost=True, norm_layer=norm_layer)

        self.model = generator_block

    def forward(self, input):
        return self.model(input)


class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc = None, submodule = None, outermost=False,innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc


        self.model = nn.Sequential()
        # self.model.add_module("downconv" , nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1))
        # self.model.add_module("downrelu", nn.LeakyReLU(0,2, True))
        # self.model.add_module("downnorm" ,norm_layer(inner_nc))
        # self.model.add_module("uprelu", nn.ReLU(True))
        # self.model.add_module("upnorm", norm_layer(outer_nc))


        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0,2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            self.model.add_module("downconv_" + str(outer_nc) , nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1))

            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down+[submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost :
            return self.model(x)
        else :
            return torch.cat([x, self.model(x)], 1)


<<<<<<< 02dfe8e8ea82b50d00c8cd4b875a3e3dc8a751ff
=======
    
>>>>>>> Added GANloss class
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        self.model = nn.Sequential()
        self.model.add_module("conv_0", nn.Conv2d(input_nc, ndf, kernel_size = 4, stride = 2, padding = 1))
        self.model.add_module("relu_0", nn.ReLU(0.2, True))

        factor = 1
        for n in range(1, n_layers):
            last = factor
            factor = 2**min(n,3)
            self.model.add_module("conv_"+str(n), nn.Conv2d(ndf*last, ndf*factor, kernel_size = 4, stride =2,  padding = 1, use_bias = use_bias))
            self.model.add_module("norm_" + str(n), norm_layer(ndf*factor))
            self.model.add_module("relu_"+str(n), nn.LeakyRelu(0.2, True))

        last = factor
        factor = 2**min(3,n_layers)
        self.model.add_module("conv_"+str(n_layers), nn.Conv2d(ndf*last, ndf*factor, kernel_size = 4, stride =1,  padding = 1, use_bias = use_bias))
        self.model.add_module("norm_" + str(n_layers), norm_layer(ndf*factor))
        self.model.add_module("relu_"+str(n_layers), nn.LeakyRelu(0.2, True))
        self.model.add_module("conv_"+str(n_layers+1), nn.Conv2d(ndf*factor, 1, kernel_size = 4, stride = 1, padding = 1))
        if use_sigmoid:
            self.model.add_module("sigmoid", nn.Sigmoid())

        def forward(self, input):
            return self.model(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan = True, target_real_label = 1.0, target_fake_label = 0.0, tensor = torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan :
            self.loss = nn.MSELoss()
        else :
            self.loss = nn.BCELoss()

    def __call__(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad = False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad = False)
            target_tensor = self.fake_label_var
        return self.loss(input, target_tensor)