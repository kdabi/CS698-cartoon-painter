from __future__ import division, print_function, unicode_literals
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import os

if torch.cuda.is_available():
    use_gpu = True

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

# from .base_model import BaseModel
class Pix2Pix(nn.Module):
	def __init__(self, opt):
		super(Pix2Pix, self).__init__()
		self.opt = opt

		self.input_A = self.Tensor()
		self.input_B = self.Tensor()

                norm_layer = get_norm_layer(norm_type=norm)
                self.GeneraterNet = Generator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout= use_dropout)
                if use_gpu:
                    self.GeneraterNet.cuda()
                self.GeneraterNet.apply(init_weights)



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


        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0,2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
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
