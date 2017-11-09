from __future__ import division, print_function, unicode_literals
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import torch.nn as nn
import os
from torch.optim import lr_scheduler
import functools
from torch.nn import init
from image_pool import ImagePool
import util.util as util
from collections import OrderedDict
import TVLoss
import loss_FV

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

def print_net(net):
	params =0
	for param in net.parameters():
		params += param.numel()
	print(net)
	print('Total number of parameters in this network is %d' % params)

class AutoPainter(nn.Module):
    def __init__(self, opt):
        super(AutoPainter, self).__init__()
        self.opt = opt
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if use_gpu else torch.Tensor

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # Assuming norm_type = batch
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        # model  of Generator Net is unet_256
        self.GeneratorNet = Generator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer,use_dropout = not opt.no_dropout)
        if use_gpu:
            self.GeneratorNet.cuda()
        self.GeneratorNet.apply(init_weights)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # model  of Discriminator Net is basic
            self.DiscriminatorNet = Discriminator(opt.input_nc+ opt.output_nc, opt.ndf, n_layers = 3, norm_layer = norm_layer, use_sigmoid = use_sigmoid)
            if use_gpu:
                self.DiscriminatorNet.cuda()
            self.DiscriminatorNet.apply(init_weights)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.GeneratorNet, 'Generator', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.DiscriminatorNet, 'Discriminator', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.learning_rate = opt.lr
            # defining loss functions
            self.criterionGAN = GANLoss(use_lsgan = not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionFV = loss_FV.FeatureVectorLoss()
            self.criterionTV = TVLoss.TVL()

            self.MySchedulers = []  # initialising schedulers
            self.MyOptimizers = []  # initialising optimizers
            self.generator_optimizer = torch.optim.Adam(self.GeneratorNet.parameters(), lr=self.learning_rate, betas = (opt.beta1, 0.999))
            self.discriminator_optimizer = torch.optim.Adam(self.DiscriminatorNet.parameters(), lr=self.learning_rate, betas = (opt.beta1, 0.999))
            self.MyOptimizers.append(self.generator_optimizer)
            self.MyOptimizers.append(self.discriminator_optimizer)
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - opt.niter)/float(opt.niter_decay+1)
                return lr_l
            for optimizer in self.MyOptimizers:
                self.MySchedulers.append(lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda_rule))
                # assuming opt.lr_policy == 'lambda'


        print('<============ NETWORKS INITIATED ============>')
        print_net(self.GeneratorNet)
        if self.isTrain:
            print_net(self.DiscriminatorNet)
        print('<=============================================>')


    def save_network(self, network, network_label, epoch_label):
        save_path = "./saved_models/%s_net_%s.pth" % (epoch_label, network_label)
        torch.save(network.cpu().state_dict(), save_path)
        if use_gpu:
            network.cuda()

    def load_network(self, network, network_label, epoch_label):
        save_path = "./saved_models/%s_net_%s.pth" % (epoch_label, network_label)
        # torch.save(network.cpu().state_dict(), save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        for scheduler in self.MySchedulers:
            scheduler.step()
        lr = self.MyOptimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_input(self, input):
        self.input = input
        if self.opt.which_direction == 'AtoB':
            input_A = input['A']
            input_B = input['B']
            self.image_paths = input['A_paths']
        else:
            input_A = input['B']
            input_B = input['A']
            self.image_paths = input['B_paths']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.generated_B = self.GeneratorNet.forward(self.real_A)
        self.real_B = Variable(self.input_B)

    def get_image_paths(self):
        return self.image_paths

    def backward_Discriminator(self):
        # fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.generated_B), 1))
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.generated_B), 1))
        self.prediction_fake = self.DiscriminatorNet.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.prediction_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.prediction_real = self.DiscriminatorNet.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.prediction_real, False)

        self.loss_Discriminator = (self.loss_D_fake+ self.loss_D_real)*0.5
        self.loss_Discriminator.backward()

    def backward_Generator(self):

        fake_AB = torch.cat((self.real_A, self.generated_B), 1)
        prediction_fake = self.DiscriminatorNet.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(prediction_fake, True)

        self.loss_G_L1 = self.criterionL1(self.generated_B, self.real_B)*self.opt.lambda_A
        self.loss_G_FV = self.criterionFV(self.generated_B, self.real_B)*self.opt.lambda_B
        self.loss_G_TV = self.criterionTV(self.generated_B)*self.opt.lambda_C

        self.loss_Generator = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_FV +self.loss_G_TV
        self.loss_Generator.backward()

    def optimize_parameters(self):
        self.forward()

        self.discriminator_optimizer.zero_grad()
        self.backward_Discriminator()
        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()
        self.backward_Generator()
        self.generator_optimizer.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
            ('G_L1', self.loss_G_L1.data[0]),
            ('G_FV', self.loss_G_FV.data[0]),
            ('G_TV', self.loss_G_TV.data[0]),
            ('D_real', self.loss_D_real.data[0]),
            ('D_fake', self.loss_D_fake.data[0])
            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.generated_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.GeneratorNet, 'Generator', label)
        self.save_network(self.DiscriminatorNet, 'Discriminator', label)




class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Generator, self).__init__()

        # constructing the unet generator structure
        generator_block = UnetBlock(ngf*8, ngf*8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs -5):
            generator_block = UnetBlock(ngf*8, ngf*8, input_nc=None, submodule=generator_block, norm_layer=norm_layer, use_dropout=use_dropout)

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
        downrelu = nn.LeakyReLU(0.2, True)
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


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator, self).__init__()
        print("[here: ]Discriminator initialized")
        use_bias = norm_layer == nn.InstanceNorm2d

        self.model = nn.Sequential()
        self.model.add_module("conv_0", nn.Conv2d(input_nc, ndf, kernel_size = 4, stride = 2, padding = 1))
        self.model.add_module("relu_0", nn.LeakyReLU(0.2, True))

        factor = 1
        for n in range(1, n_layers):
            last = factor
            factor = 2**min(n,3)
            self.model.add_module("conv_"+str(n), nn.Conv2d(ndf*last, ndf*factor, kernel_size = 4, stride =2,  padding = 1, bias = use_bias))
            self.model.add_module("norm_" + str(n), norm_layer(ndf*factor))
            self.model.add_module("relu_"+str(n), nn.LeakyReLU(0.2, True))

        last = factor
        factor = 2**min(3,n_layers)
        self.model.add_module("conv_"+str(n_layers), nn.Conv2d(ndf*last, ndf*factor, kernel_size = 4, stride =1,  padding = 1, bias = use_bias))
        self.model.add_module("norm_" + str(n_layers), norm_layer(ndf*factor))
        self.model.add_module("relu_"+str(n_layers), nn.LeakyReLU(0.2, True))
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
