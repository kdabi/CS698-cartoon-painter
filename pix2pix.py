from __future__ import division, print_function, unicode_literals
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import os

# from .base_model import BaseModel
class Pix2Pix(nn.Module):
	def __init__(self, opt):
		super(Pix2Pix, self).__init__()
		self.opt = opt

		self.input_A = self.Tensor()
		self.input_B = self.Tensor()







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

class UnetBlockGenerater(nn.Module):
	def __init__(self, outer_nc, inner_nc, )


	