import torch
import torch.nn as nn
import torchvision.transforms as transforms

class TVLoss1(torch.autograd.Function):
    @staticmethod    
    def forward(ctx,image):
        ctx.save_for_backward(image)
        return  ((image[:,:,0:-1,0:-1] - image[:,:,1:,0:-1])**2 + (image[:,:,0:-1,0:-1] - image[:,:,0:-1,1:])**2)

    @staticmethod
    def backward(ctx, grad_output):
        image, = ctx.saved_variables
        print(grad_output.__class__(), image.__class__())
        return ((grad_output[:,:,0:-1,0:-1] - grad_output[:,:,1:,0:-1])**2 + (grad_output[:,:,0:-1,0:-1] - grad_output[:,:,0:-1,1:])**2)

class TVL(nn.Module):
    def __init__(self):
         super(TVL,self).__init__()
    def forward(self, image):
         return  ((((image[:,:,0:-1,0:-1] - image[:,:,1:,0:-1])**2 + (image[:,:,0:-1,0:-1] - image[:,:,0:-1,1:])**2).sum(1))**0.5).sum()
