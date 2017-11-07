import torch
import torch.nn as nn
class TVLoss(torch.autograd.Function):
    @staticmethod    
    def forward(ctx,image):
        ctx.save_for_backward(image)
        return ((image[:,:,0:-1,0:-1] - image[:,:,1:,0:-1])**2 + (image[:,:,0:-1,0:-1] - image[:,:,0:-1,1:])**2)

    @staticmethod
    def backward(ctx, grad_output):
        image, = ctx.saved_variables
        print("type of grad_output is ",grad_output.__class__(), image.__class__())
        return grad_output * image[:,:,0:-1,0:-1]

class TVL(nn.Module):
    def __init__(self):
         super(TVL,self).__init__()
    def forward(self, image):
         print("type of image in TVLoss is ", image.__class__())
         t = TVLoss.apply(image)
         return t
