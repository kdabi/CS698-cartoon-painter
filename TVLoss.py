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
         print("type of image in TVLoss is ", image.__class__())
         t = TVLoss1.apply(image)
         return t

class MyReLU(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  def forward(self, input):
    """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
    self.save_for_backward(input)
    return input.clamp(min=0)

  def backward(self, grad_output):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    input, = self.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input
