import torch
import torch.nn as nn
import models.vgg as VGG
from torch.autograd import Variable

if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False

class FeatureVectorLoss(nn.Module):
    def __init__(self):
        super(FeatureVectorLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.model = VGG.vgg16()
    def __call__(self, generated_B, real_B):
        if use_gpu:
            self.model.cuda(0)
        inputA = self.model(generated_B)
        inputB = self.model(real_B)
        target = Variable(inputB.data, requires_grad = False)
        return self.loss(inputA, target)
