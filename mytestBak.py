import models.vgg as VGG 
from torch.autograd import Variable
import torchvision.transforms as transforms
import PIL.Image as Image
import TVLossBak
import torch

loader = transforms.Compose([transforms.Scale((224,224)), transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    image = loader(image)
    #image = image.view(-1, 3, 224, 224)
    image = Variable(image.view(-1, 3, 224, 224), requires_grad = True)
    #image = image.cuda(1)
    #image = image.unsqueeze(0)
    return image

image = image_loader("./temp.jpg")
#mymodel.cuda(1)
#output = mymodel(image)
tvloss = TVLossBak.TVL()
print("class of image in mystest is ",image.__class__())
output = tvloss(image)
xutput = ((output.sum(1) ** 0.5).sum()).backward()
print(output)
print(xutput)
