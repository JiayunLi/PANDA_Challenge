## system package
import os, sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
sys.path.append('../')
import warnings
warnings.filterwarnings("ignore")
## general package
import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from fastai.vision import *
## custom package
from utiles.mishactivation import *
from utiles.hubconf import *
from utiles.mishactivation import *
from utiles.torchvisionSegmentation import deeplabv3_resnet50, deeplabv3_resnet101


class Model(nn.Module):
    __constants__ = ['aux_classifier']
    def __init__(self, arch='deeplabv3_resnet50', n=6, pre=True):
        super(Model, self).__init__()
        model_fn = {'deeplabv3_resnet101': deeplabv3_resnet101, 'deeplabv3_resnet50': deeplabv3_resnet50}
        # m = torch.hub.load('pytorch/vision:v0.6.0', arch, pretrained = pre)
        m = model_fn[arch](pretrained = pre)
        self.backbone = m.backbone
        # self.classifier = DeepLabHead(2048, n)
        # self.aux_classifier = FCNHead(1024, n)
        self.head = nn.Sequential(AdaptiveConcatPool2d(), Flatten(), nn.Linear(2 * 2048, 512),
                                  Mish(), nn.BatchNorm1d(512), nn.Dropout(0.5), nn.Linear(512, 1))


    def forward(self, x):
        bs, n, c, h, w = x.shape
        x = x.view(-1, c, h, w)  # x: bs*N x 3 x 128 x 128
        input_shape = (h, w)
        # contract: features is a dict of tensors
        features = self.backbone(x)
        x = features["out"] ## [bs * N, 2048, h/16, w/16]
        _, c, h, w = x.shape
        ### a simple head for isup regression
        y = x.view(bs, n, c, h, w).permute(0, 2, 1, 3, 4).contiguous() \
            .view(-1, c, h * n, w)  # x: bs x C x N*4 x 4
        y = self.head(y) # x: bs x n
        return y

if __name__ == "__main__":
    img = torch.rand([2, 5, 3, 256, 256])
    model = Model(arch='deeplabv3_resnet50')
    output = model(img)
    print(output.shape)