## system package
import os, sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
sys.path.append('../')
sys.path.append('../model')
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")

import torch.nn as nn
import torch

from efficientnet_pytorch import model as enet

class Model(nn.Module):
    def __init__(self, backbone, out_dim):
        super(Model, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        result = OrderedDict()
        x = self.extract(x)
        x = self.myfc(x)
        result['out'] = x
        return result

if __name__ == "__main__":
    img = torch.rand([48, 3, 128, 128])
    model = Model(backbone='efficientnet-b0', out_dim=5)
    output = model(img)
    print(output['out'].shape)