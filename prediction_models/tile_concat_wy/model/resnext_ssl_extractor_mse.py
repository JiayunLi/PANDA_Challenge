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
from collections import OrderedDict
from fastai.vision import *
## custom package
from utiles.mishactivation import Mish
from utiles.hubconf import *


class Model_Infer(nn.Module):
    def __init__(self, arch='resnext50_32x4d'):
        super().__init__()
        # m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        m = self._resnext(semi_supervised_model_urls[arch], Bottleneck, [3, 4, 6, 3], False,
                     progress=False, groups=32, width_per_group=4)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(), Flatten(), nn.Linear(2 * nc, 512),
                                  Mish())

    def _resnext(self, url, block, layers, pretrained, progress, **kwargs):
        model = ResNet(block, layers, **kwargs)
        #state_dict = load_state_dict_from_url(url, progress=progress)
        #model.load_state_dict(state_dict)
        return model
    def forward(self, x):
        """
        x: [bs, N, 3, h, w]
        x_out: [bs, N]
        """
        result = OrderedDict()
        # bs, c, h, w = x.shape
        x = self.enc(x)  # x: bs*N x C x 4 x 4
        x = self.head(x)
        result['out'] = x
        return result

if __name__ == "__main__":
    img = torch.rand([2, 3, 1 * 256, 1 * 256]).cuda()
    model = Model_Infer().cuda()
    output = model(img)
    label = torch.tensor([[1,1,1,0,0],
                         [1,1,0,0,0]]).cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss1 = criterion(output['out'], label.float())
    loss2 = criterion(output['out'], label.float())
    losses = torch.Tensor([loss1, loss2]).cuda()
    mltLoss = MultiTaskLoss(2).cuda()
    loss = mltLoss(losses)
    print(output['out'].shape, loss)