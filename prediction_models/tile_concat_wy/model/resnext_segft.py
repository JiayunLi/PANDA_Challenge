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
from fastai.vision import *
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
## custom package
from utiles.mishactivation import Mish
from utiles.hubconf import *


class Model(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True, load=None):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(), Flatten(), nn.Linear(2 * nc, 512),
                                  Mish(), nn.BatchNorm1d(512), nn.Dropout(0.5), nn.Linear(512, 1))
        # self.classifier = DeepLabHead(2048, n)

        if load:
            self.load_pretrained(load)
        # self.aux_head = nn.Sequential(AdaptiveConcatPool2d(), Flatten(), nn.Linear(2 * 1024, 512),
        #                           Mish(), nn.BatchNorm1d(512), nn.Dropout(0.5), nn.Linear(512, 1))

    def load_pretrained(self, load):
        pretrained_dict = torch.load(load)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict)

    def forward(self, x):
        """
        x: [bs, N, 3, h, w]
        x_out: [bs, N]
        """
        result = OrderedDict()
        bs, n, c, h, w = x.shape
        input_shape = (h, w)
        x = x.view(-1, c, h, w)  # x: bs*N x 3 x 128 x 128
        x = self.enc(x)  # x: bs*N x C x 4 x 4
        _, c, h, w = x.shape
        # ## segmentation head
        # y = self.classifier(x)
        # y = F.interpolate(y, size=input_shape, mode='bilinear', align_corners=False)
        # result["out"] = y
        ## concatenate the output for tiles into a single map
        x = x.view(bs, n, c, h, w).permute(0, 2, 1, 3, 4).contiguous() \
            .view(-1, c, h * n, w)  # x: bs x C x N*4 x 4
        x = self.head(x)  # x: bs x n
        result["isup_grade"] = x
        return result

class Model_Infer(nn.Module):
    def __init__(self, arch='resnext50_32x4d', n=6, pre=True):
        super().__init__()
        # m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        m = self._resnext(semi_supervised_model_urls[arch], Bottleneck, [3, 4, 6, 3], False,
                     progress=False, groups=32, width_per_group=4)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(), Flatten(), nn.Linear(2 * nc, 512),
                                  Mish(), nn.BatchNorm1d(512), nn.Dropout(0.5), nn.Linear(512, n))

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
        bs, n, c, h, w = x.shape
        x = x.view(-1, c, h, w)  # x: bs*N x 3 x 128 x 128
        x = self.enc(x)  # x: bs*N x C x 4 x 4
        _, c, h, w = x.shape

        ## concatenate the output for tiles into a single map
        x = x.view(bs, n, c, h, w).permute(0, 2, 1, 3, 4).contiguous() \
            .view(-1, c, h * n, w)  # x: bs x C x N*4 x 4
        x = self.head(x)  # x: bs x n
        return x

if __name__ == "__main__":
    img = torch.rand([4, 12, 3, 256, 256])
    model = Model(n = 1)
    output = model(img)
    print(output['isup_grade'].shape)