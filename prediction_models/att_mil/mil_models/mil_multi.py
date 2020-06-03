import torchvision.models as models
from prediction_models.att_mil.utils import model_utils, init_helper
import torch.nn as nn
import torch
from fastai.vision import *
from prediction_models.tile_concat_wy.utiles import mishactivation
from prediction_models.att_mil.mil_models.mil import config_encoder
import torch.nn.functional as F


class AttMILBatchMulti(nn.Module):
    def __init__(self, base_encoder, pretrained, arch, input_size, feature_dim, mil_params):
        super(AttMILBatchMulti, self).__init__()
        self.tile_encoder = base_encoder
        # self.tile_encoder = base_encoder.features
        # self.tile_classifier = base_encoder.classifier
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self.hp = {"input_size": input_size,  "encoder_arch": arch,
                   "feature_dim": feature_dim, "pretrained": pretrained,
                   "mil_params": mil_params, "arch": arch}
        self.mil_params = mil_params

        self.instance_embed_1 = self._config_instance_embed()
        self.instance_embed_2 = self._config_instance_embed()

        self.embed_bag_feat, self.attention = self._config_attention()
        self.slide_classifier = self._config_classifier()
        self.softmax = nn.Softmax(dim=1)
        self._initialize()

    def _initialize(self):
        for m in self.instance_embed_1.modules():
            init_helper.weight_init(m)
        for m in self.instance_embed_2.modules():
            init_helper.weight_init(m)
        for m in self.embed_bag_feat.modules():
            init_helper.weight_init(m)
        for m in self.slide_classifier.modules():
            init_helper.weight_init(m)

    def _config_instance_embed(self):

        instance_embed = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim // 2,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.feature_dim // 2),
            AdaptiveConcatPool2d(), Flatten())
        return instance_embed

    def _config_attention(self):
        embed_bag_feat = nn.Sequential(
            nn.Linear(self.feature_dim,
                      self.mil_params['bag_embed_dim']),
            mishactivation.Mish(), nn.Dropout(0.5)
        )

        attention = nn.Sequential(
            nn.Linear(self.mil_params['bag_embed_dim'], self.mil_params['bag_hidden_dim']),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(self.mil_params["bag_hidden_dim"], 1)
        )
        return embed_bag_feat, attention

    def _config_classifier(self):
        classifier = nn.Sequential(
            nn.Linear(self.mil_params["bag_embed_dim"],
                      self.mil_params["n_slide_classes"]),
        )
        return classifier

    def forward(self, tiles_low, tiles_high, phase="regular"):
        batch_size, n_tiles_low, channel, h_low, w_low = tiles_low.shape
        _, n_tiles_high, _, h_high, w_high = tiles_high.shape
        # Share weights for the feature extractor part
        # https://discuss.pytorch.org/t/how-to-create-model-with-sharing-weight/398
        feats_low = self.tile_encoder.features(tiles_low.view(-1, channel, h_low, w_low).contiguous())
        feats_high = self.tile_encoder.features(tiles_high.view(-1, channel, h_high, w_high).contiguous())
        feats_low = self.instance_embed_1(feats_low).view(batch_size, n_tiles_low, -1)
        feats_high = self.instance_embed_1(feats_high).view(batch_size, n_tiles_high, -1)
        n_tiles = n_tiles_low + n_tiles_high
        # bs, (n_tiles_low + n_tiles_high), feature_dim
        feats = torch.cat((feats_low, feats_high), dim=1).view(-1, feats_high.size(2))
        feats = self.embed_bag_feat(feats)

        if phase == 'extract_feats':
            return feats.view(batch_size, n_tiles, -1)

        raw_atts = self.attention(feats)
        atts = self.softmax(raw_atts.view(batch_size, n_tiles, -1))
        weighted_feats = torch.matmul(atts.permute(0, 2, 1), feats.view(batch_size, n_tiles, -1))
        weighted_feats = torch.squeeze(weighted_feats, dim=1)
        probs = (self.slide_classifier(weighted_feats))

        return probs, None, atts