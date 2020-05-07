import torchvision.models as models
from prediction_models.att_mil.utils import model_utils, init_helper
import torch.nn as nn


def config_encoder(input_size, num_classes, arch, pretrained):
    encoder_name = models.__dict__[arch]
    encoder = encoder_name(pretrained=pretrained)
    # Convert encoders to have a features and a classifier
    # The classifier should be reinitialized to accommodate different number of classes
    if arch.startswith("vgg"):
        feature_dim = model_utils.config_vgg_layers(encoder, arch, input_size, num_classes)

    elif arch.startswith("res"):
        feature_dim = model_utils.config_resnet_layers(encoder, arch, num_classes)
    else:
        raise NotImplementedError(f"{arch} Haven't implemented")
    return feature_dim


class AttMIL(nn.Module):
    def __init__(self, base_encoder, pretrained, arch, input_size, n_tile_classes, feature_dim, mil_params):
        super(AttMIL, self).__init__()
        self.tile_encoder = base_encoder
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self.hp = {"input_size": input_size, "n_tile_classes": n_tile_classes, "encoder_arch": arch,
                   "feature_dim": feature_dim, "pretrained": pretrained, "mil_params": mil_params}
        self.mil_params = mil_params

        self.instance_embed = self._config_instance_embed()

        self.embed_bag_feat, self.attention = self._config_attention()
        self.slide_classifier = self._config_classifer()
        self._initialize()

    def _config_instance_embed(self):
        nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(self.mil_params['mil_in_feat_size'],
                                              self.mil_params['mil_in_feat_size'])),
            nn.Conv2d(self.feature_dim, self.mil_params["instance_embed_dim"],
                      kernel_size=1, stride=1, padding=0))
        return instance_embed

    def _config_attention(self):
        embed_bag_feat = nn.Sequential(
            nn.Linear(self.mil_params["instance_embed_dim"] *
                      self.mil_params['mil_in_feat_size'] * self.mil_params['mil_in_feat_size'],
                      self.mil_params['bag_embed_dim']),
            nn.ReLU(),
            nn.Dropout(),
        )

        attention = nn.Sequential(
            nn.Linear(self.mil_params['bag_embed_dim'], self.mil_params['bag_embed_dim']),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(self.mil_params["bag_hidden_dim"], self.mil_params["slide_n_classes"])
        )
        return embed_bag_feat, attention

    def _config_classifer(self):
        classifier = nn.Sequential(
            nn.Linear(self.mil_params["bag_hidden_dim"] * self.mil_params["slide_n_classes"],
                      self.mil_params["slide_n_classes"]),
        )
        return classifier

    def _initialize(self):
        if not self.pretrained:
            for m in self.tile_encoder.modules():
                init_helper.weight_init(m)
        for m in self.instance_embed.modules():
            init_helper.weight_init(m)
        for m in self.embed_bag_feat.modules():
            init_helper.weight_init(m)
        for m in self.slide_classifier.modules():
            init_helper.weight_init(m)

    def forward(self, tiles, phase="regular"):
        feats = self.tile_encoder.features(tiles.contiguous())
        tiles_probs = self.tile_encoder.classifier(feats)

        feats = self.instance_embed(feats)
        feats = self.embed_bag_feat(feats)

        if phase == 'extract_feats':
            return feats

        raw_atts = self.attention(feats)
        atts = torch.transpose(raw_atts, 1, 0)
        atts = F.softmax(atts, dim=1)

        # size: 1 * bag_embed_size
        weighted_feats = torch.mm(atts, feats)
        probs = (self.slide_classifier(weighted_feats.view(-1).contiguous())).unsqueeze(dim=0)

        return probs, tiles_probs, atts



