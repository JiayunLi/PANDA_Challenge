import torch.nn as nn
from prediction_models.att_mil.utils import init_helper
import json
from collections import defaultdict


def _initialize_helper(all_modules):
    for m in all_modules:
        init_helper.weight_init(m)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


def config_vgg_layers(model, arch, input_size, num_classes):
    stride, feat_dim = 32, 512
    feat_map_size = input_size // stride
    if arch == "vgg11_bn":
        # Conv, BN, ReLU, Pool, 64
        model.layer1 = model.features[0: 4]
        # Conv, BN, ReLU, Pool, 128
        model.layer2 = model.features[4: 8]
        # Conv, BN, ReLU, Conv, BN, ReLU, Pool, 256
        model.layer3 = model.features[8: 15]
        # Conv, BN, ReLU, Conv, BN, ReLU, Pool, 512
        model.layer4 = model.features[15: 22]
        # Conv, BN, ReLU, Conv, BN, ReLU, Pool, 512
        model.layer5 = model.features[22: 29]
        # dims = [64, 128, 256, 512, 512]
    elif arch == "vgg13_bn":
        model.layer1 = model.features[0: 7]
        model.layer2 = model.features[7: 14]
        model.layer3 = model.features[14: 21]
        model.layer4 = model.features[21: 28]
        model.layer5 = model.features[28: 35]
    elif arch == "vgg16_bn":
        model.layer1 = model.features[0: 7]
        model.layer2 = model.features[7: 14]
        model.layer3 = model.features[14: 24]
        model.layer4 = model.features[24: 34]
        model.layer5 = model.features[34: 44]
    elif arch == "vgg19_bn":
        model.layer1 = model.features[0: 7]
        model.layer2 = model.features[7: 14]
        model.layer3 = model.features[14: 27]
        model.layer4 = model.features[27: 40]
        model.layer5 = model.features[40: 53]
    else:
        raise NotImplementedError(f"{arch} not implemented!")

    in_features = (feat_map_size // 2) * (feat_map_size // 2) * 512
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=(feat_map_size//2, feat_map_size//2)),
        Flatten(),
        nn.Linear(in_features=in_features, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    )
    _initialize_helper(model.classifier)
    feature_dim = 512
    return feature_dim


def config_resnet_layers(model, arch, num_classes):
    model.features = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                   model.layer1, model.layer2, model.layer3, model.layer4)
    model.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), Flatten(),
                                     nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True))
    _initialize_helper(model.classifier)
    if arch == "resnet18" or arch == "resnet34":
        return 512
    else:
        return 2048


def set_train_layers_vgg(model, train_layers):
    if train_layers >= 5:
        model.tile_encoder.layer1.train()
        for p in model.tile_encoder.layer1.parameters():
            p.requires_grad = True
    if train_layers >= 4:
        model.tile_encoder.layer2.train()
        for p in model.tile_encoder.layer2.parameters():
            p.requires_grad = True
    if train_layers >= 3:
        model.tile_encoder.layer3.train()
        for p in model.tile_encoder.layer3.parameters():
            p.requires_grad = True
    if train_layers >= 2:
        model.tile_encoder.layer4.train()
        for p in model.tile_encoder.layer4.parameters():
            p.requires_grad = True
    if train_layers >= 1:
        model.tile_encoder.layer5.train()
        for p in model.tile_encoder.layer5.parameters():
            p.requires_grad = True


def set_train_layers_resnet(model, train_layers):
    if train_layers >= 5:
        model.tile_encoder.conv1.train()
        model.tile_encoder.bn1.train()
        # model.feature_extractor.layer1.train()
        for p in model.tile_encoder.conv1.parameters():
            p.requires_grad = True
        for p in model.tile_encoder.bn1.parameters():
            p.requires_grad = True
    if train_layers >= 4:
        model.tile_encoder.layer1.train()
        for p in model.tile_encoder.layer1.parameters():
            p.requires_grad = True
    if train_layers >= 3:
        model.tile_encoder.layer2.train()
        for p in model.tile_encoder.layer2.parameters():
            p.requires_grad = True
    if train_layers >= 2:
        model.tile_encoder.layer3.train()
        for p in model.tile_encoder.layer3.parameters():
            p.requires_grad = True
    if train_layers >= 1:
        model.tile_encoder.layer4.train()
        for p in model.tile_encoder.layer4.parameters():
            p.requires_grad = True


def set_train_parameters_tile_encoder(model, arch, optim, feat_lr, cls_lr, wd, train_layer=0, fix=True):
    model.tile_encoder.classifier.train()
    for p in model.tile_encoder.classifier.parameters():
        p.requires_grad = True
    if fix:
        model.tile_encoder.features.eval()
        for p in model.tile_encoder.features.parameters():
            p.requires_grad = False
    else:
        if arch.startswith("vgg"):
            set_train_layers_vgg(model, train_layer)
        elif arch.startswith("res"):
            set_train_layers_resnet(model, train_layer)
        else:
            raise NotImplementedError(f"{arch} Not implemented!")

    # Get all parameters that require gradients
    feat_p = []
    for p in model.tile_encoder.features.parameters():
        if p.requires_grad:
            feat_p.append(p)

    classifier_p = []
    for p in model.tile_encoder.classifier.parameters():
        if p.requires_grad:
            classifier_p.append(p)

    params_group = []

    if len(feat_p) > 0:
        if optim == "sgd":
            params_group.append({'params': feat_p, 'lr': feat_lr, 'momentum': 0.9})
        else:
            params_group.append({'params': feat_p, 'lr': feat_lr, 'weight_decay': wd, 'betas': (0.9, 0.999)})

    if optim == "sgd":
        params_group.append({'params': classifier_p, 'lr': cls_lr, 'momentum': 0.9})
    else:
        params_group.append({'params': classifier_p, 'lr': cls_lr, 'weight_decay': wd, 'betas': (0.9, 0.999)})
    return params_group


def set_train_parameters_mil(model, lr, optim, wd):
    params = []
    for p in model.instance_embed.parameters():
        p.requires_grad = True
        params.append(p)
    for p in model.embed_bag_feat.parameters():
        p.requires_grad = True
        params.append(p)
    for p in model.attention.parameters():
        p.requires_grad = True
        params.append(p)
    for p in model.slide_classifier.parameters():
        p.requires_grad = True
        params.append(p)
    params_group = []
    if optim == "sgd":
        params_group.append({'params': params, 'lr': lr, 'momentum': 0.9})
    else:
        params_group.append({'params': params, 'lr': lr, 'weight_decay': wd, 'betas': (0.9, 0.999)})
    return params_group


def compute_class_frequency(trainval_df, tile_labels_map, binary_only):
    tile_label_counter = defaultdict(int)
    slide_label_counter = defaultdict(int)

    for i in range(len(trainval_df)):
        slide_info = trainval_df.iloc[i]
        slide_name = slide_info['image_id']
        slide_label = int(slide_info['isup_grade'])
        if binary_only:
            slide_label = int(slide_label > 0)
        slide_label_counter[slide_label] += 1
        tile_labels = tile_labels_map[slide_name]
        for tile_label in tile_labels:
            if binary_only:
                tile_label = int(tile_label > 0)
            tile_label_counter[tile_label] += 1

    tot_tiles = sum(tile_label_counter.values())
    tot_slides = sum(slide_label_counter.values())

    tile_label_freq = []
    slide_label_freq = []

    for i in range(max(tile_label_counter.keys())+1):
        tile_label_freq.append(tot_tiles / tile_label_counter[i])

    for i in range(max(slide_label_counter.keys())+1):
        slide_label_freq.append(tot_slides / slide_label_counter[i])

    return tile_label_freq, slide_label_freq
