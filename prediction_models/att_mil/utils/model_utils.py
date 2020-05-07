import torch.nn as nn
from prediction_models.att_mil.utils import init_helper


def _initialize_helper(all_modules):
    for m in all_modules:
        init_helper.weight_init(m)


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
        nn.Linear(in_features=in_features, out_features=4096, bias=True),
        nn.ReLU(inplace),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    )
    _initialize_helper(model.classifier)
    feature_dim = 512
    return feature_dim


def config_resnet_layers(model, arch, num_classes):
    model.features = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                   model.layer1, model.layer2, model.layer3, model.layer4)
    model.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                     nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True))
    _initialize_helper(model.classifier)
    if arch == "resnet18" or arch == "resnet34":
        return 512
    else:
        return 2048
