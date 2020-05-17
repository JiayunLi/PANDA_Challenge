import pickle
import h5py
import numpy as np


class DatasetParams:
    def __init__(self, im_size, input_size, info_dir, data_dir, cache_dir, exp_dir, num_channels=3):
        self.im_size = im_size
        self.input_size = input_size
        self.num_channels = num_channels
        self.info_dir = info_dir
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.exp_dir = exp_dir


class DatasetTest:
    def __init__(self, im_size, input_size, num_classes, overlap, ts_thres, num_channels=3):
        self.im_size = im_size
        self.input_size = input_size
        self.dw_rate = im_size / input_size
        self.num_classes = num_classes
        self.overlap = overlap
        self.ts_thres = ts_thres
        self.test_file = self.test_file
        self.test_slides_dir = self.test_slides_dir
        self.num_channels = num_channels


class TrainvalParams:
    def __init__(self, lr, feat_lr, wd, train_blocks, optim, tot_epochs, feat_ft, log_every, alpha, loss_type,
                 cls_weighted):
        self.lr = lr
        self.feat_lr = feat_lr
        self.wd = wd
        self.train_blocks = train_blocks
        self.optim = optim
        self.tot_epochs = tot_epochs
        self.feat_ft = feat_ft
        self.log_every = log_every
        self.alpha = alpha
        self.loss_type = loss_type
        self.cls_weighted = cls_weighted


def set_mil_params(mil_in_feat_size, instance_embed_dim, bag_embed_dim, bag_hidden_dim, slide_n_classes,
                   tile_classes, loss_type):
    if loss_type == "mse":
        slide_n_classes = 1
        tile_classes = 1
    params = {
        "mil_in_feat_size": mil_in_feat_size,
        "instance_embed_dim": instance_embed_dim,
        "bag_embed_dim": bag_embed_dim,
        "bag_hidden_dim": bag_hidden_dim,
        "n_slide_classes": slide_n_classes,
        "n_tile_classes": tile_classes,
    }
    return params
