import pickle
import h5py
import numpy as np


class DatasetParams:
    def __init__(self, im_size, input_size, info_dir, data_dir, cache_dir, num_channels=3):
        self.im_size = im_size
        self.input_size = input_size
        self.num_channels = num_channels
        self.info_dir = info_dir
        self.data_dir = data_dir
        self.cache_dir = cache_dir


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
    def __init__(self, lr, feat_lr, train_blocks, optim, tot_epochs, feat_ft, log_every, alpha):
        self.lr = lr
        self.feat_lr = feat_lr
        self.train_blocks = train_blocks
        self.optim = optim
        self.tot_epochs = tot_epochs
        self.feat_ft = feat_ft
        self.log_every = log_every
        self.alpha = alpha


def set_mil_params(mil_in_feat_size, instance_embed_dim, bag_embed_dim, bag_hidden_dim, slide_n_classes):
    params = {
        "mil_in_feat_size": mil_in_feat_size,
        "instance_embed_dim": instance_embed_dim,
        "bag_embed_dim": bag_embed_dim,
        "bag_hidden_dim": bag_hidden_dim,
        "slide_n_classes": slide_n_classes
    }
    return params
