import pickle
import h5py
import numpy as np


class DatasetParams:
    def __init__(self, im_size, input_size, num_classes, info_dir, data_dir, cache_dir, num_channels=3):
        self.im_size = im_size
        self.input_size = input_size
        self.num_channels = num_channels
        self.num_classes = num_classes
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


def set_mil_params(mil_in_feat_size):
    params = {
        "mil_in_feat_size": mil_in_feat_size,
        "instance_embed_dim": instance_embed_dim,
        "bag_embed_dim": bag_embed_dim,
        "bag_hidden_dim": bag_hidden_dim,
        "slide_n_classes": slide_n_classes
    }
