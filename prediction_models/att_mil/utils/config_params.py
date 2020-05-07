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
    def __init__(self, im_size, input_size, num_classes, overlap, ts_thres):
        self.im_size = im_size
        self.input_size = input_size
        self.dw_rate = im_size / input_size
        self.num_classes = num_classes
        self.overlap = overlap
        self.ts_thres = ts_thres