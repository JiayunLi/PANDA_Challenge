import lmdb
import torch.utils.data as data
import time
import pandas as pd
import json
import random
import torch
import numpy as np
from collections import defaultdict
import glob
from PIL import Image
from prediction_models.att_mil.utils import file_utils


class BiopsySlidesBatchMulti(data.Dataset):
    def __init__(self, dataset_params, transform, fold, split, has_drop_rate=0, phase='train'):
        self.transform = transform
        self.split, self.fold = split, fold
        self.params = dataset_params
        self.phase = phase
        print(f"Read tiles from folder {dataset_params.data_dir_low}/tiles/")
        print(f"Read tiles from folder {dataset_params.data_dir_high}/tiles/")
        self.low_env = lmdb.open(f"{dataset_params.data_dir_low}/tiles/", max_readers=3, readonly=True,
                                   lock=False, readahead=False, meminit=False)
        self.high_env = lmdb.open(f"{dataset_params.data_dir_high}/tiles/", max_readers=3, readonly=True,
                                 lock=False, readahead=False, meminit=False)
        self.slides_df = self._config_data()
        self.has_drop_rate = has_drop_rate

    def _config_data(self):
        # Use all slides to compute mean std
        if self.phase == "meanstd":
            slides_df = pd.read_csv(f"{self.params.data_dir}/4_fold_train.csv")
        else:
            slides_df = pd.read_csv(f"{self.params.info_dir}/{self.split}_{self.fold}.csv")
        print(f"Number of {self.split} samples: {len(slides_df)}")
        return slides_df

    def __len__(self):
        return len(self.slides_df)

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        slide_label = int(slide_info.isup_grade)
        slide_name = slide_info.image_id

        tiles_low = \
            file_utils.read_lmdb_slide_tensor(self.low_env,
                                              (-1, self.params.im_size_low, self.params.im_size_low,
                                               self.params.num_channels), slide_name, self.transform,
                                              out_im_size=(self.params.num_channels, self.params.im_size_low,
                                                           self.params.im_size_low), data_type=np.uint8)
        tiles_high = \
            file_utils.read_lmdb_slide_tensor(self.high_env,
                                              (-1, self.params.im_size_high, self.params.im_size_high,
                                               self.params.num_channels), slide_name, self.transform,
                                              out_im_size=(self.params.num_channels, self.params.im_size_low,
                                                           self.params.im_size_low), data_type=np.uint8)
        if self.params.top_n > 0 and len(tiles_low) > self.params.top_n:
            tiles_low = tiles_low[:self.params.top_n, :, :, :]

        if self.params.top_n > 0 and len(tiles_high) > self.params.top_n:
            tiles_high = tiles_high[:self.params.top_n, :, :, :]
        return tiles_low, tiles_high, slide_label