import lmdb
import torch.utils.data as data
import time
import pandas as pd
import json
import random
import torch
import numpy as np
import openslide
from PIL import Image
from prediction_models.att_mil.datasets.gen_selected_tiles import RATE_MAP
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
        if self.params.top_n_low > 0 and len(tiles_low) > self.params.top_n_low:
            tiles_low = tiles_low[:self.params.top_n_low, :, :, :]

        if self.params.top_n_high > 0 and len(tiles_high) > self.params.top_n_high:
            tiles_high = tiles_high[:self.params.top_n_high, :, :, :]
        return tiles_low, tiles_high, slide_label


class BiopsySlidesBatchMultiSelect(data.Dataset):
    def __init__(self, dataset_params, selected_locs, transform, fold, split, has_drop_rate=0, phase='train'):
        self.transform = transform
        self.split, self.fold = split, fold
        self.params = dataset_params
        self.phase = phase
        self.slides_df = self._config_data()
        self.has_drop_rate = has_drop_rate
        self.selected_locs = selected_locs
        self.low_level = self.selected_locs['low_level']
        self.high_level = self.selected_locs['high_level']
        self.lowest_tile_size_low = self.selected_locs['lowest_tile_size_low']
        self.lowest_tile_size_high = self.selected_locs['lowest_tile_size_high']

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

    def _get_tiles(self, cur_slide, level, lowest_locs, lowest_im_size):
        rate = RATE_MAP[level]
        high_im_size = lowest_im_size * rate
        cur_im_shape = (cur_slide.level_dimensions[level + 3][1],
                        cur_slide.level_dimensions[level + 3][0])

        if self.phase == "meanstd":
            n = len(lowest_locs)
        else:
            n = self.params.top_n
        instances = torch.FloatTensor(n, self.params.num_channels,
                                      self.params.input_size, self.params.input_size, )
        counter = 0
        for low_i, low_j in lowest_locs:
            high_i = max(low_i * rate, 0)
            high_j = max(low_j * rate, 0)
            # print(f"{high_i}_{high_j}")
            if high_i + high_im_size > cur_im_shape[0]:
                high_i = cur_im_shape[0] - high_im_size
            if high_j + high_im_size > cur_im_shape[1]:
                high_j = cur_im_shape[1] - high_im_size
            high_tile = cur_slide.read_region((high_j, high_i), level + 3,
                                              (high_im_size, high_im_size)).convert("RGB")
            if high_im_size > self.params.input_size:
                high_tile = high_tile.resize((self.params.input_size, self.params.input_size), Image.ANTIALIAS)
            if self.transform:
                high_tile = self.transform(high_tile)
            instances[counter, :, :, :] = high_tile
            counter += 1
            if counter >= len(instances):
                break
        return instances

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        slide_label = int(slide_info.isup_grade)
        slide_id = slide_info.image_id
        cur_slide = openslide.OpenSlide(f"{self.params.data_dir}/{slide_id}.tiff")
        lowest_locs_low = self.selected_locs[slide_id]["low_res"]  # For low resolution
        tiles_low = self._get_tiles(cur_slide, self.low_level, lowest_locs_low, self.lowest_tile_size_low)
        lowest_locs_high = self.selected_locs[slide_id]["high_res"]  # For high resolution
        tiles_high = self._get_tiles(cur_slide, self.high_level, lowest_locs_high, self.lowest_tile_size_high)
        return tiles_low, tiles_high, slide_label


