import lmdb
import torch.utils.data as data
import time
import pandas as pd
import json
import random
import torch
import numpy as np
from prediction_models.att_mil.utils import file_utils

MAX_N_TILES = 500


class BiopsySlides(data.Dataset):
    def __init__(self, dataset_params, transform, fold, split, phase='train'):
        self.slides_df = pd.read_csv(f"{dataset_params.info_dir}/{split}_{fold}.csv")
        print(f"Original number of samples: {len(self.slides_df)}")
        self.slides_df.drop(self.slides_df[self.slides_df['image_id'] == '3790f55cad63053e956fb73027179707'].index,
                            inplace=True)
        print(f"Number of samples after dropping: {len(self.slides_df)}")
        self.transform = transform
        self.params = dataset_params
        self.phase = phase
        self.tiles_env = lmdb.open(f"{dataset_params.data_dir}/tiles/", max_readers=3, readonly=True,
                                   lock=False, readahead=False, meminit=False)

        self.slide_tiles_map = json.load(open(f"{dataset_params.data_dir}/slides_tiles_mapping.json", "r"))
        self.tiles_df = pd.read_csv(f"{dataset_params.info_dir}/trainval_tiles.csv", index_col='tile_name')

    def __len__(self):
        return len(self.slides_df)

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        slide_label = int(slide_info.isup_grade)
        slide_name = str(slide_info.image_id)
        tile_names = self.slide_tiles_map[slide_name]
        start_time = time.time()
        # If tile-level is not usable, labels will be -1 (e.g., Karo slides with different PG and SG)
        tiles, labels = file_utils.read_lmdb_tiles_tensor(f"{self.params.data_dir}",
                                                          (self.params.im_size, self.params.im_size,
                                                           self.params.num_channels),
                                                          tile_names, self.transform,
                                                          out_im_size=(self.params.num_channels, self.params.input_size,
                                                                       self.params.input_size),
                                                          tiles_df=self.tiles_df, env=self.tiles_env,
                                                          data_type=np.uint8)
        print(f"Total time to load one slide {time.time() - start_time}")
        if len(tiles) > MAX_N_TILES:
            sample_ids = random.sample(range(0, len(tiles)), MAX_N_TILES)
            tiles = tiles[sample_ids, :, :, :]
            labels = [labels[idx] for idx in sample_ids]
        print(len(tiles))
        return tiles, labels, slide_label, tile_names


class BiopsySlidesChunk(data.Dataset):
    def __init__(self, dataset_params, transform, fold, split, phase='train'):
        self.transform = transform
        self.split, self.fold = split, fold
        self.params = dataset_params
        self.phase = phase
        if dataset_params.normalized:
            folder = "tiles"
        else:
            folder = "orig_tiles"
        self.tiles_env = lmdb.open(f"{dataset_params.data_dir}/{folder}/", max_readers=3, readonly=True,
                                   lock=False, readahead=False, meminit=False)
        self.tile_labels = json.load(open(f"{dataset_params.data_dir}/tile_labels_{dataset_params.dataset}.json", "r"))
        self.slides_df = self._config_data()

    def _config_data(self):
        # Use all slides to compute mean std
        if self.phase == "meanstd":
            slides_df = pd.read_csv(f"{self.params.data_dir}/train.csv")
        else:
            slides_df = pd.read_csv(f"{self.params.info_dir}/{self.split}_{self.fold}.csv")
        empty_df = pd.read_csv(f"{self.params.data_dir}/empty_slides.csv")
        print(f"Original number of samples: {len(slides_df)}")
        for i in range(len(empty_df)):
            slide_id = empty_df.iloc[i]['slide_name']
            slides_df.drop(slides_df[slides_df['image_id'] == slide_id].index, inplace=True)
        print(f"Number of samples after dropping out: {len(slides_df)}")
        return slides_df

    def __len__(self):
        return len(self.slides_df)

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        slide_label = int(slide_info.isup_grade)
        slide_name = slide_info.image_id

        tiles = \
            file_utils.read_lmdb_slide_tensor(self.tiles_env,
                                              (-1, self.params.im_size, self.params.im_size, self.params.num_channels),
                                              slide_name, self.transform,
                                              out_im_size=(self.params.num_channels, self.params.input_size,
                                                           self.params.input_size), data_type=np.uint8)
        labels = self.tile_labels[slide_name] if slide_name in self.tile_labels else [-1] * len(tiles)
        if len(tiles) > MAX_N_TILES:
            sample_ids = random.sample(range(0, len(tiles)), MAX_N_TILES)
            tiles = tiles[sample_ids, :, :, :]
            labels = [labels[idx] for idx in sample_ids]
        return tiles, labels, slide_label, list(range(len(tiles)))


FIX_N_TILES=16


class BiopsySlidesBatch(BiopsySlidesChunk):
    def __init__(self, dataset_params, transform, fold, split, phase='train'):
        super().__init__(dataset_params, transform, fold, split, phase)

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        slide_label = int(slide_info.isup_grade)
        slide_name = slide_info.image_id

        tiles = \
            file_utils.read_lmdb_slide_tensor(self.tiles_env,
                                              (-1, self.params.im_size, self.params.im_size, self.params.num_channels),
                                              slide_name, self.transform,
                                              out_im_size=(self.params.num_channels, self.params.input_size,
                                                           self.params.input_size), data_type=np.uint8)
        # labels = self.tile_labels[slide_name] if slide_name in self.tile_labels else [-1] * FIX_N_TILES
        if len(tiles) < FIX_N_TILES:
            pad_len = FIX_N_TILES - len(tiles)
            tiles = torch.cat([tiles, torch.zeros(pad_len, 3, self.params.input_size, self.params.input_size),
                               ], dim=0)
            # labels += [0] * pad_len
        elif len(tiles) > FIX_N_TILES:
            tiles = tiles[:FIX_N_TILES, :, :, :]
            # labels = labels[:FIX_N_TILES]
        return tiles, slide_label, slide_label, list(range(len(tiles)))