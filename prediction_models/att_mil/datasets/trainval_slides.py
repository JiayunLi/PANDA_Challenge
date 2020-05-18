import lmdb
import torch.utils.data as data
import time
import pandas as pd
import json
import numpy as np
from prediction_models.att_mil.utils import file_utils


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

        self.slide_tiles_map = json.load(open(f"{dataset_params.info_dir}/slides_tiles_mapping.json", "r"))
        self.tiles_df = pd.read_csv(f"{dataset_params.info_dir}/trainval_tiles.csv", index_col='tile_name')

    def __len__(self):
        return len(self.slides_df)

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        slide_label = int(slide_info.isup_grade)
        slide_name = slide_info.image_id
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
        return tiles, labels, slide_label, tile_names


class BiopsySlidesChunk(data.Dataset):
    def __init__(self, dataset_params, transform, fold, split, phase='train'):
        self.slides_df = pd.read_csv(f"{dataset_params.info_dir}/{split}_{fold}.csv")
        print(f"Original number of samples: {len(self.slides_df)}")
        self.slides_df.drop(self.slides_df[self.slides_df['image_id'] == '3790f55cad63053e956fb73027179707'].index,
                            inplace=True)
        self.transform = transform
        self.params = dataset_params
        self.phase = phase
        self.tiles_env = lmdb.open(f"{dataset_params.data_dir}/tiles/", max_readers=3, readonly=True,
                                   lock=False, readahead=False, meminit=False)
        self.tile_labels = json.load(open(f"{dataset_params.data_dir}/tiles_labels.json", "r"))

    def __len__(self):
        return len(self.slides_df)

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        slide_label = int(slide_info.isup_grade)
        slide_name = slide_info.image_id
        labels = self.tile_labels[slide_name]
        tiles = \
            file_utils.read_lmdb_slide_tensor(self.tiles_env,
                                              (-1, self.params.im_size, self.params.im_size, self.params.num_channels),
                                              slide_name, self.transform,
                                              out_im_size=(self.params.num_channels, self.params.input_size,
                                                           self.params.input_size), data_type=np.uint8)
        return tiles, labels, slide_label, list(range(len(tiles)))
