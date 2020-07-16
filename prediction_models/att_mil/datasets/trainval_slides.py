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
import  os
import openslide
from PIL import Image
from prediction_models.att_mil.utils import file_utils
from prediction_models.att_mil.datasets.gen_selected_tiles import RATE_MAP
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
        print(folder)
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
        # empty_df = pd.read_csv(f"{self.params.data_dir}/empty_slides.csv")
        # print(f"Original number of samples: {len(slides_df)}")
        # for i in range(len(empty_df)):
        #     slide_id = empty_df.iloc[i]['slide_name']
        #     slides_df.drop(slides_df[slides_df['image_id'] == slide_id].index, inplace=True)
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


class BiopsySlidesBatchV2(data.Dataset):
    def __init__(self, dataset_params, transform, fold, split, has_drop_rate=0, phase='train'):
        self.transform = transform
        self.split, self.fold = split, fold
        self.params = dataset_params
        self.phase = phase

        if dataset_params.normalized:
            self.tiles_env = lmdb.open(f"{dataset_params.data_dir}/norm_tiles/", max_readers=3, readonly=True,
                                       lock=False, readahead=False, meminit=False)
            print(f"Read tiles from folder {dataset_params.data_dir}/norm_tiles/")
        else:
            self.tiles_env = lmdb.open(f"{dataset_params.data_dir}/tiles/", max_readers=3, readonly=True,
                                       lock=False, readahead=False, meminit=False)
            print(f"Read tiles from folder {dataset_params.data_dir}/tiles/")
        if not os.path.isfile(f"{dataset_params.data_dir}/tile_labels_{dataset_params.dataset}.json"):
            self.tile_labels = None
        else:
            self.tile_labels = json.load(open(f"{dataset_params.data_dir}/tile_labels_{dataset_params.dataset}.json", "r"))
        self.slides_df = self._config_data()
        self.has_drop_rate = has_drop_rate

    def _config_data(self):
        # Use all slides to compute mean std
        if self.phase == "meanstd":
            slides_df = pd.read_csv(f"{self.params.data_dir}/4_fold_train.csv")
        elif self.phase in {"train_tiles", "val_tiles"}:
            print("build dataset for training with only tiles-level losses")
            slides_df = pd.read_csv(f"{self.params.info_dir}/{self.split}_{self.fold}.csv")
            new_df = []
            for i in range(len(slides_df)):
                cur = slides_df.iloc[i].to_dict()
                if cur['data_provider'] == "radboud":
                    new_df.append(cur)
            columns = list(slides_df.columns)
            slides_df = pd.DataFrame(data=new_df, columns=columns)
        else:
            slides_df = pd.read_csv(f"{self.params.info_dir}/{self.split}_{self.fold}.csv")
        print(f"Number of {self.split} samples: {len(slides_df)}")
        return slides_df

    def __len__(self):
        return len(self.slides_df)

    def _w_instance_drop(self, tiles, labels):
        random_indices = torch.rand(len(tiles))
        mask = torch.ones(len(tiles), 1, 1, 1)
        for idx in range(len(tiles)):
            # similar as dropout
            if random_indices[idx] < self.has_drop_rate:
                mask[idx, :, :, :] = 0
                labels[idx] = 0
        # mask out regions (have the pixel value same to mean, after normalization, it should be 0)
        tiles *= mask

        return tiles, labels

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        slide_label = int(slide_info.isup_grade)
        slide_name = slide_info.image_id
        # print(slide_name)
        tiles = \
            file_utils.read_lmdb_slide_tensor(self.tiles_env,
                                              (-1, self.params.im_size, self.params.im_size, self.params.num_channels),
                                              slide_name, self.transform,
                                              out_im_size=(self.params.num_channels, self.params.input_size,
                                                           self.params.input_size), data_type=np.uint8)
        if not self.tile_labels or slide_name not in self.tile_labels:
            labels = [-1] * len(tiles)
        else:
            labels = self.tile_labels[slide_name]
        if self.has_drop_rate > 0:
            print("Use dropout instance")
            tiles, labels = self._w_instance_drop(tiles, labels)
        if self.params.top_n > 0 and len(tiles) > self.params.top_n:
            tiles = tiles[:self.params.top_n, :, :, :]
            labels = [labels[i] for i in range(self.params.top_n)]
        if len(tiles) < 32:
            tiles = torch.cat([tiles, torch.zeros(32 - len(tiles), 3, self.params.input_size, self.params.input_size),
                               ], dim=0)
            labels += [-1] * (32 - len(tiles))
        if self.params.loss_type == "bce":
            isup_grade = slide_label
            slide_label = np.zeros(5).astype(np.float32)
            slide_label[:isup_grade] = 1.
        return tiles, labels, slide_label, list(range(len(tiles)))


class BiopsySlidesSelectedOTF(data.Dataset):
    def __init__(self, dataset_params, selected_locs, transform, fold, split, has_drop_rate=0, phase='train'):
        self.transform = transform
        self.split, self.fold = split, fold
        self.params = dataset_params
        self.phase = phase
        self.selected_locs = selected_locs
        self.has_drop_rate = has_drop_rate
        self.slides_df = self._config_data()

    def _config_data(self):
        # Use all slides to compute mean std
        if self.phase == "meanstd":
            slides_df = pd.read_csv(f"{self.params.info_dir}/4_fold_train.csv")
        else:
            slides_df = pd.read_csv(f"{self.params.info_dir}/{self.split}_{self.fold}.csv")
        print(f"Number of {self.split} samples: {len(slides_df)}")
        return slides_df

    def __len__(self):
        return len(self.slides_df)

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        slide_label = int(slide_info.isup_grade)
        slide_id = slide_info.image_id
        tiles = self._get_tiles(slide_id)
        if self.has_drop_rate > 0:
            print("Use dropout instance")
            tiles = self._w_instance_drop(tiles)
        if self.params.loss_type == "bce":
            isup_grade = slide_label
            slide_label = np.zeros(5).astype(np.float32)
            slide_label[:isup_grade] = 1.
        return tiles, [-1] * len(tiles), slide_label, list(range(len(tiles)))

    def _get_high_tiles(self, low_i, low_j, rate, cur_slide, high_im_size, cur_im_shape):
        high_i = max(low_i * rate, 0)
        high_j = max(low_j * rate, 0)
        # print(f"{high_i}_{high_j}")
        if high_i + high_im_size > cur_im_shape[0]:
            high_i = cur_im_shape[0] - high_im_size
        if high_j + high_im_size > cur_im_shape[1]:
            high_j = cur_im_shape[1] - high_im_size
        high_tile = cur_slide.read_region((high_j, high_i), self.params.level + 3,
                                          (high_im_size, high_im_size)).convert("RGB")
        if high_im_size > self.params.input_size:
            high_tile = high_tile.resize((self.params.input_size, self.params.input_size), Image.ANTIALIAS)
        if self.transform:
            high_tile = self.transform(high_tile)
        return high_tile

    def _get_tiles(self, slide_id):
        rate = RATE_MAP[-3]
        # high_im_size = self.params.lowest_im_size * rate
        # TODO:high_im_size = self.selected_locs[slide_id]['lowest_tile_size_high'] * rate
        high_im_size = 64 * rate
        cur_slide = openslide.OpenSlide(f"{self.params.data_dir}/{slide_id}.tiff")
        cur_im_shape = (cur_slide.level_dimensions[self.params.level + 3][1],
                        cur_slide.level_dimensions[self.params.level + 3][0])
        lowest_locs = self.selected_locs[slide_id]['high_res']
        # print(len(lowest_locs))

        if self.phase == "meanstd":
            n = len(lowest_locs)
        else:
            n = self.params.top_n
        instances = torch.FloatTensor(n, self.params.num_channels,
                                      self.params.input_size, self.params.input_size,)
        counter = 0
        for low_ij in lowest_locs:
            if len(low_ij) == 1:
                low_i, low_j = low_ij[0][0], low_ij[0][1]
                high_tile = self._get_high_tiles(low_i, low_j, rate, cur_slide, high_im_size, cur_im_shape)
                instances[counter, :, :, :] = high_tile
                counter += 1
            else:
                for cur_lowij in low_ij:
                    low_i, low_j = cur_lowij[0][0], cur_lowij[0][1]
                    high_tile = self._get_high_tiles(low_i, low_j, rate, cur_slide, high_im_size, cur_im_shape)
                    instances[counter, :, :, :] = high_tile
                    counter += 1
            if counter >= len(instances):
                break
        return instances

    def _w_instance_drop(self, tiles):
        random_indices = torch.rand(len(tiles))
        mask = torch.ones(len(tiles), 1, 1, 1)
        for idx in range(len(tiles)):
            # similar as dropout
            if random_indices[idx] < self.has_drop_rate:
                mask[idx, :, :, :] = 0
        # mask out regions (have the pixel value same to mean, after normalization, it should be 0)
        tiles *= mask
        return tiles


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


STAGE_ONE_N = 12


class BiopsySlidesImage(data.Dataset):
    def __init__(self, dataset_params, transform, fold, split, phase='train'):
        self.transform = transform
        self.split, self.fold = split, fold
        self.params = dataset_params
        self.phase = phase
        self.slides_df, self.slides_tile_mapping = self._config_data()

    def _config_data(self):
        # Use all slides to compute mean std
        if self.phase == "meanstd":
            slides_df = pd.read_csv(f"{self.params.data_dir}/train.csv")
        else:
            slides_df = pd.read_csv(f"{self.params.info_dir}/{self.split}_{self.fold}.csv")
        slides_tile_mapping = defaultdict(list)
        slides_loc = glob.glob(f"{self.params.data_dir}/train/*.png")
        for tile_loc in slides_loc:
            slide_id = tile_loc.split("/")[-1].split("_")[0]
            slides_tile_mapping[slide_id].append(tile_loc)

        # tiles_dir = f"{self.params.data_dir}/train/"
        to_drop = []
        for i in range(len(slides_df)):
            slide_id = str(slides_df.iloc[i].image_id)
            # tiles_loc = glob.glob(f"{tiles_dir}/{slide_id}*.png")
            # if len(tiles_loc) == 0:
            if slide_id not in slides_tile_mapping:
                to_drop.append(slide_id)
            # else:
            #     slides_tile_mapping[slide_id] = tiles_loc
        for slide_id in to_drop:
            slides_df.drop(slides_df[slides_df['image_id'] == slide_id].index, inplace=True)
        print(f"Number of samples: {len(slides_df)}")
        print(len(slides_tile_mapping))
        return slides_df, slides_tile_mapping

    def __len__(self):
        return len(self.slides_df)

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        slide_label = int(slide_info.isup_grade)
        slide_name = slide_info.image_id
        # cur_tiles_loc = self.slides_tile_mapping[str(slide_name)]
        tiles = torch.FloatTensor(STAGE_ONE_N, self.params.num_channels, self.params.input_size,
                                  self.params.input_size)
        # for i, tile_loc in enumerate(cur_tiles_loc):
        #     tile = Image.open(tile_loc)
        #     if self.transform:
        #         tile = self.transform(tile)
        #     tiles[i, :, :, :] = tile
        for idx in range(STAGE_ONE_N):
            tile_loc = f"{self.params.data_dir}/train/{slide_name}_{idx}.png"
            tile = Image.open(tile_loc)
            if self.transform:
                tile = self.transform(tile)
            tiles[idx, :, :, :] = tile

        return tiles, [-1] * len(tiles), slide_label, list(range(len(tiles)))
