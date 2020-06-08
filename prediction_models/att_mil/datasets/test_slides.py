"""
DataLoader to extract tiles from a slide
"""
from PIL import Image
from preprocessing.tile_generation import generate_grid_br as generate_grid
import torch
import torch.utils.data as data
import numpy as np
import skimage.io
from prediction_models.att_mil.datasets import gen_selected_tiles


class BiopsySlides(data.Dataset):
    def __init__(self, params, test_df, transform, tile_normalizer):
        self.params = params
        self.test_df = test_df
        self.test_slides_dir = params.test_slides_dir
        self.transform = transform
        self.tile_normalizer = tile_normalizer

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, ix):
        slide_info = self.test_df.iloc[ix]
        tile_generator = generate_grid.TileGeneratorGridBr(self.test_slides_dir, f"{slide_info.image_id}.tiff",
                                                           None, verbose=False)
        norm_tiles, _, _ = tile_generator.extract_top_tiles(self.params.im_size, self.params.overlap,
                                                         self.params.ts_thres, self.params.dw_rate,
                                                         self.params.top_n, normalizer=self.tile_normalizer)
        instances = torch.FloatTensor(len(norm_tiles),
                                      self.params.num_channels, self.params.input_size, self.params.input_size)
        for i, norm_tile in enumerate(norm_tiles):
            if self.transform:
                instances[i, :, :, :] = self.transform(norm_tile)
        del norm_tiles
        return instances, slide_info.image_id


class BiopsySlidesBatch(BiopsySlides):
    def __init__(self, params, test_df, transform, tile_normalizer):
        super().__init__(params, test_df, transform, tile_normalizer)

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, ix):
        slide_info = self.test_df.iloc[ix]
        tile_generator = generate_grid.TileGeneratorGridBr(self.test_slides_dir, f"{slide_info.image_id}.tiff",
                                                           None, verbose=False)
        norm_tiles, _, _ = tile_generator.extract_top_tiles(self.params.im_size, self.params.overlap,
                                                         self.params.ts_thres, self.params.dw_rate,
                                                         self.params.top_n, normalizer=self.tile_normalizer)
        instances = torch.FloatTensor(len(norm_tiles),
                                      self.params.num_channels, self.params.input_size, self.params.input_size)
        for i, norm_tile in enumerate(norm_tiles):
            if self.transform:
                instances[i, :, :, :] = self.transform(norm_tile)
        del norm_tiles
        if len(instances) < self.params.top_n:
            pad_len = self.params.top_n - len(instances)
            instances = \
                torch.cat([instances, torch.zeros(pad_len, 3, self.params.input_size, self.params.input_size)], dim=0)
        return instances, slide_info.image_id


class BiopsySlidesLowest(data.Dataset):
    def __init__(self, params, test_df, transform, phase='test'):
        self.params = params
        print(params)
        self.test_df = test_df
        self.test_slides_dir = params.test_slides_dir
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, ix):
        slide_info = self.test_df.iloc[ix]
        img = skimage.io.MultiImage(f"{self.test_slides_dir}/{slide_info.image_id}.tiff")[-1]
        img, pad_top, pad_left = gen_selected_tiles.get_padded(img, self.params.input_size)
        n_row, n_col = img.shape[0] // self.params.input_size, img.shape[1] // self.params.input_size

        tiles, tile_idxs = gen_selected_tiles.tile_padded(img, self.params.input_size, self.params.top_n)
        instances = torch.FloatTensor(len(tiles),
                                      self.params.num_channels, self.params.input_size, self.params.input_size)
        tile_locs = torch.FloatTensor(len(tiles), 2)
        for i, tile in enumerate(tiles):
            if self.transform:
                instances[i, :, :, :] = self.transform(tile)
            tile_id = tile_idxs[i]
            x, y = tile_id // n_col, tile_id % n_col
            tile_locs[i, 0], tile_locs[i, 1] = float(max(x * self.params.input_size - pad_top, 0)), \
                                               float(max(y * self.params.input_size - pad_left, 0))
        # if self.phase == "test":
        #     return instances, slide_info.image_id
        # else:
        return instances, slide_info.image_id, tile_locs


class BiopsySlideSelected(data.Dataset):
    # slides_df, slides_dir,
    # lowest_im_size, input_size, num_channels, level, top_n, transform

    def __init__(self, params, test_df, transform, mode=0, phase='test'):
        self.params = params
        self.lowest_im_size = params.lowest_im_size
        self.slides_dir = params.test_slides_dir
        self.level = params.level
        rate_map = {-1: 1, -2: 4, -3: 16}
        self.rate = rate_map[self.level]
        self.top_n = params.top_n
        self.slides_df = test_df
        self.input_size, self.num_channels = params.input_size, params.num_channels
        self.transform = transform
        self.phase = phase
        self.mode = mode

    def __len__(self):
        return len(self.slides_df)

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        orig = skimage.io.MultiImage(f"{self.slides_dir}/{slide_info.image_id}.tiff")
        pad_img, tile_idxs, pad_top, pad_left = gen_selected_tiles.select_at_lowest(orig[-1], self.lowest_im_size,
                                                                                    self.top_n, True, self.mode)

        results = gen_selected_tiles.get_highres_tiles(orig, tile_idxs, pad_top, pad_left, self.lowest_im_size,
                                    (pad_img.shape[0], pad_img.shape[1]),
                                    level=self.level, top_n=self.top_n, desire_size=self.input_size, orig_mask=None)
        instances = torch.FloatTensor(len(results['tiles']),
                                      self.num_channels, self.input_size, self.input_size)
        tile_idxs = torch.FloatTensor(tile_idxs.tolist())
        for i, tile in enumerate(results['tiles']):
            if self.transform:
                instances[i, :, :, :] = self.transform(tile)

        if self.phase == "w_atts":
            if len(tile_idxs) < len(instances):
                tile_idxs = torch.cat([tile_idxs, torch.zeros(len(instances) - len(tile_idxs))], dim=0)
            return instances, slide_info.image_id, tile_idxs, pad_top, pad_left
        return instances, slide_info.image_id


