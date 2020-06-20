"""
DataLoader to extract tiles from a slide
"""
from PIL import Image
from preprocessing.tile_generation import generate_grid_br as generate_grid
import torch
import torch.utils.data as data
import numpy as np
import skimage.io
import openslide
from prediction_models.att_mil.datasets import gen_selected_tiles, get_selected_locs


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
                                    level=self.level, top_n=self.top_n, desire_size=self.input_size, orig_mask=None,
                                    normalize=self.params.stain_norm)
        instances = torch.FloatTensor(len(results['tiles']),
                                      self.num_channels, self.input_size, self.input_size)

        tile_idxs = torch.FloatTensor(tile_idxs.tolist())
        for i, tile in enumerate(results['tiles']):

            if self.transform:
                instances[i, :, :, :] = self.transform(tile)

        if self.phase == "w_atts":
            padded_low_shape = pad_img.shape
            sub_tile_locs = \
                get_selected_locs.select_sub_simple_4x4(tile_idxs, self.lowest_im_size, None,
                                                        pad_top, pad_left, lowest_sub_size=16, sub_size=64,
                                                        tiles=results['tiles'], max_per_tile=2, n_row=results['nrow'],
                                                        n_col=results['ncol'])
            sub_tile_locs = sub_tile_locs['high_res']
            sub_tile_locs = torch.FloatTensor(sub_tile_locs)
            if len(tile_idxs) < len(instances):
                pad_n = len(instances) - len(tile_idxs)
                # tile_idxs = torch.cat([tile_idxs, torch.zeros(pad_n) - 1], dim=0)
                # n_tiles, n_sub_tiles, xy
                sub_tile_locs = torch.cat([sub_tile_locs, torch.zeros(pad_n, 2, 2) - 1], dim=0)
            # return instances, slide_info.image_id, tile_idxs, pad_top, pad_left, results['nrow'], results['ncol']
            return instances, slide_info.image_id, sub_tile_locs
        return instances, slide_info.image_id


class BiopsyHighresSelected(data.Dataset):
    def __init__(self, dataset_params, test_df, selected_locs, transform,  phase='test'):
        self.transform = transform
        self.params = dataset_params
        self.phase = phase
        self.selected_locs = selected_locs
        self.slides_df = test_df

    def __len__(self):
        return len(self.slides_df)

    def __getitem__(self, ix):
        slide_info = self.slides_df.iloc[ix]
        slide_id = slide_info.image_id
        tiles = self._get_tiles(slide_id)
        return tiles

    def _get_tiles(self, slide_id):
        rate = gen_selected_tiles.RATE_MAP[-3]
        high_im_size = self.params.lowest_im_size * rate

        cur_slide = openslide.OpenSlide(f"{self.params.test_slides_dir}/{slide_id}.tiff")
        cur_im_shape = (cur_slide.level_dimensions[self.params.level + 3][1],
                        cur_slide.level_dimensions[self.params.level + 3][0])
        lowest_locs = self.selected_locs[slide_id]
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
            high_tile = cur_slide.read_region((high_j, high_i), self.params.level + 3,
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