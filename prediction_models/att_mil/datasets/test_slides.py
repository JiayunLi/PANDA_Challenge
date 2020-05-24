"""
DataLoader to extract tiles from a slide
"""
from PIL import Image
from preprocessing.tile_generation import generate_grid_br as generate_grid
import torch
import torch.utils.data as data


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
        norm_tiles, _ = tile_generator.extract_top_tiles(self.params.im_size, self.params.overlap,
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
        norm_tiles, _ = tile_generator.extract_top_tiles(self.params.im_size, self.params.overlap,
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
