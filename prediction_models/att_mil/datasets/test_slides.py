"""
DataLoader to extract tiles from a slide
"""
import pandas as pd
from preprocessing.tile_generation import generate_grid
import torch


class BiopsySlides(data.Dataset):
    def __init__(self, params, transform, tile_normalizer):
        self.params = params
        self.test_df = pd.read_csv(params.test_file)
        self.test_slides_dir = params.test_slides_dir
        self.transform = transform
        self.tile_normalizer = tile_normalizer

    def __len(self):
        return len(self.test_df)

    def __getitem__(self, ix):
        slide_info = self.test_df.iloc[ix]
        tile_generator = generate_grid.TileGeneratorGrid(self.test_slides_dir,
                                                         f"{slide_info.image_id}.tiff", None, verbose=False)
        _, norm_tiles, _, _, _ \
            = tile_generator.extract_all_tiles(self.params.im_size, self.params.overlap,
                                               self.params.ts_thres, self.params.dw_rate, self.tile_normalizer)
        instances = torch.FloatTensor(len(norm_tiles), params.num_channels, params.input_size, params.input_size)
        for i, norm_tile in enumerate(norm_tiles):
            if self.transform:
                instances[i, :, :, :] = self.transform(norm_tile)
        return instances
