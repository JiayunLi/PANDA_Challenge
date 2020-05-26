import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.tile_generation.tile_generation_abc import TileGeneratorABC
from preprocessing.tile_generation.utils import prep_utils
from PIL import Image
import numpy as np
import openslide
import time
from preprocessing.tissue_detection import threshold_based

"""
This tile generation script is only intended for slides with almost same maximum magnification. 
Need to specify tile_size at highest magnification
This code doesn't support dataset that has slides scanned at different highest magnifications!
"""

class TileGeneratorGrid(TileGeneratorABC):
    def __init__(self, slides_dir, slide_name, masks_dir=None, check_ihc=False, verbose=False):
        """
        Create a DeepZoomGenerator wrapping an OpenSlide object.
        :param slides_dir: location for the slide
        :param slide_name: slide_name should include .svs / .tiff

        :param check_ihc: whether to check current slide is IHC slide
        :param verbose: print logs
        """
        self.slide = openslide.OpenSlide(f"{slides_dir}/{slide_name}")
        self.slide_id = slide_name.split(".")[0]
        if os.path.isfile(f'{masks_dir}/{self.slide_id}_mask.tiff'):
            self.label_mask = openslide.OpenSlide(f'{masks_dir}/{self.slide_id}_mask.tiff')
        else:
            self.label_mask = None

        # whether to print logs
        self.verbose = verbose
        self.ihc = prep_utils.check_ihc_slide(self.slide) if check_ihc else False

    def is_ihc_slide(self):
        return self.ihc

    # Generate tiles from grid, with optional overlap
    def get_tile_locations(self, tile_size, overlap, thres, dw_rate, top_n=None):
        """
        Generate tile locations from the grid

        :param tile_size: Tile size at highest scanned magnification (in this case: 20x)
        :param overlap: Overlapping size (only used when tiles are extracted from the grid)
        :param thres: tissue threshold (i.e., tile contains tissue region greater than thres)
        :return: counter: how many tiles were generated
                 location_tracker: tile locations
        """
        # how much overlap on the required magnification
        # Relative rate change: image of size 128 downsample 4 times -> of size 512 at original scanning magnification.
        tile_size = dw_rate * tile_size
        overlap = int(overlap * tile_size)
        # Get the lowest rate for ROI
        lowest_rate = int(self.slide.level_downsamples[self.slide.level_count - 1])
        # Calculate the tile size to be used on
        small_tile_size = int(tile_size / lowest_rate)
        small_overlap = int(overlap / lowest_rate)
        interval = small_tile_size - small_overlap
        counter = 0
        location_tracker = {}
        tissue_roi = threshold_based.get_tissue_area(self.slide)
        for i in range(0, int(self.slide.level_dimensions[self.slide.level_count - 1][1]), interval):
            for j in range(0, int(self.slide.level_dimensions[self.slide.level_count - 1][0]), interval):
                if (i * lowest_rate + tile_size) <= self.slide.level_dimensions[0][1] and (
                            j * lowest_rate + tile_size) <= self.slide.level_dimensions[0][0]:
                    size_x = small_tile_size if i + small_tile_size <= \
                                                self.slide.level_dimensions[self.slide.level_count - 1][1] \
                        else self.slide.level_dimensions[self.slide.level_count - 1][1] - i
                    size_y = small_tile_size if j + small_tile_size <= \
                                                self.slide.level_dimensions[self.slide.level_count - 1][0] \
                        else self.slide.level_dimensions[self.slide.level_count - 1][0] - j
                    small_tile = tissue_roi[i:i + size_x, j:j + size_y]
                    if (float(np.count_nonzero(small_tile)) / float(size_x * size_y)) >= thres:
                        # Original slide were down-sampled by lowest_rate to separate tissue from background
                        location_tracker[counter] = np.zeros((2,))
                        location_tracker[counter][0] = int(j * lowest_rate)
                        location_tracker[counter][1] = int(i * lowest_rate)
                        counter += 1
        if self.verbose:
            print("Generate %d tiles in the grid" % counter)
        return counter, location_tracker

    def get_read_level(self, dw_rate):
        for level, rate in enumerate(self.slide.level_downsamples):
            if int(rate) == dw_rate:
                return level
            if int(rate) > dw_rate:
                return level - 1
        return len(self.slide.level_downsamples) - 1

    # Generate tiles based on location at highest magnification
    def extract_tile(self, location, tile_size, level, normalizer=None, return_image=False):
        # (x, y) tuple giving the top left pixel in the level 0 reference frame
        orig_tile = self.slide.read_region((location[0], location[1]), level, (tile_size, tile_size))
        orig_tile = np.asarray(orig_tile.convert('RGB'))
        _, tissue_mask = prep_utils.generate_binary_mask(orig_tile)

        norm_tile = None
        if normalizer:
            norm_method = normalizer.get_norm_method()
            try:
                if norm_method == 'reinhard':
                    norm_tile = \
                        Image.fromarray(normalizer.transform(orig_tile.astype(np.uint8), tissue_mask).astype(np.uint8))
                else:
                    norm_tile = Image.fromarray(normalizer.transform(orig_tile).astype(np.uint8))
            except:
                print(self.slide_id)
                orig_tile = Image.fromarray(orig_tile)
                norm_tile = orig_tile
                orig_tile.save("error_tile.png")
        else:
            norm_tile = Image.fromarray(orig_tile)

        # if norm_tile.size[0] > tile_size:
        #     orig_tile = Image.fromarray(orig_tile.astype(np.uint8)).resize((tile_size, tile_size),
        #                                                                    Image.ANTIALIAS)
        #     norm_tile = norm_tile.resize((tile_size, tile_size),Image.ANTIALIAS)
        #     rate = norm_tile.size[0] // tile_size
        #     tissue_mask = tissue_mask[::rate, ::rate]
        if return_image:
            return Image.fromarray(orig_tile), norm_tile, tissue_mask
        else:
            return np.asarray(orig_tile), np.asarray(norm_tile), tissue_mask

    def extract_label_mask(self, location, tile_size, level):
        tile_mask = self.label_mask.read_region((location[0], location[1]), level, (tile_size, tile_size))
        tile_mask = np.asarray(tile_mask.split()[0])
        if tile_mask.shape[0] > tile_size:
            rate = tile_mask.shape[0] // tile_size
            tile_mask = tile_mask[::rate, ::rate]
        return tile_mask

    def extract_all_tiles(self, tile_size, overlap, thres, dw_rate, normalizer=None, w_label_mask=True):
        """
        :param tile_size:
        :param overlap:
        :param thres:
        :param dw_rate:
        :param normalizer:
        :param w_label_mask:
        :return:
        """
        start_time = time.time()

        counter, location_tracker = self.get_tile_locations(tile_size, overlap, thres, dw_rate)
        norm_tiles = np.zeros((counter, tile_size, tile_size, 3), dtype=np.uint8)
        orig_tiles = np.zeros((counter, tile_size, tile_size, 3), dtype=np.uint8)
        tissue_masks = np.zeros((counter, tile_size, tile_size), dtype=np.uint8)
        locations = np.zeros((counter, 2), dtype=np.int64)
        get_label_mask = self.label_mask and w_label_mask
        if get_label_mask:
            label_masks = np.zeros((counter, tile_size, tile_size), dtype=np.uint8)
        else:
            label_masks = None
        level = self.get_read_level(dw_rate)

        for tile_id in range(counter):
            cur_loc = location_tracker[tile_id]
            # generate normalized tiles
            orig_tile, norm_tile, tissue_mask = \
                self.extract_tile([int(cur_loc[0]), int(cur_loc[1])],
                                  tile_size, level, normalizer=normalizer)

            orig_tiles[tile_id, :, :, :] = orig_tile
            norm_tiles[tile_id, :, :, :] = norm_tile
            tissue_masks[tile_id, :, :] = tissue_mask.astype(np.uint8)
            locations[tile_id, 0] = int(cur_loc[0])
            locations[tile_id, 1] = int(cur_loc[1])

            if get_label_mask:
                label_mask = self.extract_label_mask(
                    [int(cur_loc[0]), int(cur_loc[1])], tile_size, level)
                label_masks[tile_id, :, :] = label_mask.astype(np.uint8)
        if self.verbose:
            print("Time to generate %d tiles from %s slide: %.2f"
                  % (counter, str(self.slide_id), time.time() - start_time))
        return orig_tiles, norm_tiles, locations, tissue_masks, label_masks

    def extract_top_tiles(self, param, param1, param2, param3, param4, normalizer):
        pass






