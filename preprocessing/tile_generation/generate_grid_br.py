# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.tile_generation.generate_grid import TileGeneratorGrid
import numpy as np
import time
import heapq
from preprocessing.tissue_detection import threshold_based


class TileGeneratorGridBr(TileGeneratorGrid):
    def __init__(self, slides_dir, slide_name, masks_dir=None, check_ihc=False, verbose=False):
        super().__init__(slides_dir, slide_name, masks_dir, check_ihc, verbose)

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
        tile_size *= dw_rate
        overlap = int(overlap * tile_size)
        lowest_rate = int(self.slide.level_downsamples[self.slide.level_count - 1])
        # Calculate the tile size to be used on
        small_tile_size = int(tile_size / lowest_rate)
        small_overlap = int(overlap / lowest_rate)
        interval = small_tile_size - small_overlap
        counter = 0
        location_tracker = {}
        tissue_roi, slide_br = threshold_based.get_tissue_area(self.slide, w_br=True)

        i_limit = (self.slide.level_dimensions[0][1] - tile_size) / lowest_rate
        j_limit = (self.slide.level_dimensions[0][0] - tile_size) / lowest_rate
        heap = []
        for i in range(0, int(self.slide.level_dimensions[self.slide.level_count - 1][1]), interval):
            for j in range(0, int(self.slide.level_dimensions[self.slide.level_count - 1][0]), interval):
                if i <= i_limit and j <= j_limit:
                    size_x = small_tile_size if i + small_tile_size <= \
                                                self.slide.level_dimensions[self.slide.level_count - 1][1] \
                        else self.slide.level_dimensions[self.slide.level_count - 1][1] - i
                    size_y = small_tile_size if j + small_tile_size <= \
                                                self.slide.level_dimensions[self.slide.level_count - 1][0] \
                        else self.slide.level_dimensions[self.slide.level_count - 1][0] - j
                    small_tile = tissue_roi[i:i + size_x, j:j + size_y]
                    if (float(np.count_nonzero(small_tile)) / float(size_x * size_y)) >= thres:
                        mean_br = np.mean(slide_br[i:i + size_x, j:j + size_y])
                        if len(heap) >= top_n:
                            if heap[0][1] < mean_br:
                                heapq.heappop(heap)
                                heapq.heappush(heap, (counter, mean_br))
                                location_tracker[counter] = (int(j * lowest_rate), int(i * lowest_rate))
                                counter += 1
                        else:
                            heapq.heappush(heap, (counter, mean_br))
                            location_tracker[counter] = (int(j * lowest_rate), int(i * lowest_rate))
                            counter += 1
        if self.verbose:
            print("Generate %d tiles in the grid" % counter)
        return counter, location_tracker, heap

    def extract_top_tiles(self, tile_size, overlap, thres, dw_rate, top_n, normalizer=None):
        """
        :param tile_size:
        :param overlap:
        :param thres:
        :param dw_rate:
        :param top_n
        :param normalizer:
        :return:
        """
        start_time = time.time()

        counter, location_tracker, top_tile_brs = self.get_tile_locations(tile_size, overlap, thres, dw_rate, top_n)
        norm_tiles, orig_tiles = [], []
        locations = np.zeros((len(top_tile_brs), 2), dtype=np.int64)
        idx = 0
        level = self.get_read_level(dw_rate)
        for tile_id, _ in top_tile_brs:
            cur_loc = location_tracker[tile_id]
            # generate normalized tiles
            orig_tile, norm_tile, _ = \
                self.extract_tile([int(cur_loc[0]), int(cur_loc[1])],
                                  tile_size, level, normalizer=normalizer,
                                  return_image=True)

            norm_tiles.append(norm_tile)
            locations[idx, 0] = int(cur_loc[0])
            locations[idx, 1] = int(cur_loc[1])
            idx += 1

        if self.verbose:
            print("Time to generate %d tiles from %s slide: %.2f" % (idx, str(self.slide_id), time.time() - start_time))
        return norm_tiles, orig_tiles, locations