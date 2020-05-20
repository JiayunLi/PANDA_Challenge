from abc import ABC, abstractmethod


class TileGeneratorABC(ABC):
    @abstractmethod
    def is_ihc_slide(self):
        """
        :return: if the slide is an IHC slide
        """

    @abstractmethod
    def get_tile_locations(self, tile_size, overlap, thres, dw_rate):
        """Return locations for tiles that can be extracted from the slide
        """

    @abstractmethod
    def extract_tile(self, location, tile_size, dw_rate, normalizer=None):
        """Given the location and normalizer,
        return the original and normalized tile at desired magnification"""

    @abstractmethod
    def extract_label_mask(self, location, tile_size, dw_rate):
        """
        :param location:
        :param tile_size:
        :param dw_rate:
        :return: labeled mask for a given tile
        """

    @abstractmethod
    def extract_all_tiles(self, tile_size, overlap, thres, dw_rate, normalizer=None, w_label_mask=True):
        """
        :param tile_size: Size to extract tiles at highest scanned magnification
        :param overlap: Overlap percentage at highesst scanned magnification
        :param thres: Keep tiles with tissue areas higher than the thres
        :param dw_rate: Down sample tiles
        :param normalizer: Normalization method
        :param w_label_mask: Whether to extract labeled masks
        :return: orig_tiles, norm_tiles, locations, label_masks arrays
        """






