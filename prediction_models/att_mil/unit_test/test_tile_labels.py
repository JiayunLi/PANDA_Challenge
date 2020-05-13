import pandas as pd
import json
import lmdb


def check_tile_name_exist(slide_tiles_mapping_file, trainval_tiles_file):
    slide_tiles_mapping = json.load(open(slide_tiles_mapping_file, "r"))
    trainval_tiles = pd.read_excel(trainval_tiles_file, index_col='tile_name')
    trainval_tilenames = set(list(trainval_tiles.index))
    for slide, tile_names in slide_tiles_mapping.items():
        for tile_name in tile_names:
            if tile_name not in trainval_tilenames:
                print(f"{tile_name} not Found!!!")


def check_tile_label(trainval_tiles_file, trainval_file):
    trainval_tiles = pd.read_excel(trainval_tiles_file, index_col='tile_name')

