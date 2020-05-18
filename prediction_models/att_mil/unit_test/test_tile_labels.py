import pandas as pd
import json
import lmdb
import argparse


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


def check_slide_mapping_file(slide_tiles_mapping_file, trainval_file, log_dir):
    trainval = pd.read_csv(trainval_file)
    slide_tiles_mapping_file = json.load(open(slide_tiles_mapping_file, "r"))
    errors = []

    for i in range(len(trainval)):
        data = trainval.iloc[i]
        slide_name = str(data['image_id'])
        if slide_name == "3790f55cad63053e956fb73027179707":
            continue
        if slide_name not in slide_tiles_mapping_file:
            print(slide_name)
            errors.append({'slide_name': slide_name})

    log_df = pd.DataFrame(columns=["slide_name"], data=errors)
    log_df.to_csv(f"{log_dir}/slides_not_in_mapping.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attention MIL PANDA challenge')

    # File location
    parser.add_argument('--data_dir', type=str, default='/data/', help='Root directory for processed data')
    parser.add_argument('--log_dir', type=str, default='./cache/logs/')

    args = parser.parse_args()
    check_slide_mapping_file(f"{args.data_dir}/slides_tiles_mapping.json", f"{args.data_dir}/trainval.csv",
                             args.log_dir)