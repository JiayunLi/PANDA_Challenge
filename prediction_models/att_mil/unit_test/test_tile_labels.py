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
    slide_tiles_mapping = json.load(open(slide_tiles_mapping_file, "r"))
    print(f"Total number of slides {len(set(slide_tiles_mapping.keys()))}")
    errors = []
    slides = set()
    for i in range(len(trainval)):
        data = trainval.iloc[i]
        slide_name = str(data['image_id'])
        if slide_name == "3790f55cad63053e956fb73027179707":
            continue
        if slide_name not in slide_tiles_mapping:
            print(slide_name)
            errors.append({'slide_name': slide_name})
        else:
            slides.add(slide_name)

    print(f"{len(slides)} slides in slides_mapping file, {len(trainval)} slides in train file")

    log_df = pd.DataFrame(columns=["slide_name"], data=errors)
    log_df.to_csv(f"{log_dir}/slides_not_in_mapping.csv")


def check_cv_file(info_dir, n_folds, slide_tiles_mapping_file):
    slide_tiles_mapping = json.load(open(slide_tiles_mapping_file, "r"))
    print(f"Total number of slides {len(set(slide_tiles_mapping.keys()))}")
    splits = ['train', 'val']
    for fold in range(n_folds):
        for split in splits:
            df = pd.read_csv(f"{info_dir}/{split}_{fold}.csv")
            print(f"{len(df)} slides in {split}")
            has = 0
            for i in range(len(df)):
                data = df.iloc[i]
                slide_name = str(data.image_id)
                if slide_name == "3790f55cad63053e956fb73027179707":
                    has += 1
                    continue
                if slide_name not in slide_tiles_mapping:
                    print(slide_name)
                else:
                    has += 1
            assert has == len(df), f"Some slides not in slide tile mapping file for fold {fold}, split {split}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attention MIL PANDA challenge')

    # File location
    parser.add_argument('--data_dir', type=str, default='/data/', help='Root directory for processed data')
    parser.add_argument('--info_dir', type=str, default='./info/')
    parser.add_argument('--log_dir', type=str, default='./cache/logs/')

    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()
    check_slide_mapping_file(f"{args.data_dir}/slides_tiles_mapping.json", f"{args.data_dir}/train.csv",
                             args.log_dir)
    check_cv_file(f"{args.info_dir}", args.n_folds, f"{args.data_dir}/slides_tiles_mapping.json")