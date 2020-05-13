import pandas as pd
import os
import shutil
import lmdb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from prediction_models.att_mil.utils import file_utils
from prediction_models.att_mil.utils import convert_labels


def parse_gleason(raw_gleason):
    if raw_gleason == "negative":
        return 0, 0
    tempt = raw_gleason.split("+")
    pg, sg = int(tempt[0]), int(tempt[1])
    return pg, sg


# Generate tile label given a tile label mask
def generate_tile_label(lmdb_dir, tile_info_dir, mask_size, trainval_file, binary_label=False):
    tile_label_data = []
    env_label_masks = lmdb.open(f"{lmdb_dir}", max_readers=3, readonly=True, lock=False,
                                readahead=False, meminit=False)
    logs = []
    rad_converter = convert_labels.ConvertRad(logs, binary_label)
    karo_converter = convert_labels.ConvertKaro(logs, binary_label)

    trainval_df = pd.read_csv(trainval_file, index_col='image_id')
    with env_label_masks.begin() as txn:
        tot = txn.stat()['entries']
    counter = 0
    # Only for slide with label!!!
    with env_label_masks.begin(write=False) as txn_labels:
        for tile_name, mask_buff in txn_labels.cursor():
            tile_name = str(tile_name.decode('ascii'))
            slide_name = tile_name.split("_")[0]
            tile_mask = file_utils.decode_buffer(mask_buff, (mask_size, mask_size), np.uint8)
            slide_info = trainval_df.loc[slide_name]
            slide_pg, slide_sg = parse_gleason(slide_info.gleason_score)
            if slide_info.data_provider == "radboud":
                tile_label = rad_converter.convert(tile_mask, slide_name, slide_pg, slide_sg)
            else:
                tile_label = karo_converter.convert(tile_mask, slide_name, slide_pg, slide_sg)
            tile_loc_x = tile_name.split("_")[1]
            tile_loc_y = tile_name.split("_")[2]
            tile_label_data.append({
                "tile_name": tile_name,
                "tile_label": tile_label,
                "loc_x": tile_loc_x,
                "loc_y": tile_loc_y
            })
            print(f"Finished tile_label generation: {counter + 1}/{tot}")
            counter += 1

    tiles_data_df = pd.DataFrame(columns=["tile_name", "tile_label", "loc_x", "loc_y"],
                                 data=tile_label_data)
    tiles_data_df.to_csv(f"{tile_info_dir}/trainval_tiles.csv")
    return


# Generate n files for n fold cross validation
def generate_cv_split(trainval_file, out_dir, n_fold, seed, delete_dir=False):
    if not delete_dir and os.path.isdir(out_dir) and os.path.isfile(f"{out_dir}/train_{n_fold-1}.csv"):
        print("Cross validation file already generate!")
        return

    if delete_dir or (not os.path.isdir(out_dir)):
        if delete_dir:
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    train_df = pd.read_csv(trainval_file)
    splits = StratifiedKFold(n_splits=n_fold, random_state=seed, shuffle=True)
    for fold, (train_ids, valid_ids) in enumerate(splits.split(train_df, train_df.isup_grade)):
        cur_train_df = train_df.iloc[train_ids]
        cur_val_df = train_df.iloc[valid_ids]
        cur_train_df.to_csv(f"{out_dir}/train_{fold}.csv")
        cur_val_df.to_csv(f"{out_dir}/val_{fold}.csv")
    print("Finish generate cross valiadtion file")
    return


