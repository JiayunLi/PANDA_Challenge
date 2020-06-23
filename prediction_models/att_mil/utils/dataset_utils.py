import pandas as pd
import os
import json
import shutil
import lmdb
import numpy as np
import tqdm
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from prediction_models.att_mil.utils import file_utils
from prediction_models.att_mil.utils import convert_labels


def parse_gleason(raw_gleason):
    if raw_gleason == "negative":
        return 0, 0
    tempt = raw_gleason.split("+")
    pg, sg = int(tempt[0]), int(tempt[1])
    return pg, sg


def default(o):
    if isinstance(o, np.int64): return int(o)
    if isinstance(o, tuple): return o
    raise TypeError


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


# Generate tile label given a tile label mask
def generate_tile_label_json(lmdb_dir, tile_info_dir, mask_size, trainval_file, dataset_name="dw_sample_16",
                             binary_label=False):
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
    slide_tiles_labels = dict()
    with env_label_masks.begin(write=False) as txn_labels:
        for slide_name, masks_buff in txn_labels.cursor():
            slide_name = str(slide_name.decode('ascii'))
            all_masks = file_utils.decode_buffer(masks_buff, (-1, mask_size, mask_size), np.uint8)
            slide_info = trainval_df.loc[slide_name]
            slide_pg, slide_sg = parse_gleason(slide_info.gleason_score)
            slide_tiles_labels[slide_name] = []
            for i in range(len(all_masks)):
                tile_mask = all_masks[i, :, :]
                # Only use radbound label
                if slide_info.data_provider == "radboud":
                    tile_label = rad_converter.convert(tile_mask, slide_name, slide_pg, slide_sg)
                # else:
                #     tile_label = karo_converter.convert(tile_mask, slide_name, slide_pg, slide_sg)
                    slide_tiles_labels[slide_name].append(tile_label)
            print(f"Finished tile_label generation: {counter + 1}/{tot}")
            counter += 1
    json.dump(slide_tiles_labels, open(f"{tile_info_dir}/tile_labels_{dataset_name}.json", "w"), default=default)
    return


def convert_cv_split(trainval_file, out_dir, n_folds):
    trainval_df = pd.read_csv(trainval_file)
    for fold in range(n_folds):
        train_data = []
        val_data = []
        for i in range(len(trainval_file)):
            cur = trainval_df.iloc[i].to_dict()
            if int(cur['split']) == i:
                val_data.append(cur)
            else:
                train_data.append(cur)
        train_df = pd.DataFrame(data=train_data)
        val_df = pd.DataFrame(data=val_data)
        train_df.to_csv(f"{out_dir}/train_{fold}.csv")
        val_df.to_csv(f"{out_dir}/val_{fold}.csv")


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


def save_downsample_tiles(orig_tiles_lmdb_dir, dw_lmdb_dir, tile_size=512, dw_rate=4):
    orig_env = lmdb.open(f"{orig_tiles_lmdb_dir}/tiles", max_readers=3, readonly=True, lock=False,
                         readahead=False, meminit=False)
    dw_lmdb_env = lmdb.open(f"{dw_lmdb_dir}/tiles", map_size=6e+12)
    with orig_env.begin(write=False) as txn:
        tot = txn.stat()['entries']

    counter = 0
    with orig_env.begin(write=False) as txn_orig, dw_lmdb_env.begin(write=True) as txn_dw:
        for tile_name, tile_buff in txn_orig.cursor():
            tile = file_utils.decode_buffer(tile_buff, (tile_size, tile_size, 3), np.uint8)
            tile = Image.fromarray(tile)
            tile_dw = tile.resize((tile_size // dw_rate, tile_size // dw_rate), Image.ANTIALIAS)
            tile_dw = np.asarray(tile_dw)
            tile_name = str(tile_name.decode('ascii'))
            txn_dw.put(str(tile_name).encode(), tile_dw.astype(np.uint8).tobytes())
            counter += 1
            print(f"Finish write down sampled tiles: {counter}/{tot}")


def read_tiles_helper(pqueue, orig_env, tile_size, tile_labels_df, slide_tiles_map, slide_to_process):
    with orig_env.begin(write=False) as txn:
        for counter, slide_name in enumerate(slide_to_process):
            tile_names = slide_tiles_map[slide_name]
            tiles = np.zeros((len(tile_names), tile_size, tile_size, 3), dtype=np.uint8)
            labels = []
            tile_name_mapping = dict()
            for i, tile_name in enumerate(tile_names):
                if tile_name in tile_labels_df.index:
                    label = tile_labels_df.loc[tile_name].tile_label
                else:
                    label = -1
                labels.append(label)
                tile_name_mapping[i] = tile_name
                buffer = txn.get(str(tile_name).encode())
                buffer = np.frombuffer(buffer, dtype=np.uint8)
                buffer = buffer.reshape((tile_size, tile_size, 3))
                tiles[i, :, :, :] = buffer
            data = {
                "slide_name": slide_name,
                "tile_name_mapping": tile_name_mapping,
                "tiles": tiles,
                "labels": labels
            }
            pqueue.put(data)
            print(f"Finish decode tiles {counter + 1} / {len(slide_to_process)}")
        pqueue.put('Done')


def change_slide_encode(opts, tile_size=128):
    slide_tiles_map = json.load(open(f"{opts.orig_data_dir}/slides_tiles_mapping.json", "r"))
    orig_env = lmdb.open(f"{opts.orig_data_dir}/tiles", max_readers=3, readonly=True, lock=False,
                         readahead=False, meminit=False)
    env = lmdb.open(f"{opts.new_data_dir}/tiles", map_size=6e+12)
    tile_labels_df = pd.read_csv(opts.tile_labels_file, index_col='tile_name')
    slides_to_process = list(slide_tiles_map.keys())
    num_ps = 6
    batch_size = len(slides_to_process) // num_ps
    reader_processes = []
    pqueue = Queue()
    start_idx = 0
    for i in range(num_ps-1):
        end_idx = start_idx + batch_size
        reader_p = Process(target=read_tiles_helper, args=(pqueue, orig_env, tile_size, tile_labels_df, slide_tiles_map,
                                                         slides_to_process[start_idx: end_idx]))
        reader_p.start()
        reader_processes.append(reader_p)
        start_idx = end_idx
    reader_p = Process(target=read_tiles_helper, args=(pqueue, orig_env, tile_size, tile_labels_df, slide_tiles_map,
                                                       slides_to_process[start_idx: len(slides_to_process)]))
    reader_p.start()
    reader_processes.append(reader_p)

    counter, num_done = 0, 0
    tiles_id_name_map = dict()
    tiles_labels = dict()
    with env.begin(write=True) as txn:
        while True:
            # Block if necessary until an item is available.
            data = pqueue.get()
            # Done indicates job on one process is finished.
            if data == "Done":
                num_done += 1
                print("One part is done!")
                if num_done == num_ps:
                    break
            else:
                slide_name = data['slide_name']
                tiles = data['tiles']
                tiles_labels[slide_name] = data['labels']
                tiles_id_name_map[slide_name] = data['tile_name_mapping']
                txn.put(str(slide_name).encode(), tiles.astype(np.uint8).tobytes())
                counter += 1
                print(f"Finish write {counter} / {len(slides_to_process)}")

    for process in reader_processes:
        process.join()
    json.dump(tiles_id_name_map, open(f"{opts.new_data_dir}/tiles_id_name_map.json", "w"), default=default)
    json.dump(tiles_labels, open(f"{opts.new_data_dir}/tiles_labels.json", "w"), default=default)


def convert_img_lmdb(opts):
    env = lmdb.open(f"{opts.new_data_dir}/tiles", map_size=6e+12)
    df = pd.read_csv(f"{opts.orig_data_dir}/4_fold_train.csv")
    with env.begin(write=True) as txn:
        for i in tqdm.tqdm(range(len(df))):
            cur = df.iloc[i]
            image_id = cur.image_id
            tiles = np.zeros((16, 128, 128, 3), dtype=np.uint8)

            for tile_id in range(16):
                tile = Image.open(f"{opts.orig_data_dir}/train/{image_id}_{tile_id}.png")
                tiles[tile_id, :, :, :] = np.asarray(tile)
            txn.put(str(image_id).encode(), tiles.astype(np.uint8).tobytes())


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Attention MIL PANDA challenge')

    # File location
    parser.add_argument('--orig_data_dir', type=str, default='/data/jiayun/PANDA_Challenge/processed',
                        help='Root directory for processed data')

    parser.add_argument('--dw_sample', action='store_true')
    parser.add_argument('--dw_data_dir', type=str, default='/data/jiayun/PANDA_Challenge/dw_sampled_128',
                        help='Root directory for processed data')
    parser.add_argument('--dw_rate', type=int, default=4)

    parser.add_argument('--compress_slide', action="store_true")
    parser.add_argument('--tile_labels_file', default='./info/trainval_tiles.csv')
    parser.add_argument('--new_data_dir', type=str, default='/data/jiayun/PANDA_Challenge/slides_encode_128',
                        help='Root directory for processed data')

    parser.add_argument('--convert_imgs', action='store_true')

    args = parser.parse_args()
    if args.convert_imgs:
        convert_img_lmdb(args)
    if args.dw_sample:
        if not os.path.isdir(args.dw_data_dir):
            os.mkdir(args.dw_data_dir)
        save_downsample_tiles(args.orig_data_dir, args.dw_data_dir, dw_rate=args.dw_rate)
    if args.compress_slide:
        from multiprocessing import Process, Queue
        change_slide_encode(args, tile_size=128)
