import skimage.io
from PIL import Image
import pandas as pd
import numpy as np
import time
from prediction_models.att_mil.utils import reinhard_fast
from skimage import color
import json
from skimage import morphology as skmp

# -1: lowest resolution
RATE_MAP = {-1: 1, -2: 4, -3: 16}


def default(o):
    if isinstance(o, np.int64): return int(o)
    if isinstance(o, tuple): return o
    raise TypeError


def compute_blue_ratio(dw_sample):
    tempt1 = (100.0 * dw_sample[:, :, 2]) / (1.0 + dw_sample[:, :, 0] + dw_sample[:, :, 1])
    tempt2 = 256.0 / (1.0 + dw_sample[:, :, 0] + dw_sample[:, :, 1] + dw_sample[:, :, 2])
    br = tempt1 * tempt2
    return br


def get_tissue_roi(dw_sample):
    dw_sample_hsv = color.rgb2hsv(dw_sample)

    # Get first ROI to remove all kinds of markers (Blue, Green, black)
    roi1 = (dw_sample_hsv[:, :, 0] <= 0.67) | (
            (dw_sample_hsv[:, :, 1] <= 0.15) & (dw_sample_hsv[:, :, 2] <= 0.75))
    # exclude marker roi
    roi1 = ~roi1
    skmp.remove_small_holes(roi1, area_threshold=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi1, min_size=300, connectivity=20, in_place=True)

    # remove background: regions with low value(black) or very low saturation (white)
    roi2 = (dw_sample_hsv[:, :, 1] >= 0.05) & (dw_sample_hsv[:, :, 2] >= 0.25)
    roi2 *= roi1

    skmp.remove_small_holes(roi2, area_threshold=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi2, min_size=300, connectivity=20, in_place=True)
    return roi2


def get_padded(img, sz):
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = np.pad(img, [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]], constant_values=255)
    return img, pad0 // 2, pad1 // 2


def tile_padded(padded_img, sz, top_n):

    padded_img = padded_img.reshape(padded_img.shape[0] // sz, sz, padded_img.shape[1] // sz, sz, 3)
    padded_img = padded_img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)

    if len(padded_img) < top_n:
        padded_img = np.pad(padded_img, [[0, top_n - len(padded_img)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(padded_img.reshape(padded_img.shape[0], -1).sum(-1))[:top_n]
    padded_img = padded_img[idxs]
    return padded_img, idxs[:top_n]


def select_at_lowest(img, sz, n_tiles, mask_tissue):
    pad_img, pad_top, pad_left = get_padded(img, sz)
    br_img = compute_blue_ratio(pad_img)
    if mask_tissue:
        roi = get_tissue_roi(pad_img)
        br_img *= roi
    br_img = br_img.reshape(br_img.shape[0] // sz, sz, br_img.shape[1] // sz, sz)
    br_img = br_img.transpose(0, 2, 1, 3).reshape(-1, sz, sz)
    # Select with blue ratio
    idxs = np.argsort(br_img.reshape(br_img.shape[0], -1).sum(-1))[::-1][:n_tiles]

    return pad_img, idxs, pad_top, pad_left


def get_highres_tiles(orig_img, selected_idxs, pad_top, pad_left, low_im_size, padded_low_shape,
                      level, top_n, desire_size, orig_mask=None):
    cur_rate = RATE_MAP[level]
    cur_im = orig_img[level]
    if orig_mask:
        try:
            cur_mask = orig_mask[level][:, :, 0]
        except:
            cur_mask = orig_mask[level+1][:, :, 0]
            cur_mask = cur_mask.repeat(4, axis=0).repeat(4, axis=1)
    tiles, masks, norm_tiles = [], [], []
    high_im_size = low_im_size * cur_rate
    n_row, n_col = padded_low_shape[0] // low_im_size, padded_low_shape[1] // low_im_size
    normalizer = reinhard_fast.ReinhardNormalizer()
    for tile_id in selected_idxs:
        i, j = tile_id // n_col, tile_id % n_col
        high_i = max(i * low_im_size - pad_top, 0) * cur_rate
        high_j = max(j * low_im_size - pad_left, 0) * cur_rate
        if high_i + high_im_size > cur_im.shape[0]:
            high_i = cur_im.shape[0] - high_im_size

        if high_j + high_im_size > cur_im.shape[1]:
            high_j = cur_im.shape[1] - high_im_size
        high_tile = cur_im[high_i: high_i + high_im_size, high_j: high_j + high_im_size, :].astype(np.uint8)
        norm_high_tile = normalizer.transform(high_tile)
        if high_im_size > desire_size:
            high_tile = Image.fromarray(high_tile.astype(np.uint8)).resize((desire_size, desire_size), Image.ANTIALIAS)
            norm_high_tile = Image.fromarray(norm_high_tile.astype(np.uint8)).resize((desire_size, desire_size), Image.ANTIALIAS)
            norm_high_tile = np.asarray(norm_high_tile)
            high_tile = np.asarray(high_tile)
        if orig_mask:
            high_tile_mask = cur_mask[high_i: high_i + high_im_size, high_j: high_j + high_im_size]
            if high_im_size > desire_size:
                rate = high_im_size // desire_size
                high_tile_mask = high_tile_mask[::rate, ::rate]
            masks.append(high_tile_mask)
        tiles.append(high_tile)
        norm_tiles.append(norm_high_tile)

    if len(tiles) < top_n:
        tiles = np.pad(tiles, [[0, top_n - len(tiles)], [0, 0], [0, 0], [0, 0]], constant_values=255, mode='constant')
    tiles = np.stack(tiles)
    norm_tiles = np.stack(norm_tiles)
    results = {"tiles": tiles.astype(np.uint8), "ids": selected_idxs, 'norm_tiles': norm_tiles.astype(np.uint8)}
    if orig_mask:
        masks = np.pad(masks, [[0, top_n - len(masks)], [0, 0], [0, 0]], constant_values=0, mode='constant')
        masks = np.stack(masks)
        results["label_masks"] = masks.astype(np.uint8)
    else:
        results['label_masks'] = None
    return results


def generate_helper(pqueue, slides_dir, masks_dir, lowest_im_size, level, top_n, desire_size, slides_list):
    counter = 0
    for slide_name in slides_list:
        orig = skimage.io.MultiImage(f"{slides_dir}/{slide_name}.tiff")
        if os.path.isfile(f"{masks_dir}/{slide_name}_mask.tiff"):
            mask = skimage.io.MultiImage(f"{masks_dir}/{slide_name}_mask.tiff")
        else:
            mask = None
        lowest = orig[-1]
        pad_img, idxs, pad_top, pad_left = select_at_lowest(lowest, lowest_im_size, top_n, True)
        results = get_highres_tiles(orig, idxs, pad_top, pad_left, lowest_im_size,
                                    (pad_img.shape[0], pad_img.shape[1]),
                                    level=level, top_n=top_n, desire_size=desire_size, orig_mask=mask)
        results['slide_name'] = slide_name
        pqueue.put(results)
        counter += 1
        print("Put tiled slide [%s] on to queue: [%d]/[%d]" % (slide_name, counter, len(slides_list)))
    pqueue.put('Done')


def write_batch_data(env_tiles, env_norm, env_label_masks, tile_ids_map, batch_data, tot_len, start_counter):
    end_counter = start_counter + len(batch_data)
    with env_tiles.begin(write=True) as txn_tiles, env_norm.begin(write=True) as txn_norm, \
            env_label_masks.begin(write=True) as txn_labels:
        while len(batch_data) > 0:
            data = batch_data.pop()
            write_start = time.time()
            slide_name = data['slide_name']
            print(data['tiles'].shape)
            txn_tiles.put(str(slide_name).encode(), (data['tiles'].astype(np.uint8)).tobytes())
            txn_norm.put(str(slide_name).encode(), (data['norm_tiles'].astype(np.uint8)).tobytes())
            if data['label_masks'] is None:
                tempt = None
            else:
                txn_labels.put(str(slide_name).encode(), data['label_masks'].astype(np.uint8).tobytes())
            tile_ids_map[slide_name] = data['ids']
    print("Finish writing [%d]/[%d], time: %f" % (end_counter, tot_len, time.time() - write_start))
    return end_counter


def save_tiled_lmdb(slides_list, num_ps, write_batch_size, out_dir, slides_dir, masks_dir, lowest_im_size, level,
                    top_n, desire_size, loc_only):
    slides_to_process = []
    env_norm = lmdb.open(f"{out_dir}/norm_tiles", map_size=6e+12)
    env_tiles = lmdb.open(f"{out_dir}/tiles", map_size=6e+12)
    env_label_masks = lmdb.open(f"{out_dir}/label_masks", map_size=6e+11)
    tile_ids_map = dict()
    if not loc_only:
        print("Get not processed list")
        with env_label_masks.begin(write=False) as txn:
            for slide_name in slides_list:
                if txn.get(slide_name.encode()) is None:
                    slides_to_process.append(slide_name)
                    print(len(slides_to_process))
    else:
        slides_to_process = slides_list
    # slides_to_process = slides_to_process[:5]
    print("Total %d slides to process" % len(slides_to_process))
    batch_size = len(slides_to_process) // num_ps
    reader_processes = []
    pqueue = Queue()
    start_idx = 0
    for i in range(num_ps - 1):
        end_idx = start_idx + batch_size
        reader_p = Process(target=generate_helper, args=(pqueue, slides_dir, masks_dir, lowest_im_size,
                                                         level, top_n, desire_size,
                                                         slides_to_process[start_idx: end_idx]))
        reader_p.start()
        reader_processes.append(reader_p)
        start_idx = end_idx
    # Ensure all slides are processed by processes.
    reader_p = Process(target=generate_helper, args=(pqueue, slides_dir, masks_dir, lowest_im_size,
                                                     level, top_n, desire_size,
                                                     slides_to_process[start_idx: len(slides_to_process)]))
    reader_p.start()
    reader_processes.append(reader_p)

    counter, num_done = 0, 0
    batches = []

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
            batches.append(data)
        if not loc_only:
            # Write a batch of data.
            if len(batches) == write_batch_size:
                counter = \
                    write_batch_data(env_tiles, env_norm, env_label_masks, tile_ids_map, batches,
                                     len(slides_to_process), counter)
        else:
            slide_name = data['slide_name']
            tile_ids_map[slide_name] = data['ids'].tolist()
            counter += 1
            print(f"Put tile location {counter}/{len(slides_to_process)}")
    if not loc_only:
        if len(batches) > 0:
            counter = \
                write_batch_data(env_tiles, env_norm, env_label_masks, tile_ids_map, batches,
                                 len(slides_to_process), counter)

    for process in reader_processes:
        process.join()
    assert counter == len(slides_to_process), "%d processed slides, %d slides to be processed" \
                                              % (counter, len(slides_to_process))
    assert len(tile_ids_map) == len(slides_to_process)
    np.save(f"{out_dir}/tile_lowest_ids.npy", tile_ids_map)


def fix():
    from prediction_models.att_mil.utils import file_utils
    import tqdm
    orig_norm_env = lmdb.open("/data/br_256_2x/norm_tiles_tempt/",
                              max_readers=3, readonly=True,
                              lock=False, readahead=False, meminit=False)
    norm_env = lmdb.open("/data/br_256_2x/norm_tiles", map_size=6e+12)
    df = pd.read_csv("/data/4_fold_train.csv")
    with orig_norm_env.begin(write=False) as orig_txn, norm_env.begin(write=True) as new_txn:
        for i in tqdm.tqdm(range(len(df))):
            cur = df.iloc[i]
            image_id = cur['image_id']
            orig_tiles = orig_txn.get(image_id.encode())
            orig_tiles = file_utils.decode_buffer(orig_tiles, (-1, 512, 512, 3), data_type=np.uint8)
            new_tiles = []
            for tile_id in range(len(orig_tiles)):
                cur_tile = Image.fromarray(orig_tiles[tile_id, :, :, :]).resize((256, 256), Image.ANTIALIAS)
                new_tiles.append(cur_tile.asarray())
            new_tiles = np.stack(new_tiles)
            new_txn.put(str(image_id).encode(), (new_tiles.astype(np.uint8)).tobytes())



if __name__ == "__main__":
    from multiprocessing import Process, Queue
    import lmdb
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/storage_slides/PANDA_challenge/")
    parser.add_argument("--slides_dir", default="train_images/")
    parser.add_argument('--masks_dir', default='train_label_masks/',
                        help='location for data index files')
    parser.add_argument('--train_slide_file', default="4_fold_train.csv")
    parser.add_argument('--out_dir', default='/br_256_256/')

    parser.add_argument("--lowest_im_size", default=64, type=int)
    parser.add_argument('--desire_size', default=256, type=int)
    parser.add_argument("--level", default=-2, type=int, help="Generate tiles downsampled")
    parser.add_argument("--verbose", action='store_true', help="Whether to print debug information")

    parser.add_argument("--num_ps", default=5, type=int, help="How many processor to use")
    parser.add_argument("--write_batch_size", default=10, type=int, help="Write of batch of n slides")
    parser.add_argument('--top_n', type=int, default=40)

    parser.add_argument('--loc_only', action='store_true')
    parser.add_argument('--fix', action='store_true')

    args = parser.parse_args()
    if args.fix:
        fix()
        exit()
    args.slides_dir = f"{args.data_dir}/{args.slides_dir}/"
    args.masks_dir = f"{args.data_dir}/{args.masks_dir}/"
    args.train_slide_file = f"{args.data_dir}/{args.train_slide_file}"
    # args.out_dir = f"{args.data_dir}/{args.out_dir}/"

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    pickle.dump(args, open(f"{args.out_dir}/dataset_options.pkl", "wb"))
    process_list = pd.read_csv(args.train_slide_file)['image_id'].to_list()
    save_tiled_lmdb(process_list, args.num_ps, args.write_batch_size, args.out_dir, args.slides_dir, args.masks_dir,
                    args.lowest_im_size, args.level, args.top_n, args.desire_size, args.loc_only)


# python -m prediction_models.att_mil.datasets.gen_selected_tiles --data_dir /data/