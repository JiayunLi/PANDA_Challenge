import sys
from multiprocessing import Process, Queue
import numpy as np
import os
import argparse
import json
import pandas as pd
import time
import lmdb
sys.path.append("..")
from preprocessing.tile_generation import generate_grid
from preprocessing.normalization import reinhard_bg


def generate_helper(pqueue, slides_dir, masks_dir, tile_size, overlap, thres, dw_rate, verbose, slides_to_process):
    if verbose:
        print("Queue len: %d" % pqueue.qsize())
    tile_normalizer = reinhard_bg.ReinhardNormalizer()
    # use the pre-computed LAB mean and std values
    tile_normalizer.fit(None)
    counter = 0
    for slide_name in slides_to_process:
        tile_generator = generate_grid.TileGeneratorGrid(slides_dir, f"{slide_name}.tiff", masks_dir, verbose=verbose)
        # extract_all_tiles(self, tile_size, overlap, thres, dw_rate, normalizer=None, w_label_mask=True)
        _, norm_tiles, locations, tissue_masks, label_masks \
            = tile_generator.extract_all_tiles(tile_size, overlap, thres, dw_rate, tile_normalizer)
        if len(norm_tiles) == 0:
            counter += 1
            data = {
                "status": "empty",
                "slide_name": slide_name,
            }
            pqueue.put(data)
            continue

        data = {
            "slide_name": slide_name,
            "norm_tiles": norm_tiles,
            "tissue_masks": tissue_masks,
            "label_masks": label_masks,
            "locations": locations,
            "status": "normal"
        }
        pqueue.put(data)
        counter += 1
        print("Put tiled slide [%s] on to queue: [%d]/[%d]" % (slide_name, counter, len(slides_to_process)))
    pqueue.put('Done')


def write_batch_data(env_tiles, env_tissue_masks, env_label_masks, env_locations, batch_data, tot_len, start_counter, verbose):
    end_counter = start_counter + len(batch_data)
    with env_tiles.begin(write=True) as txn_tiles, env_tissue_masks.begin(write=True) as txn_masks, \
            env_label_masks.begin(write=True) as txn_labels, env_locations.begin(write=True) as txn_locs:
        while len(batch_data) > 0:
            data = batch_data.pop()
            write_start = time.time()
            slide_name = data['slide_name']
            # Encode each tile separately
            tot_n_tiles = len(data['norm_tiles'])
            for i in range(tot_n_tiles):
                cur_tile, cur_mask, cur_loc = data['norm_tiles'][i], data['tissue_masks'][i], data['locations'][i]
                tile_name = f"{slide_name}_{cur_loc[0]}_{cur_loc[1]}"
                txn_tiles.put(str(tile_name).encode(), cur_tile.astype(np.uint8).tobytes())
                txn_masks.put(str(tile_name).encode(), cur_mask.astype(np.uint8).tobytes())
                # Workaround to deal with deciding if an object is None or numpy array
                if data['label_masks'] is None:
                    cur_label = None
                else:
                    cur_label = data['label_masks'][i]
                    txn_labels.put(str(tile_name).encode(), cur_label.astype(np.uint8).tobytes())
            txn_locs.put(str(slide_name).encode(), data['locations'].astype(np.int64).tobytes())
    print("Finish writing [%d]/[%d], time: %f" % (end_counter, tot_len, time.time() - write_start))
    return end_counter


def handle_errors(processes, message):
    for process in processes:
        process.join()
    print(message)
    exit()


def save_tiled_lmdb(slides_list, num_ps, write_batch_size, out_dir, slides_dir, masks_dir, tile_size,
                    overlap, thres, dw_rate, verbose):

    slides_to_process = []
    env_tiles = lmdb.open(f"{out_dir}/tiles", map_size=6e+13)
    env_label_masks = lmdb.open(f"{out_dir}/label_masks", map_size=6e+12)
    env_tissue_masks = lmdb.open(f"{out_dir}/tissue_masks", map_size=6e+12)
    env_locations = lmdb.open(f"{out_dir}/locations", map_size=6e+11)

    with env_locations.begin(write=False) as txn:
        for slide_name in slides_list:
            if txn.get(slide_name.encode()) is None:
                slides_to_process.append(slide_name)
    # slides_to_process = slides_to_process[:5]
    print("Total %d slides to process" % len(slides_to_process))
    batch_size = len(slides_to_process) // num_ps
    # Spawn multiple processes to extract tiles: (each handle a portion of data).
    # If any tiled slide becomes available, the main p
    # process will get it from the queue and write to dataset.
    reader_processes = []
    pqueue = Queue()
    start_idx = 0
    for i in range(num_ps-1):
        end_idx = start_idx + batch_size
        reader_p = Process(target=generate_helper, args=(pqueue, slides_dir, masks_dir, tile_size,
                                                         overlap, thres, dw_rate, verbose,
                                                         slides_to_process[start_idx: end_idx]))
        reader_p.start()
        reader_processes.append(reader_p)
        start_idx = end_idx
    # Ensure all slides are processed by processes.
    reader_p = Process(target=generate_helper, args=(pqueue, slides_dir, masks_dir, tile_size,
                                                     overlap, thres, dw_rate, verbose,
                                                     slides_to_process[start_idx: len(slides_to_process)]))
    reader_p.start()
    reader_processes.append(reader_p)

    counter, num_done = 0, 0
    batches = []
    empty_slides = []

    while True:
        # Block if necessary until an item is available.
        data = pqueue.get()
        # Done indicates job on one process is finished.
        if data == "Done":
            num_done += 1
            print("One part is done!")
            if num_done == num_ps:
                break
        elif data["status"] == "empty":
            counter += 1
            empty_slides.append(data['slide_name'])
        else:
            batches.append(data)
        # Write a batch of data.
        if len(batches) == write_batch_size:
            try:
                counter = \
                    write_batch_data(env_tiles, env_tissue_masks, env_label_masks, env_locations, batches,
                                     len(slides_to_process), counter, verbose)
            except lmdb.KeyExistsError:
                handle_errors(reader_processes, "Key exist!")
            except lmdb.TlsFullError:
                handle_errors(reader_processes, "Thread-local storage keys full - too many environments open.")
            except lmdb.MemoryError:
                handle_errors(reader_processes, "Out of LMDB data map size.")
            except lmdb.DiskError:
                handle_errors(reader_processes, "Out of disk memory")
            except lmdb.Error:
                handle_errors(reader_processes, "Unknown LMDB write errors")
    try:
        # Write the rest data.
        if len(batches) > 0:
            counter = write_batch_data(env_tiles, env_tissue_masks, env_label_masks, env_locations, batches,
                                       len(slides_to_process), counter, verbose)
    except lmdb.KeyExistsError:
        handle_errors(reader_processes, "Key exist!")
    except lmdb.TlsFullError:
        handle_errors(reader_processes, "Thread-local storage keys full - too many environments open.")
    except lmdb.MemoryError:
        handle_errors(reader_processes, "Out of LMDB data map size.")
    except lmdb.DiskError:
        handle_errors(reader_processes, "Out of disk memory")
    except lmdb.Error:
        handle_errors(reader_processes, "Unknown LMDB write errors")

    for process in reader_processes:
        process.join()
    assert counter == len(slides_to_process), "%d processed slides, %d slides to be processed" \
                                              % (counter, len(slides_to_process))
    print("Number of empty slides: %d" % len(empty_slides))
    log_df = pd.DataFrame(columns=["slide_name"], data=empty_slides)
    log_df.to_csv(f"{out_dir}/empty_slides.csv")

    slides_tiles_mapping = dict()
    with env_locations.begin(write=False) as txn:
        for slide_name, locations in txn.cursor():
            slide_name = str(slide_name.decode('ascii'))
            slides_tiles_mapping[slide_name] = []
            locations = np.frombuffer(locations, dtype=np.int64)
            locations = locations.reshape(-1, 2)
            for loc in locations:
                slides_tiles_mapping[slide_name].append(f"{slide_name}_{loc[0]}_{loc[1]}")
    json.dump(slides_tiles_mapping, open(f"{out_dir}/slides_tiles_mappding.json", "w"))


def main(opts):
    train_df = pd.read_csv(opts.train_slide_file, index_col="image_id")
    slides_list = list(train_df.index) ## image_id
    save_tiled_lmdb(slides_list, opts.num_ps, opts.write_batch_size, opts.out_dir, opts.slides_dir, opts.masks_dir,
                    opts.tile_size, opts.overlap, opts.ts_thres, opts.dw_rate, opts.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/storage_slides/PANDA_challenge/")
    parser.add_argument("--slides_dir", default="train_images/")
    parser.add_argument('--masks_dir', default='train_label_masks/',
                        help='location for data index files')
    parser.add_argument('--train_slide_file', default="train.csv")
    parser.add_argument('--out_dir', default='/processed/')

    parser.add_argument("--tile_size", default=512, type=int)
    parser.add_argument("--overlap", default=0.25, type=float)
    parser.add_argument("--ts_thres", default=0.5, type=float)
    parser.add_argument("--dw_rate", default=1, type=int, help="Generate tiles downsampled")
    parser.add_argument("--verbose", action='store_true', help="Whether to print debug information")

    parser.add_argument("--num_ps", default=5, type=int, help="How many processor to use")
    parser.add_argument("--write_batch_size", default=10, type=int, help="Write of batch of n slides")

    args = parser.parse_args()
    args.slides_dir = f"{args.data_dir}/{args.slides_dir}/"
    args.masks_dir = f"{args.data_dir}/{args.masks_dir}/"
    args.train_slide_file = f"{args.data_dir}/{args.train_slide_file}"
    args.out_dir = f"{args.data_dir}/{args.out_dir}/"

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    main(args)


