import sys
from multiprocessing import Process, Queue
import numpy as np
import os
import argparse
import json
import pandas as pd
import time
import lmdb
from preprocessing.tile_generation import generate_grid_br
from preprocessing.normalization import reinhard_bg


def tile(img, mask, im_size, top_n):
    result = []
    shape = img.shape
    pad0,pad1 = (im_size - shape[0] % im_size)% im_size, (im_size - shape[1]% im_size) % im_size
    img = np.pad(img, [[pad0//2, pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]], constant_values=255)
    mask = np.pad(mask,[[pad0//2, pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=0)
    img = img.reshape(img.shape[0]//im_size, im_size, img.shape[1]// im_size,im_size,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,im_size,im_size,3)
    mask = mask.reshape(mask.shape[0]//im_size,im_size,mask.shape[1]//im_size,im_size,3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,im_size,im_size,3)
    if len(img) < top_n:
        mask = np.pad(mask,[[0,top_n-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,top_n-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:top_n]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'mask':mask[i], 'idx':i})
    return result


def generate_helper(pqueue, slides_dir, masks_dir, tile_size, overlap, thres, dw_rate, top_n, verbose, slides_to_process):
    if verbose:
        print("Queue len: %d" % pqueue.qsize())
    tile_normalizer = reinhard_bg.ReinhardNormalizer()
    # use the pre-computed LAB mean and std values
    tile_normalizer.fit(None)
    counter = 0
    for slide_name in slides_to_process:
        tile_generator = generate_grid_br.TileGeneratorGridBr(slides_dir, f"{slide_name}.tiff", masks_dir,
                                                              verbose=verbose)
        orig_tiles, norm_tiles, locations, _, label_masks \
            = tile_generator.extract_top_tiles_save(tile_size, overlap, thres, dw_rate, top_n, normalizer=None)
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
            "orig_tiles": orig_tiles,
            "label_masks": label_masks,
            "locations": locations,
            "status": "normal"
        }
        pqueue.put(data)
        counter += 1
        print("Put tiled slide [%s] on to queue: [%d]/[%d]" % (slide_name, counter, len(slides_to_process)))
    pqueue.put('Done')


def write_batch_data(env_tiles, env_orig_tiles, env_label_masks, env_locations, batch_data, tot_len, start_counter, verbose):
    end_counter = start_counter + len(batch_data)
    with env_tiles.begin(write=True) as txn_tiles, env_orig_tiles.begin(write=True) as txn_orig, \
            env_label_masks.begin(write=True) as txn_labels, env_locations.begin(write=True) as txn_locs:
        while len(batch_data) > 0:
            data = batch_data.pop()
            write_start = time.time()
            slide_name = data['slide_name']
            txn_tiles.put(str(slide_name).encode(), (data['norm_tiles'].astype(np.uint8)).tobytes())
            txn_orig.put(str(slide_name).encode(), (data['orig_tiles'].astype(np.uint8)).tobytes())
            txn_locs.put(str(slide_name).encode(), data['locations'].astype(np.int64).tobytes())
            if data['label_masks'] is None:
                tempt = None
            else:
                txn_labels.put(str(slide_name).encode(), data['label_masks'].astype(np.uint8).tobytes())
    print("Finish writing [%d]/[%d], time: %f" % (end_counter, tot_len, time.time() - write_start))
    return end_counter


def handle_errors(processes, message):
    for process in processes:
        process.join()
    print(message)
    exit()


def save_tiled_lmdb(slides_list, num_ps, write_batch_size, out_dir, slides_dir, masks_dir, tile_size,
                    overlap, thres, dw_rate, top_n, verbose):

    slides_to_process = []
    env_tiles = lmdb.open(f"{out_dir}/tiles", map_size=6e+12)
    env_orig_tiles = lmdb.open(f"{out_dir}/orig_tiles", map_size=6e+12)
    env_label_masks = lmdb.open(f"{out_dir}/label_masks", map_size=6e+12)
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
                                                         overlap, thres, dw_rate, top_n, verbose,
                                                         slides_to_process[start_idx: end_idx]))
        reader_p.start()
        reader_processes.append(reader_p)
        start_idx = end_idx
    # Ensure all slides are processed by processes.
    reader_p = Process(target=generate_helper, args=(pqueue, slides_dir, masks_dir, tile_size,
                                                     overlap, thres, dw_rate, top_n, verbose,
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
                    write_batch_data(env_tiles, env_orig_tiles, env_label_masks, env_locations, batches,
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
            counter = write_batch_data(env_tiles, env_orig_tiles, env_label_masks, env_locations, batches,
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



def main(opts):
    train_df = pd.read_csv(opts.train_slide_file, index_col="image_id")
    slides_list = list(train_df.index)
    save_tiled_lmdb(slides_list, opts.num_ps, opts.write_batch_size, opts.out_dir, opts.slides_dir, opts.masks_dir,
                    opts.tile_size, opts.overlap, opts.ts_thres, opts.dw_rate, opts.top_n, opts.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/storage_slides/PANDA_challenge/")
    parser.add_argument("--slides_dir", default="train_images/")
    parser.add_argument('--masks_dir', default='train_label_masks/',
                        help='location for data index files')
    parser.add_argument('--train_slide_file', default="train.csv")
    parser.add_argument('--out_dir', default='/downsample_16/')

    parser.add_argument("--tile_size", default=128, type=int)
    parser.add_argument("--overlap", default=0.125, type=float)
    parser.add_argument("--ts_thres", default=0.01, type=float)
    parser.add_argument("--dw_rate", default=16, type=int, help="Generate tiles downsampled")
    parser.add_argument("--verbose", action='store_true', help="Whether to print debug information")

    parser.add_argument("--num_ps", default=5, type=int, help="How many processor to use")
    parser.add_argument("--write_batch_size", default=10, type=int, help="Write of batch of n slides")
    parser.add_argument('--top_n', type=int, default=30)

    args = parser.parse_args()
    args.slides_dir = f"{args.data_dir}/{args.slides_dir}/"
    args.masks_dir = f"{args.data_dir}/{args.masks_dir}/"
    args.train_slide_file = f"{args.data_dir}/{args.train_slide_file}"
    args.out_dir = f"{args.data_dir}/{args.out_dir}/"

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    main(args)


