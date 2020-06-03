import lmdb
import numpy as np
from PIL import Image
import torch
import gc


def decode_buffer(buff, data_shape, data_type):
    buff = np.frombuffer(buff, dtype=data_type)
    buff = buff.reshape(data_shape)
    return buff


def read_lmdb(lmdb_dir, data_shape, keys, env=None, data_type=np.uint8):
    if not env:
        env = lmdb.open(lmdb_dir, max_readers=3, readonly=True, lock=False,
                        readahead=False, meminit=False)
    results = []
    with env.begin(write=False) as txn:
        for key in keys:
            buffer = txn.get(str(key).encode())
            buffer = decode_buffer(buffer, data_shape, data_type)
        results.append(buffer)
    return np.asarray(results)


def read_lmdb_tiles_tensor(lmdb_dir, data_shape, keys, transform, out_im_size,
                           tiles_df=None, env=None, data_type=np.uint8):
    if not env:
        env = lmdb.open(lmdb_dir, max_readers=3, readonly=True, lock=False,
                        readahead=False, meminit=False)
    # data_shape: im_size * im_size * channel
    # Transformed: n_batch * channel * out_im_size * out_im_size
    results = torch.FloatTensor(len(keys), *out_im_size)
    labels = []
    with env.begin(write=False) as txn:
        for i, key in enumerate(keys):
            buffer = txn.get(str(key).encode())
            buffer = decode_buffer(buffer, data_shape, data_type)
            buffer = Image.fromarray(buffer)
            if transform:
                buffer = transform(buffer)
            results[i, :, :, :] = buffer
            if tiles_df is None:
                continue
            if key not in tiles_df.index:
                continue
            labels.append(int(tiles_df.loc[key].tile_label))
            gc.collect()
    gc.collect()
    return results, labels


def read_lmdb_slide_tensor(env, data_shape, key, transform, out_im_size, data_type=np.uint8):

    with env.begin(write=False) as txn:
        buffer = txn.get(str(key).encode())
        if buffer == None:
            print(key)
        buffer = decode_buffer(buffer, data_shape, data_type)
    results = torch.FloatTensor(len(buffer), *out_im_size)
    for i in range(len(buffer)):
        tile = Image.fromarray(buffer[i, :, :, :])
        if transform:
            tile = transform(tile)
        results[i, :, :, :] = tile
    return results
