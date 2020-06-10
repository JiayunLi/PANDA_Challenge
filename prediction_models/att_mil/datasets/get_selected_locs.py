import numpy as np
import skimage.io
import pandas as pd
import tqdm
from prediction_models.att_mil.datasets import gen_selected_tiles


def select_sub_simple_4x4(lowest_ids, low_im_size, padded_low_shape, pad_top, pad_left, lowest_sub_size,
                          sub_size, tiles, max_per_tile):
    """
    Select lowest_ids with top attention value (50%). Divide the mid-resolution tile into 4 * 4 sub-grid and select
    two sub-tiles. Around each sub-tile, select a low_im_size // 2 region as the region to be zoomed in.
    :param lowest_ids:
    :param low_im_size:
    :param padded_low_shape:
    :param pad_top:
    :param pad_left:
    :param lowest_sub_size:
    :param sub_size:
    :param tiles:
    :param max_per_tile:
    :return:
    """

    n_row, n_col = padded_low_shape[0] // low_im_size, padded_low_shape[1] // low_im_size

    locs = []
    shift_size = lowest_sub_size // 2
    n_sub_col = low_im_size // lowest_sub_size
    for idx, tile_id in enumerate(lowest_ids):
        if tile_id == "-1":
            continue
        i, j = tile_id // n_col, tile_id % n_col
        i = max(i * low_im_size - pad_top, 0)
        j = max(j * low_im_size - pad_left, 0)
        cur_tile = tiles[idx]
        ss_tiles = cur_tile.reshape(cur_tile.shape[0] // sub_size, sub_size, cur_tile.shape[0] // sub_size, sub_size, 3)
        ss_tiles = ss_tiles.transpose(0, 2, 1, 3, 4).reshape(-1, sub_size, sub_size, 3)
        sub_ids = np.argsort(ss_tiles.reshape(ss_tiles.shape[0], -1).sum(-1))[:max_per_tile]

        for sub_id in sub_ids:
            sub_i = i + (sub_id // n_sub_col) * lowest_sub_size - shift_size
            sub_j = j + (sub_id % n_sub_col) * lowest_sub_size - shift_size
            locs.append((sub_i, sub_j))
    return locs


def att_select_locs_helper(slides_dir, slide_id, attention_selected, att_low_tile_size, att_level, select_n,
                           select_sub_size, select_per_tile, method='4x4'):
    cur_atts = attention_selected['atts'][slide_id]
    cur_tile_ids = attention_selected['tile_ids'][slide_id]
    max_atts_ids = np.argsort(cur_atts)[::-1]
    cur_tile_ids = cur_tile_ids[max_atts_ids]
    cur_tile_ids_fix = []
    for tile_id in cur_tile_ids:
        if tile_id == "-1":
            continue
        cur_tile_ids_fix.append(int(tile_id))
    orig = skimage.io.MultiImage(f"{slides_dir}/{slide_id}.tiff")
    lowest = orig[-1]
    select_sub_size_lowest = select_sub_size // gen_selected_tiles.RATE_MAP[att_level]
    cur_pad_top, cur_pad_left = attention_selected['pad_infos'][slide_id]
    pad0, pad1 = (att_low_tile_size - lowest.shape[0] % att_low_tile_size) % att_low_tile_size, (
                  att_low_tile_size - lowest.shape[1] % att_low_tile_size) % att_low_tile_size
    padded_low_shape = (lowest.shape[0] + pad0, lowest.shape[1] + pad1)
    results = gen_selected_tiles.get_highres_tiles(orig, cur_tile_ids_fix, cur_pad_top, cur_pad_left, att_low_tile_size,
                                                   padded_low_shape, att_level, top_n=-1)
    if method == "4x4":
        selected_locs = select_sub_simple_4x4(cur_tile_ids_fix[:select_n], att_low_tile_size, padded_low_shape, cur_pad_top,
                                              cur_pad_left, select_sub_size_lowest, select_sub_size,
                                              results['tiles'], select_per_tile)
    else:
        raise NotImplementedError(f"{method} hasn't been implemented!")
    return selected_locs


def att_select_locs(slides_dir, slides_df_loc, attention_selected_loc, att_low_tile_size, att_level, select_n,
                    select_sub_size, select_per_tile, method='4x4'):
    attention_selected = np.load(attention_selected_loc, allow_pickle=True)
    attention_selected = dict(attention_selected.tolist())
    slides_df = pd.read_csv(slides_df_loc)
    all_selected_locs = dict()
    print("Generate selected tile locations")
    for i in tqdm.tqdm(range(len(slides_df))):
        slide_id = str(slides_df.iloc[i].image_id)
        selected_locs = att_select_locs_helper(slides_dir, slide_id, attention_selected, att_low_tile_size, att_level,
                                               select_n, select_sub_size, select_per_tile, method)
        all_selected_locs[slide_id] = selected_locs
    return all_selected_locs



