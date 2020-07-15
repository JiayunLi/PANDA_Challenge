import numpy as np
import skimage.io
import pandas as pd
import tqdm
from prediction_models.att_mil.datasets import gen_selected_tiles


def select_sub_simple_4x4(lowest_ids, low_im_size, padded_low_shape, pad_top, pad_left, lowest_sub_size,
                          sub_size, tiles, max_per_tile, n_row=None, n_col=None, use_orig=False):
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
    if not n_row or not n_col:
        n_row, n_col = padded_low_shape[0] // low_im_size, padded_low_shape[1] // low_im_size

    locs = {"high_res": [], "low_res": []}
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
        sub_locs = []
        if use_orig:
            sub_locs.append((i, j))
        else:
            for sub_id in sub_ids:
                sub_i = i + (sub_id // n_sub_col) * lowest_sub_size - shift_size
                sub_j = j + (sub_id % n_sub_col) * lowest_sub_size - shift_size
                sub_locs.append((sub_i, sub_j))
        # locs.append([sub_i, sub_j])
        locs['high_res'].append(sub_locs)
        locs['low_res'].append((i, j))
    return locs


def select_sub_orig(lowest_ids, low_im_size, padded_low_shape, pad_top, pad_left, n_row=None, n_col=None):
    if not n_row or not n_col:
        n_row, n_col = padded_low_shape[0] // low_im_size, padded_low_shape[1] // low_im_size
    locs = {"high_res": [], "low_res": []}
    for idx, tile_id in enumerate(lowest_ids):
        if tile_id == "-1":
            continue
        sub_locs = []
        i, j = tile_id // n_col, tile_id % n_col
        i = max(i * low_im_size - pad_top, 0)
        j = max(j * low_im_size - pad_left, 0)
        sub_locs.append((i, j))
        locs['high_res'].append(sub_locs)
        locs['low_res'].append(sub_locs)
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
    cur_pad_top, cur_pad_left, _, _ = attention_selected['pad_infos'][slide_id]
    pad0, pad1 = (att_low_tile_size - lowest.shape[0] % att_low_tile_size) % att_low_tile_size, (
                  att_low_tile_size - lowest.shape[1] % att_low_tile_size) % att_low_tile_size
    padded_low_shape = (lowest.shape[0] + pad0, lowest.shape[1] + pad1)
    results = gen_selected_tiles.get_highres_tiles(orig, cur_tile_ids_fix, cur_pad_top, cur_pad_left, att_low_tile_size,
                                                   padded_low_shape, att_level, top_n=-1)
    if method == "4x4":
        selected_locs = select_sub_simple_4x4(cur_tile_ids_fix[:select_n], att_low_tile_size, padded_low_shape, cur_pad_top,
                                              cur_pad_left, select_sub_size_lowest, select_sub_size,
                                              results['tiles'], select_per_tile)
    elif method == 'orig':
        selected_locs = select_sub_orig(cur_tile_ids_fix[:select_n], att_low_tile_size, padded_low_shape, cur_pad_top,
                                        cur_pad_left)
    else:
        raise NotImplementedError(f"{method} hasn't been implemented!")
    return selected_locs


def att_select_locs(slides_dir, slides_df_loc, attention_selected_loc, att_low_tile_size, att_level, select_n,
                    select_sub_size, select_per_tile, method='4x4'):
    attention_selected = np.load(attention_selected_loc, allow_pickle=True)
    attention_selected = dict(attention_selected.tolist())
    all_slides = []
    for i in range(4):
        val_df = f"{slides_df_loc}/val_{i}.csv"
        val_df = pd.read_csv(val_df)
        for ii in range(len(val_df)):
            cur = val_df.iloc[ii].to_dict()
            all_slides.append(cur)
    all_slides = pd.DataFrame(data=all_slides, columns=list(all_slides[0].keys()))

    all_selected_locs = dict()
    print("Generate selected tile locations")

    for i in tqdm.tqdm(range(len(all_slides))):
        slide_id = str(all_slides.iloc[i].image_id)
        selected_locs = att_select_locs_helper(slides_dir, slide_id, attention_selected, att_low_tile_size, att_level,
                                               select_n, select_sub_size, select_per_tile, method)
        all_selected_locs[slide_id] = selected_locs
    all_selected_locs['low_level'] = att_level
    all_selected_locs['high_level'] = att_level - 1
    all_selected_locs['lowest_tile_size_low'] = att_low_tile_size
    all_selected_locs['lowest_tile_size_high'] = select_sub_size
    return all_selected_locs



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Attention MIL PANDA challenge')
    # File location
    parser.add_argument('--data_dir', type=str, default='/data/', help='Root directory for processed data')
    parser.add_argument('--info_dir', type=str, default='./info/16_128_128')
    parser.add_argument('--select_model', default="resnext50_3e-4_bce_256",
                        help="Use which model to generate attention map")
    parser.add_argument('--att_dir', type=str, default='./info/att_selected/',
                        help='Directory for cross validation information')
    parser.add_argument('--n_low_res_tiles', type=int, default=36)
    parser.add_argument('--low_res_fov', type=int, default=64, help='Field of view at lowest magnification')
    parser.add_argument('--select_method', default='4x4')
    parser.add_argument('--high_res_fov', type=int, default=32, help='Filed of view at lowest magnification')
    parser.add_argument('--select_per', type=float, default=0.25)
    parser.add_argument('--select_per_tile', type=int, default=1)
    opts = parser.parse_args()

    low_res_tile_size = 4 * opts.low_res_fov
    select_tot_n = int(opts.select_per * opts.n_low_res_tiles) * opts.select_per_tile
    select_locs_file_loc = f"{opts.att_dir}/{opts.select_model}_n_{opts.n_low_res_tiles}_sz_{low_res_tile_size}.npy"

    all_selected = att_select_locs(opts.data_dir, opts.info_dir,
                                   select_locs_file_loc, att_low_tile_size=opts.low_res_fov, att_level=-2,
                                   select_n=select_tot_n, select_sub_size=opts.low_res_fov // 4,  # 4*4 grid.
                                   select_per_tile=opts.select_per_tile,
                                   method=opts.select_method)
    np.save(f"{opts.att_dir}/{opts.select_model}_n_{select_tot_n}_sz_{opts.high_res_fov}_locs.npy",
            all_selected)