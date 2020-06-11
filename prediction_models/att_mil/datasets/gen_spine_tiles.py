import skimage.io
from PIL import Image
import pandas as pd
import numpy as np
import time
from prediction_models.att_mil.utils import reinhard_fast
import tqdm
import os
import cv2
from prediction_models.att_mil.datasets import spine
from skimage import morphology as skmp


def detect_spine_tile(kwargs, slide_id, slides_dir,  has_marker=False, sz=256, N=36):

    slide = skimage.io.MultiImage(os.path.join(slides_dir, slide_id + '.tiff'))
    mask = None
    # 1/2 of lowest resolution
    img0 = cv2.resize(slide[-1], (int(slide[-1].shape[1] / 2), int(slide[-1].shape[0] / 2)))
    if has_marker:
        # TODO: Use tissue detection method
        img0, _, _ = spine.remove_pen_marks(img0, scale=8)

    result = spine.spine(img0, **kwargs)
    # ra = np.sum(np.multiply((result['patch_mask'] > 0).astype('int'), result['mask'])) / np.sum(result['mask'])
    # ratio.append(ra)
    # tile_number.append(len(result['tile_location']))
    img = slide[1]
    # tile the img and mask to N patches with size (sz,sz,3)
    tiles = spine.tile(img, mask, result['tile_location'], result['IOU'], sz=sz, N=N)
    return tiles


def get_saved_tiles_locs(slide_df, slides_dir, out_dir, marker_slides):
    kwargs = {'step_size': 1,
              'h_step_size': 0.2,
              'patch_size': 32,
              'slide_thresh': 0.6,
              'overlap_thresh': 0.6,
              'min_size': 40}
    x_tot, x2_tot = [], []
    tile_locations = dict()
    for i in range(len(slide_df)):
        slide_id = slide_df.iloc[i].image_id
        if slide_id in marker_slides:
            has_marker = True
        else:
            has_marker = False
        tiles = detect_spine_tile(kwargs, slide_id, slides_dir,  has_marker=has_marker, sz=256, N=36)
        for idx, t in enumerate(tiles):
            img, mask = t['img'], t['mask']
            x_tot.append((img / 255.0).reshape(-1, 3).mean(0))  ## append channel mean
            x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))
            cur_loc = t['orig_location']
            tile_name = '{0:s}_{1:d}'.format(slide_id, idx)
            tile_locations[tile_name] = cur_loc
    np.save(f"{out_dir}/spine_tile_locations.npy", tile_locations)
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)  # variance = sqrt(E(X^2) - E(X)^2)
    img_std = np.sqrt(img_std)
    print('mean:', img_avr, ', std:', img_std)


if __name__ == "__main__":
    import pandas as pd
    trainval_df = pd.read_csv("./info/16_128_128/4_fold_train.csv")
    tile_sz = 256 # image patch size
    top_n = 64 # how many patches selected from each slide
    trainval_slides_dir = "/slides_data/"
    pen_marker_slides = set(np.load("./info/marker_images.npy", allow_pickle=True).tolist())
    save_dir = './info/'
    get_saved_tiles_locs(trainval_df, trainval_slides_dir, save_dir, pen_marker_slides)
