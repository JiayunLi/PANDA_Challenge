import os, sys
sys.path.append('../')
import cv2
import skimage.io
from tqdm import tqdm
import zipfile
import pickle
import numpy as np
from utiles import utils
import argparse
from PIL import Image
from new_tile import compute_coords, tile, tile_img
from spine import remove_pen_marks

def write_2_file(Source_Folder, Des_File, names, markers, sz = 256, N = 36):
    """
    Extract patches from orginal images and save them to des file.
    :param Source_Folder: list, contains the original image and mask folder
    :param Des_File: list, contains the final image and mask path for zip file
    :param names: list, contain the id for images needed to be processed
    :param sz: image patch size
    :param N: how many patches selected from each slide
    :return:
    """
    TRAIN, MASKS = Source_Folder
    OUT_TRAIN, OUT_MASKS = Des_File
    ## x_tot: [np.array(r_mean,g_mean,b_mean), np.array(r_mean,g_mean,b_mean),....]
    ## x2_tot: [np.array(r^2_mean,g^2_mean,b_mean), np.array(r^2_mean,g^2_mean,b^2_mean),....]
    x_tot, x2_tot = [], []
    tile_number = []
    tile_location = {}

    for name in tqdm(names):
        if os.path.exists(os.path.join(os.path.dirname(OUT_TRAIN), "train/{0:s}_0.png".format(name))):
            print("Skip {}.".format(name))
            continue
        ## read the image and label with the lowest res by [-1]
        try:
            biopsy = skimage.io.MultiImage(os.path.join(TRAIN, name + '.tiff'))
            mask = skimage.io.MultiImage(os.path.join(MASKS, name + '_mask.tiff'))[1]
        except:
            continue

        img0 = cv2.resize(biopsy[-1], (int(biopsy[-1].shape[1] / 2), int(biopsy[-1].shape[0] / 2)))
        img = biopsy[1]
        if name in markers:
            img0, _, _ = remove_pen_marks(img0, scale=8)
            img, _, _ = remove_pen_marks(img, scale=1)

        coords = compute_coords(img0,
                                patch_size=sz//8,
                                precompute=True,
                                min_patch_info=0.35,
                                min_axis_info=0.35,
                                min_consec_axis_info=0.35,
                                min_decimal_keep=0.7)
        coords = coords * 8
        tiles = tile(img, mask, coords, sz=sz)
        tile_number.append(len(coords))
        loc = []
        for idx, t in enumerate(tiles):
            img, mask, t_loc = t['img'], t['mask'], t['location']
            x_tot.append((img / 255.0).reshape(-1, 3).mean(0))  ## append channel mean
            x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))
            PILIMG = Image.fromarray(img)
            PILIMG.save(os.path.join(OUT_TRAIN, '{0:s}_{1:d}.png'.format(name, idx)))
            PILMASK = Image.fromarray(mask)
            PILMASK.save(os.path.join(OUT_MASKS, '{0:s}_{1:d}.png'.format(name, idx)))
            loc.append(t_loc)
        tile_location[name] = loc
    # image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)  ## variance = sqrt(E(X^2) - E(X)^2)
    img_std = np.sqrt(img_std)
    print('mean:', img_avr, ', std:', img_std, 'tile_number:', np.mean(tile_number))
    return (img_avr, img_std, tile_number, tile_location)

def write_2_file_img(Source_Folder, Des_File, names, markers, sz = 256):
    """
    Extract patches from orginal images and save them to des file.
    :param Source_Folder: list, contains the original image and mask folder
    :param Des_File: list, contains the final image and mask path for zip file
    :param names: list, contain the id for images needed to be processed
    :param sz: image patch size
    :param N: how many patches selected from each slide
    :return:
    """
    ## x_tot: [np.array(r_mean,g_mean,b_mean), np.array(r_mean,g_mean,b_mean),....]
    ## x2_tot: [np.array(r^2_mean,g^2_mean,b_mean), np.array(r^2_mean,g^2_mean,b^2_mean),....]
    x_tot, x2_tot = [], []

    tile_number = []
    tile_location = {}
    for name in tqdm(names):
        if os.path.exists(os.path.join(os.path.dirname(Des_File), "train/{0:s}_0.png".format(name))):
            print("Skip {}.".format(name))
            continue
        ## read the image and label with the lowest res by [-1]
        try:
            biopsy = skimage.io.MultiImage(os.path.join(Source_Folder, name + '.tiff'))
        except:
            continue

        img0 = cv2.resize(biopsy[-1], (int(biopsy[-1].shape[1] / 2), int(biopsy[-1].shape[0] / 2)))
        img = biopsy[1]
        if name in markers:
            img0, _, _ = remove_pen_marks(img0, scale=8)
            img, _, _ = remove_pen_marks(img, scale=1)

        coords = compute_coords(img0,
                                patch_size=sz // 8,
                                precompute=True,
                                min_patch_info=0.35,
                                min_axis_info=0.35,
                                min_consec_axis_info=0.35,
                                min_decimal_keep=0.7)
        coords = coords * 8
        tiles = tile_img(img, coords, sz=sz)
        tile_number.append(len(coords))
        loc = []
        for idx, t in enumerate(tiles):
            img, t_loc = t['img'],t['location']
            x_tot.append((img / 255.0).reshape(-1, 3).mean(0))  ## append channel mean
            x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))
            PILIMG = Image.fromarray(img)
            PILIMG.save(os.path.join(Des_File, '{0:s}_{1:d}.png'.format(name, idx)))
            loc.append(t_loc)
        tile_location[name] = loc
    # image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)  ## variance = sqrt(E(X^2) - E(X)^2)
    img_std = np.sqrt(img_std)
    print('mean:', img_avr, ', std:', img_std, 'tile_number:', np.mean(tile_number))
    return (img_avr, img_std, tile_number, tile_location)

if __name__ == "__main__":
    """Define Your Input"""
    parser = argparse.ArgumentParser(description='Optional arguments')
    parser.add_argument('--process_num', type = int, default = 0, help = 'An optional integer for batch to process')
    args = parser.parse_args()
    process_num = args.process_num
    TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'  ## train image folder
    MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'  ## train mask folder
    MARKER = "../input/prostate-cancer-grade-assessment/marker_images/"
    OUT_TRAIN = '../input/panda-36x256x256-tiles-data-opt/train/'  ## output image folder
    OUT_MASKS = '../input/panda-36x256x256-tiles-data-opt/masks/'  ## ouput label folder
    utils.check_folder_exists(os.path.dirname(OUT_TRAIN))
    utils.check_folder_exists(os.path.dirname(OUT_MASKS))
    sz = 256 ## image patch size
    N = 36 ## how many patches selected from each slide
    step_size = 20000
    img_ids = [name[:-10] for name in os.listdir(MASKS)]
    img_ids = img_ids[step_size*process_num:min(len(img_ids), step_size*(process_num + 1))]
    # img_ids = img_ids[1020:]
    with open('slide_has_less_tiles.pkl', 'rb') as f:
        slide_has_less_tiles = pickle.load(f)
    pen_marked_images = [name[:-4] for name in os.listdir(MARKER)]
    temp = []
    for i in img_ids:
        if i not in slide_has_less_tiles:
            temp.append(i)
    img_ids = temp
    print(len(img_ids))  ## only images that have masks
    """Process Image"""
    Source_Folder = [TRAIN, MASKS]
    Des_File = [OUT_TRAIN, OUT_MASKS]
    # Source_Folder = TRAIN
    # Des_File = OUT_TRAIN
    mean, std, tile_number, tile_location = write_2_file(Source_Folder, Des_File, img_ids, pen_marked_images, sz, N)
    # save tile locations:
    with open(os.path.join(os.path.dirname(OUT_MASKS), 'tile_loc_{}.pkl'.format(process_num)), 'wb') as f:
        pickle.dump(tile_location, f)