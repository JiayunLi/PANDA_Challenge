import os, sys
sys.path.append('../')
import cv2
import skimage.io
from tqdm import tqdm
import zipfile
import numpy as np
from utiles import utils
from spine import spine, tile, tile_rect, remove_pen_marks, tile_img, tile_rect_img

def write_2_zip(Source_Folder, Des_File, names, markers, sz = 256, N = 36):
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
    kwargs = {'step_size': 5,
              'h_step_size': 0.15,
              'patch_size': 33,
              'slide_thresh': 0.1,
              'overlap_thresh': 0.5,
              'min_size': 1,
              'iou_cover_thresh':0.84,
              'low_tile_mode': 'random'}
    ratio = []
    tile_number = []
    with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out, \
            zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
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
            if name in markers:
                img0, _, _ = remove_pen_marks(img0, scale=8)
            result = spine(img0, **kwargs)
            ra = np.sum(np.multiply((result['patch_mask'] > 0).astype('int'), result['mask'])) / np.sum(result['mask'])

            img = biopsy[1]
            ## tile the img and mask to N patches with size (sz,sz,3)
            if ra < kwargs['iou_cover_thresh'] or len(result['tile_location']) < N:
                tiles, ra, _ = tile_rect(img, mask, result['mask'], sz=sz,
                                  N = N, overlap_ratio = 0.8, mode = kwargs['low_tile_mode'])
            else:
                tiles = tile(img, mask, result['tile_location'], result['IOU'], sz=sz, N=N)

            ratio.append(ra)
            tile_number.append(len(result['tile_location']))
            for idx, t in enumerate(tiles):
                img, mask = t['img'], t['mask']
                x_tot.append((img / 255.0).reshape(-1, 3).mean(0))  ## append channel mean
                x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))
                # if read with PIL RGB turns into BGR
                img = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr('{0:s}_{1:d}.png'.format(name, idx), img)
                mask = cv2.imencode('.png', mask[:, :, 0])[1]
                mask_out.writestr('{0:s}_{1:d}.png'.format(name, idx), mask)

    # image stats
    # print(np.array(x_tot).shape) ## (168256, 3)
    img_avr = np.array(x_tot).mean(0)
    # print(np.array(x_tot).shape) ## (168256, 3)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)  ## variance = sqrt(E(X^2) - E(X)^2)
    img_std = np.sqrt(img_std)
    print('mean:', img_avr, ', std:', img_std, ', ratio:', np.mean(ratio), ', tile_number:', np.mean(tile_number))
    return (img_avr, img_std, ratio, tile_number)

def write_2_zip_img(Source_Folder, Des_File, names, markers, sz = 128, N = 16):
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
    kwargs = {'step_size': 5,
              'h_step_size': 0.15,
              'patch_size': 33,
              'slide_thresh': 0.1,
              'overlap_thresh': 0.5,
              'min_size': 1,
              'iou_cover_thresh': 0.84,
              'low_tile_mode': 'random'}
    ratio = []
    tile_number = []
    with zipfile.ZipFile(Des_File, 'w') as img_out:
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
            if name in markers:
                img0, _, _ = remove_pen_marks(img0, scale=8)
            result = spine(img0, **kwargs)
            ra = np.sum(np.multiply((result['patch_mask'] > 0).astype('int'), result['mask'])) / np.sum(result['mask'])

            img = biopsy[1]

            ## tile the img and mask to N patches with size (sz,sz,3)
            if ra < kwargs['iou_cover_thresh'] or len(result['tile_location']) < N:
                tiles, ra, _ = tile_rect_img(img,result['mask'], sz=sz,
                                         N=N, overlap_ratio=0.8, mode=kwargs['low_tile_mode'])
            else:
                tiles = tile_img(img, result['tile_location'], result['IOU'], sz=sz, N=N)

            ratio.append(ra)
            tile_number.append(len(result['tile_location']))

            for idx, t in enumerate(tiles):
                img = t['img']
                x_tot.append((img / 255.0).reshape(-1, 3).mean(0))  ## append channel mean
                x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))
                # if read with PIL RGB turns into BGR
                img = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr('{0:s}_{1:d}.png'.format(name, idx), img)
    # image stats
    # print(np.array(x_tot).shape) ## (168256, 3)
    img_avr = np.array(x_tot).mean(0)
    # print(np.array(x_tot).shape) ## (168256, 3)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)  ## variance = sqrt(E(X^2) - E(X)^2)
    img_std = np.sqrt(img_std)
    print('mean:', img_avr, ', std:', img_std, ', ratio:', np.mean(ratio), ', tile_number:', np.mean(tile_number))
    return (img_avr, img_std, ratio, tile_number)

if __name__ == "__main__":
    """Define Your Input"""
    TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'  ## train image folder
    MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'  ## train mask folder
    MARKER = "../input/prostate-cancer-grade-assessment/marker_images/"
    OUT_TRAIN = '../input/panda-36x256x256-tiles-data-spine/train2.zip'  ## output image folder
    OUT_MASKS = '../input/panda-36x256x256-tiles-data-spine/masks2.zip'  ## ouput label folder
    utils.check_folder_exists(os.path.dirname(OUT_TRAIN))
    utils.check_folder_exists(os.path.dirname(OUT_MASKS))
    sz = 256 ## image patch size
    N = 36 ## how many patches selected from each slide
    img_ids = [name[:-10] for name in os.listdir(MASKS)][5000:7500]
    pen_marked_images = [name[:-4] for name in os.listdir(MARKER)]
    print(len(img_ids))  ## only images that have masks
    """Process Image"""
    Source_Folder = [TRAIN, MASKS]
    Des_File = [OUT_TRAIN, OUT_MASKS]
    # Source_Folder = TRAIN
    # Des_File = OUT_TRAIN
    mean, std, ratio, tile_number = write_2_zip(Source_Folder, Des_File, img_ids, pen_marked_images, sz, N)