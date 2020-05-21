import os
import cv2
import skimage.io
from tqdm import tqdm
import zipfile
import numpy as np

def tile(img, mask, sz=128, N=16):
    '''
    Tile img and mask into N patches with size (sz, sz, 3)
    img: (h,w,3)
    mask: (h,w,3)
    '''
    result = []
    shape = img.shape
    ## calculate padding size so the dimesion is a multiple of sz
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    ## pad image and msk
    img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], mode='constant',
                 constant_values=255)  # img (h',w',3)
    mask = np.pad(mask, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], mode='constant',
                  constant_values=0)  # mask (h',w',3)

    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)  # img (h'/128, 128, w'/128, 128, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)  # img (h'*w'/(128**2), 128, 128, 3)
    mask = mask.reshape(mask.shape[0] // sz, sz, mask.shape[1] // sz, sz, 3)  # mask (h'/128, 128, w'/128, 128, 3)
    mask = mask.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)  # mask (h'*w'/(128**2), 128, 128, 3)

    ## if number of pathces cropped from the image is smaller than N, padding more patches with constant
    if len(img) < N:
        mask = np.pad(mask, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], mode='constant', constant_values=0)
        img = np.pad(img, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], mode='constant', constant_values=255)
    ## sort the patches by pixel value from smallest to largest
    ## pick up the first N smallest patches, thoses are the patches that most likely contains tissues
    ## as white space (255, 255, 255) contains the largest pixel value
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img': img[i], 'mask': mask[i], 'idx': i})
    ## result: list [{'img': img[0], 'mask': mask[0], 'idx': 0},{'img': img[1], 'mask': mask[1], 'idx': 1},...]
    return result

def write_2_zip(Source_Folder, Des_File, names, sz = 128, N = 16):
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
    with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out, \
            zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
        for name in tqdm(names):
            ## read the image and label with the lowest res by [-1]
            img = skimage.io.MultiImage(os.path.join(TRAIN, name + '.tiff'))[-1]
            mask = skimage.io.MultiImage(os.path.join(MASKS, name + '_mask.tiff'))[-1]
            ## tile the img and mask to N patches with size (sz,sz,3)
            tiles = tile(img, mask, sz, N)
            for t in tiles:
                img, mask, idx = t['img'], t['mask'], t['idx']
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
    print('mean:', img_avr, ', std:', img_std)
    return (img_avr, img_std)

if __name__ == "__main__":
    """Define Your Input"""
    TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'  ## train image folder
    MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'  ## train mask folder
    OUT_TRAIN = '../input/panda-16x128x128-tiles-data/train.zip'  ## output image folder
    OUT_MASKS = '../input/panda-16x128x128-tiles-data/masks.zip'  ## ouput label folder
    sz = 128 ## image patch size
    N = 16 ## how many patches selected from each slide
    names = [name[:-10] for name in os.listdir(MASKS)]
    print(len(names))  ## only images that have masks

    """Process Image"""
    Source_Folder = [TRAIN, MASKS]
    Des_File = [OUT_TRAIN, OUT_MASKS]
    mean, std = write_2_zip(Source_Folder, Des_File, names, sz, N)