import os,sys
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from collections import OrderedDict
import skimage.io
import cv2
from new_tile import compute_coords, tile_img
import time

class PandaPatchDatasetInfer(Dataset):
    """
        gls2isu = {"0+0":0,'negative':0,'3+3':1,'3+4':2,'4+3':3,'4+4':4,'3+5':4,'5+3':4,'4+5':5,'5+4':5,'5+5':5}
        """
    gls = {"0+0": [0, 0], 'negative': [0, 0], '3+3': [1, 1], '3+4': [1, 2], '4+3': [2, 1], '4+4': [2, 2],
           '3+5': [1, 3], '5+3': [3, 1], '4+5': [2, 3], '5+4': [3, 2], '5+5': [3, 3]}
    """Panda Tile dataset. With fixed tiles for each slide."""

    def __init__(self, csv_file, image_dir, image_size, N=36, transform=None, rand=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            N (interger): Number of tiles selected for each slide.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_csv = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        self.N = N
        self.rand = rand

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx, mode = 'mean_pix'):
        result = OrderedDict()
        kwargs = {'step_size': 5,
                  'h_step_size': 0.15,
                  'patch_size': 33,
                  'slide_thresh': 0.1,
                  'overlap_thresh': 0.5,
                  'min_size': 1,
                  'iou_cover_thresh': 0.84,
                  'low_tile_mode': 'random'}
        name = self.train_csv.image_id[idx]
        biopsy = skimage.io.MultiImage(os.path.join(self.image_dir, name + '.tiff'))
        img0 = cv2.resize(biopsy[-1], (int(biopsy[-1].shape[1] / 2), int(biopsy[-1].shape[0] / 2)))
        img = biopsy[1]

        coords = compute_coords(img0,
                                patch_size=self.image_size // 8,
                                precompute=True,
                                min_patch_info=0.35,
                                min_axis_info=0.35,
                                min_consec_axis_info=0.35,
                                min_decimal_keep=0.7)
        coords = coords * 8
        tiles = tile_img(img, coords, sz=self.image_size)
        tile_number = len(tiles)
        if tile_number == self.N:
            idxes = [i for i in range(self.N)]
        elif tile_number > self.N:
            pix_intensity = []
            for i in range(tile_number):
                pix_intensity.append(tiles[i][mode])
            if mode == "mean_pix":
                idxes = list(np.argsort(pix_intensity)[:self.N])
            else:
                idxes = list(np.argsort(pix_intensity)[::-1][:self.N])
            # idxes = np.random.choice(list(range(tile_number)), self.N, replace=False)
        else:
            idxes = list(range(tile_number))
            pix_intensity = []
            for i in range(tile_number):
                pix_intensity.append(tiles[i][mode])
            if mode == "mean_pix":
                idxes += list(np.argsort(pix_intensity)[:self.N - tile_number])
            else:
                idxes += list(np.argsort(pix_intensity)[::-1][:self.N - tile_number])
            # idxes += list(np.random.choice(list(range(tile_number)), self.N - tile_number, replace=True))

        imgs = []
        for i in idxes:
            img = tiles[i]['img']
            # img = Image.fromarray(img).convert('RGB')
            # img = np.asarray(img)
            imgs.append({'img': img, 'idx': i})

        if self.rand:  ## random shuffle the order of tiles
            idxes = np.random.choice(list(range(self.N)), self.N, replace=False)
        else:
            idxes = list(range(self.N))

        n_row_tiles = int(np.sqrt(self.N))

        images = np.zeros((self.image_size * n_row_tiles, self.image_size * n_row_tiles, 3)).astype(np.uint8)
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
                if len(imgs) > idxes[i]:
                    this_img = imgs[idxes[i]]['img'].astype(np.uint8)
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                # this_img = 255 - this_img  ## todo: see how this trik plays out
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']
                h1 = h * self.image_size
                w1 = w * self.image_size
                images[h1:h1 + self.image_size, w1:w1 + self.image_size] = this_img

        if self.transform is not None:
            images = self.transform(image=images)['image']
        images = images.astype(np.float32)
        images /= 255.0
        mean = np.asarray([0.79667089, 0.59347025, 0.75775308])
        std = np.asarray([0.07021654, 0.13918451, 0.08442586])
        images = (images - mean) / (std)  ## normalize the image
        images = images.transpose(2, 0, 1)
        datacenter = self.train_csv.loc[idx, 'data_provider']
        result['img'] = torch.tensor(images)
        result['datacenter'] = datacenter
        result['name'] = name
        return result