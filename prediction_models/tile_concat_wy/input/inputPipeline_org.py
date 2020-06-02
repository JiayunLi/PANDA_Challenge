import os
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations
import skimage.io

class crossValInx(object):
    def __init__(self, csv_file):
        self.crossVal_csv = pd.read_csv(csv_file)

    def __call__(self, fold = 0):
        val_idx = self.crossVal_csv.index[self.crossVal_csv['split'] == fold].tolist()
        train_idx = list(set([x for x in range(len(self.crossVal_csv))]) - set(val_idx))
        return train_idx, val_idx

class crossValDataloader(object):
    def __init__(self, csv_file, dataset, bs = 4):
        self.inx = crossValInx(csv_file)
        self.dataset = dataset
        self.bs = bs

    def __call__(self, fold = 0):
        train_idx, val_idx = self.inx(fold)
        train = torch.utils.data.Subset(self.dataset, train_idx)
        val = torch.utils.data.Subset(self.dataset, val_idx)
        trainloader = torch.utils.data.DataLoader(train, batch_size=self.bs, shuffle=True, num_workers=4,
                                                  collate_fn=None, pin_memory=True,
                                                  drop_last=True)
        valloader = torch.utils.data.DataLoader(val, batch_size=self.bs, shuffle=True, num_workers=4,
                                                collate_fn=None, pin_memory=True,
                                                drop_last=True)
        return trainloader, valloader

class PandaPatchDataset(Dataset):
    """Panda Tile dataset. With fixed tiles for each slide."""
    def __init__(self, csv_file, image_dir, image_size, N = 12, transform=None, rand=False):
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

    def __getitem__(self, idx):
        img_id = self.train_csv.loc[idx, 'image_id']
        fname = os.path.join(self.image_dir, img_id+'.tiff')
        image = skimage.io.MultiImage(fname)[1]
        imgs, OK = get_tiles(image, self.image_size, self.N)

        if self.rand: ## random shuffle the order of tiles
            idxes = np.random.choice(list(range(self.N)), self.N, replace=False)
        else:
            idxes = list(range(self.N))

        n_row_tiles = int(np.sqrt(self.N))

        images = np.zeros((self.image_size * n_row_tiles, self.image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
                if len(imgs) > idxes[i]:
                    this_img = imgs[idxes[i]]['img']
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                this_img = 255 - this_img ## todo: see how this trik plays out
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']
                h1 = h * self.image_size
                w1 = w * self.image_size
                images[h1:h1 + self.image_size, w1:w1 + self.image_size] = this_img

        if self.transform is not None:
            images = self.transform(image=images)['image']
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)
        label = np.zeros(5).astype(np.float32)
        isup_grade = self.train_csv.loc[idx, 'isup_grade']
        label[:isup_grade] = 1.
        return torch.tensor(images), torch.tensor(label)

class PandaPatchDatasetInfer(Dataset):
    def __init__(self, csv_file, image_dir, image_size, N=12, transform=None, rand=False):
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

    def __getitem__(self, idx):
        img_id = self.train_csv.loc[idx, 'image_id']
        name = self.train_csv.image_id[idx]
        fname = os.path.join(self.image_dir, img_id + '.tiff')
        image = skimage.io.MultiImage(fname)[1]
        imgs, OK = get_tiles(image, self.image_size, self.N)

        if self.rand:  ## random shuffle the order of tiles
            idxes = np.random.choice(list(range(self.N)), self.N, replace=False)
        else:
            idxes = list(range(self.N))

        n_row_tiles = int(np.sqrt(self.N))

        images = np.zeros((self.image_size * n_row_tiles, self.image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
                if len(imgs) > idxes[i]:
                    this_img = imgs[idxes[i]]['img']
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                this_img = 255 - this_img  ## todo: see how this trik plays out
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']
                h1 = h * self.image_size
                w1 = w * self.image_size
                images[h1:h1 + self.image_size, w1:w1 + self.image_size] = this_img

        if self.transform is not None:
            images = self.transform(image=images)['image']
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)
        return torch.tensor(images), name

def get_tiles(img, tile_size, n_tiles, mode=0):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
                  mode='constant', constant_values=255)
    img3 = img2.reshape(
        img2.shape[0] // tile_size,
        tile_size,
        img2.shape[1] // tile_size,
        tile_size,
        3
    )

    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    n_tiles_with_info = (img3.reshape(img3.shape[0], -1).sum(1) < tile_size ** 2 * 3 * 255).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(img3, [[0, n_tiles - len(img3)], [0, 0], [0, 0], [0, 0]], mode='constant', constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img': img3[i], 'idx': i})
    return result, n_tiles_with_info >= n_tiles

def data_transform():
    tsfm = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
    ])
    return tsfm


if __name__ == "__main__":
    ## input files and folders
    nfolds = 5
    bs = 4
    csv_file = './panda-16x128x128-tiles-data/{}_fold_train.csv'.format(nfolds)
    image_dir = './panda-32x256x256-tiles-data/train/'
    ## image statistics
    mean = torch.tensor([0.90949707, 0.8188697, 0.87795304])
    std = torch.tensor([0.36357649, 0.49984502, 0.40477625])
    ## image transformation
    tsfm = data_transform(mean, std)
    ## dataset, can fetch data by dataset[idx]
    dataset = PandaPatchDataset(csv_file, image_dir, transform=tsfm)
    ## dataloader
    dataloader = DataLoader(dataset, batch_size=bs,
                            shuffle=True, num_workers=4, collate_fn=dataloader_collte_fn)

    ## fetch data from dataloader
    img, target = iter(dataloader).next()
    print("image size:{}, target sise:{}.".format(img.size(), target.size()))

    ## cross val dataloader
    crossValData = crossValDataloader(csv_file, dataset, bs)
    trainloader, valloader = crossValData(0)
    img, target = iter(trainloader).next()
    print("image size:{}, target sise:{}.".format(img.size(), target.size()))