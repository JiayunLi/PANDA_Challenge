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
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from collections import OrderedDict
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
        trainloader = torch.utils.data.DataLoader(train, batch_size=self.bs, shuffle=False, num_workers=4,
                                                  sampler=RandomSampler(train),collate_fn=None, pin_memory=True,
                                                  drop_last=True)
        valloader = torch.utils.data.DataLoader(val, batch_size=self.bs, shuffle=False, num_workers=4,
                                                collate_fn=None, pin_memory=True, sampler=SequentialSampler(val),
                                                drop_last=True)
        return trainloader, valloader

class PandaPatchDataset(Dataset):
    """Panda Tile dataset. With fixed tiles for each slide."""
    def __init__(self, csv_file, image_dir, image_size, N = 36, transform=None, rand=False):
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
        result = OrderedDict()
        img_id = self.train_csv.loc[idx, 'image_id']
        fnames = [os.path.join(self.image_dir, img_id + '_' + str(i) + '.png')
                  for i in range(self.N)]
        imgs = []
        for i, fname in enumerate(fnames):
            img = self.open_image(fname)
            imgs.append({'img': img, 'idx': i})

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
        # label = np.zeros(5).astype(np.float32)
        isup_grade = self.train_csv.loc[idx, 'isup_grade']
        datacenter = self.train_csv.loc[idx, 'data_provider']
        # label[:isup_grade] = 1.
        result['img'] = torch.tensor(images)
        result['isup_grade'] = torch.tensor(isup_grade)
        result['datacenter'] = datacenter
        return result

    def open_image(self, fn, convert_mode='RGB', after_open=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
            x = Image.open(fn).convert(convert_mode)
            x= np.asarray(x)
        if after_open:
            x = after_open(x)
        return x

class PandaPatchDatasetInfer(Dataset):
    def __init__(self, csv_file, image_dir, transform = None, N = 12, sz = 128):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            N (interger): Number of tiles selected for each slide.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.test_csv = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.image_id = list(self.test_csv.image_id)
        self.N = N
        self.sz = sz
        self.transform = transform

    def __len__(self):
        return len(self.test_csv)

    def __getitem__(self, idx):
        name = self.test_csv.image_id[idx]
        img = skimage.io.MultiImage(os.path.join(self.image_dir, name + '.tiff'))[1] # get the lowest resolution
        imgs, OK = self.tile_image(img, self.tile_size, self.n_tiles) ## list of tiles per slide

        if self.rand:
            idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
        else:
            idxes = list(range(self.n_tiles))

        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        ## convert the output to tensor
        # imgs = [torch.tensor(img) for img in imgs]
        imgs = torch.stack(imgs)
        return {'image': imgs, 'name': name}

def data_transform():
    tsfm = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
    ])
    return tsfm

def dataloader_collte_fn(batch):
    result = OrderedDict()
    imgs = [item['img'] for item in batch]
    imgs = torch.stack(imgs)
    target = [item['isup_grade'] for item in batch]
    target = torch.stack(target)
    datacenter = [item['datacenter'] for item in batch]
    result['img'] = imgs
    result['isup_grade'] = target
    result['datacenter'] = datacenter
    return result


if __name__ == "__main__":
    ## input files and folders
    nfolds = 5
    bs = 4
    sz = 256
    csv_file = './panda-16x128x128-tiles-data/{}_fold_train.csv'.format(nfolds)
    image_dir = './panda-32x256x256-tiles-data/train/'
    ## image transformation
    tsfm = data_transform()
    ## dataset, can fetch data by dataset[idx]
    dataset = PandaPatchDataset(csv_file, image_dir, sz, transform=tsfm, N=16, rand=True)
    ## dataloader
    dataloader = DataLoader(dataset, batch_size=bs,
                            shuffle=True, num_workers=4, collate_fn=dataloader_collte_fn)

    ## fetch data from dataloader
    data = iter(dataloader).next()
    print("image size:{}, target sise:{}.".format(data['img'].size(), data['isup_grade'].size()))

    ## cross val dataloader
    crossValData = crossValDataloader(csv_file, dataset, bs)
    trainloader, valloader = crossValData(0)
    data = iter(valloader).next()
    print("image size:{}, target sise:{}.".format(data['img'].size(), data['isup_grade'].size()))