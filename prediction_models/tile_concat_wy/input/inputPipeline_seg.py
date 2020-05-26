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
                                                  collate_fn=dataloader_collte_fn, pin_memory=True)
        valloader = torch.utils.data.DataLoader(val, batch_size=self.bs, shuffle=True, num_workers=4,
                                                collate_fn=dataloader_collte_fn, pin_memory=True)
        return trainloader, valloader

def __horizontal_flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __vertical_flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def __rotate(img, rotation):
    return img.rotate(90 * rotation)

def get_transform(params, normalize=False):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __horizontal_flip(img, params['horizontal_flip'])))
    transform_list.append(transforms.Lambda(lambda img: __vertical_flip(img, params['vertical_flip'])))
    transform_list.append(transforms.Lambda(lambda img: __rotate(img, params['rotation'])))

    if normalize:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize(params['mean'],
                                                params['std'])]
    return transforms.Compose(transform_list)

class PandaPatchDatasetSeg(Dataset):
    """Panda Tile dataset. With fixed tiles for each slide."""

    def __init__(self, csv_file, image_dir, mask_dir, stats, N=12):
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
        self.mask_dir = mask_dir
        self.N = N
        self.mean, self.std = stats

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgfnames = [os.path.join(self.image_dir, self.train_csv.loc[idx, 'image_id'] + '_' + str(i) + '.png')
                     for i in range(self.N)]
        maskfnames = [os.path.join(self.mask_dir, self.train_csv.loc[idx, 'image_id'] + '_' + str(i) + '.png')
                      for i in range(self.N)]
        imgs = [self.open_image(fname, convert_mode='RGB') for fname in imgfnames]
        masks = [self.open_image(fname) for fname in maskfnames]
        # isup_grade = self.train_csv.loc[idx, 'isup_grade']
        tsfmParams = self.get_params()
        imgtsfm = get_transform(tsfmParams, True)
        masktsfm = get_transform(tsfmParams, False)

        imgs = [imgtsfm(img) for img in imgs]
        masks = [masktsfm(mask) for mask in masks]
        ## convert the output to tensor
        imgs = [torch.tensor(img) for img in imgs]
        imgs = torch.stack(imgs)
        masks = [torch.tensor(np.asarray(mask)) for mask in masks]
        masks = torch.stack(masks)
        # isup_grade = torch.tensor(isup_grade)
        sample = {'image': imgs, 'seg_mask': masks}
        return sample

    def open_image(self, fn, convert_mode=None, after_open=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
            x = Image.open(fn)
            if convert_mode:
                x = x.convert(convert_mode)
        if after_open:
            x = after_open(x)
        return x

    def get_params(self):
        horizontal_flip = np.random.random() > 0.5
        vertical_flip = np.random.random() > 0.5
        rotation = np.random.randint(0, 4)
        return {'horizontal_flip': horizontal_flip, 'vertical_flip': vertical_flip, 'rotation': rotation,
                'mean': self.mean, 'std': self.std}

class PandaPatchDatasetSegInfer(Dataset):
    """Panda Tile dataset. With fixed tiles for each slide."""

    def __init__(self, csv_file, image_dir, stats, N=12):
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
        self.N = N
        self.mean, self.std = stats

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.train_csv.image_id[idx]
        imgfnames = [os.path.join(self.image_dir, self.train_csv.loc[idx, 'image_id'] + '_' + str(i) + '.png')
                     for i in range(self.N)]
        imgs = [self.open_image(fname, convert_mode='RGB') for fname in imgfnames]
        # isup_grade = self.train_csv.loc[idx, 'isup_grade']
        tsfmParams = self.get_params()
        imgtsfm = get_transform(tsfmParams, True)

        imgs = [imgtsfm(img) for img in imgs]
        ## convert the output to tensor
        imgs = [torch.tensor(img) for img in imgs]
        imgs = torch.stack(imgs)
        # isup_grade = torch.tensor(isup_grade)
        sample = {'image': imgs, 'name':name}
        return sample

    def open_image(self, fn, convert_mode=None, after_open=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
            x = Image.open(fn)
            if convert_mode:
                x = x.convert(convert_mode)
        if after_open:
            x = after_open(x)
        return x

    def get_params(self):
        horizontal_flip = False
        vertical_flip = False
        rotation = 0
        return {'horizontal_flip': horizontal_flip, 'vertical_flip': vertical_flip, 'rotation': rotation,
                'mean': self.mean, 'std': self.std}

def dataloader_collte_fn(batch):
    imgs = [item['image'] for item in batch]
    imgs = torch.stack(imgs)
    target = [item['seg_mask'] for item in batch]
    target = torch.stack(target)
    return [imgs, target]

def dataloader_collte_fn_infer(batch):
    imgs = [item['image'] for item in batch]
    imgs = torch.stack(imgs)
    names = [item['name'] for item in batch]
    return [imgs, names]

if __name__ == "__main__":
    ## input files and folders
    nfolds = 5
    bs = 4
    csv_file = './panda-16x128x128-tiles-data/{}_fold_train.csv'.format(nfolds)
    image_dir = './panda-16x128x128-tiles-data/train/'
    mask_dir = './panda-16x128x128-tiles-data/masks/'
    ## image statistics
    mean = torch.tensor([0.90949707, 0.8188697, 0.87795304])
    std = torch.tensor([0.36357649, 0.49984502, 0.40477625])
    ## dataset, can fetch data by dataset[idx]
    dataset = PandaPatchDatasetSeg(csv_file, image_dir, mask_dir, [mean, std])
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