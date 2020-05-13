import pickle
import os
from PIL import ImageFilter, Image
from prediction_models.att_mil.datasets import trainval_slides, test_slides
from preprocessing.normalization import reinhard_bg
import torch
import random
import glob
from shutil import copyfile
import torchvision.transforms as T
import gc

INTERP = 3


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def compute_meanstd(dataset_params, fold, num_workers, from_k_slides=200):
    print("Start compute mean and std")
    cur_transform = T.Compose([
        T.Resize(dataset_params.input_size, interpolation=Image.ANTIALIAS),
        T.ToTensor()
    ])
    dataset = trainval_slides.BiopsySlides(dataset_params, cur_transform, fold, split="val", phase="meanstd")
    train_mean = torch.zeros(3)
    train_std = torch.zeros(3)
    loader = \
        torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False,
                                    num_workers=num_workers, pin_memory=False)
    loader_iter = iter(loader)

    cur_num, step = 0, 0
    tot = min(len(loader), from_k_slides)
    while step < tot:
        tiles, _, _, _ = loader_iter.next()
        tiles = torch.squeeze(tiles, dim=0)
        train_mean += torch.sum(torch.mean(tiles.view(tiles.size(0), tiles.size(1), -1), dim=2), dim=0)
        train_std += torch.sum(torch.std(tiles.view(tiles.size(0), tiles.size(1), -1), dim=2), dim=0)
        cur_num += tiles.size(0)
        step += 1
        print(f"{step}/{tot}")
        gc.collect()
    train_std /= cur_num
    train_mean /= cur_num
    meanstd = {
        "mean": train_mean.cpu().numpy(),
        "std": train_std.cpu().numpy(),
    }
    print("Finish compute mean and std")
    gc.collect()
    return meanstd


def build_dataset_loader(batch_size, num_workers, dataset_params, split, phase, fold=None):
    """

    :param batch_size:
    :param num_workers:
    :param dataset_params:
    :param split: Which split of data to use
    :param phase: Whether to train/val on the split of data
    :param fold:
    :return:
    """

    if fold is not None:
        if not os.path.isdir(f"{dataset_params.exp_dir}/{fold}/"):
            os.mkdir(f"{dataset_params.exp_dir}/{fold}/")
        meanstd_file = f"{dataset_params.exp_dir}/{fold}/meanstd.pkl"
    else:
        meanstd_file = f"{dataset_params.exp_dir}/meanstd.pkl"

    if not os.path.isfile(meanstd_file):
        other_exp_dirs = glob.glob(f"{dataset_params.cache_dir}/*/")
        has_computed = False
        for other_dir in other_exp_dirs:
            exp_name = other_dir.split("/")[-2]
            if exp_name == "debug":
                continue
            if os.path.isdir(f"{other_dir}/{fold}/") and os.path.isfile(f"{other_dir}/{fold}/meanstd.pkl"):
                copyfile(f"{other_dir}/{fold}/meanstd.pkl", meanstd_file)
                has_computed = True
                break
        if not has_computed:
            meanstd = compute_meanstd(dataset_params, fold, num_workers)
            pickle.dump(meanstd, open(meanstd_file, "wb"))
    meanstd = pickle.load(open(meanstd_file, "rb"))

    # Define different transformations
    normalize = [
        T.Resize(dataset_params.input_size, interpolation=Image.ANTIALIAS),
        T.ToTensor(),
        T.Normalize(mean=meanstd['mean'], std=meanstd['std'])]
    augmentation = [
        T.RandomResizedCrop(dataset_params.input_size, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                            interpolation=INTERP),
        T.RandomApply([
            T.ColorJitter(0.2, 0.2, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        T.RandomHorizontalFlip(),
        ] + normalize

    if split == "train":
        transform = T.Compose(augmentation)
    elif split == "val" or split == "test":
        transform = T.Compose(normalize)
    else:
        raise NotImplementedError(f"Not implemented for split {split}")

    if split == "train" or split == "val":
        dataset = trainval_slides.BiopsySlides(dataset_params, transform, fold, split, phase=phase)
    else:
        tile_normalizer = reinhard_bg.ReinhardNormalizer()
        # use the pre-computed LAB mean and std values
        tile_normalizer.fit(None)
        dataset = test_slides.BiopsySlides(dataset_params, transform, tile_normalizer)
    shuffle = True if phase == "train" else False

    loader = \
        torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                                    num_workers=num_workers, pin_memory=True)
    return loader





