import pickle
import os
from PIL import ImageFilter, Image
from prediction_models.att_mil.datasets import trainval_slides, trainval_slides_multi
import torch
import random
import torchvision.transforms as T
import tqdm
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


def compute_meanstd(num_workers, dataset, batch_size=1):
    print("Start compute mean and std")

    train_mean = torch.zeros(3)
    train_std = torch.zeros(3)
    loader = \
        torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                    num_workers=num_workers, pin_memory=False)
    loader_iter = iter(loader)

    cur_num, step = 0, 0
    for step in tqdm.tqdm(range(len(loader))):
        tiles, _, _, _ = loader_iter.next()
        tiles = torch.squeeze(tiles, dim=0)
        train_mean += torch.sum(torch.mean(tiles.view(tiles.size(0), tiles.size(1), -1), dim=2), dim=0)
        train_std += torch.sum(torch.std(tiles.view(tiles.size(0), tiles.size(1), -1), dim=2), dim=0)
        cur_num += tiles.size(0)
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


def get_meanstd(dataset_name, dataset_params=None, all_selected=None, num_workers=0):
    if dataset_name == "dw_sample_16":
        meanstd = {'mean': [0.8992915, 0.79110736, 0.8844037],
                    'std': [0.13978645, 0.2604748, 0.14999403]}
    elif dataset_name == "multi":
        meanstd = {"mean": [0.90949707, 0.8188697, 0.87795304],
                   "std": [0.36357649, 0.49984502, 0.40477625]}
    elif dataset_name == 'dw_sample_16v2':
        meanstd = {'mean': [0.878095, 0.807221, 0.8544836],
                    'std': [0.09033537, 0.17091176, 0.11264059]}
    elif dataset_name == "16_128_128":
        # {"mean": [1.0 - 0.90949707, 1.0 - 0.8188697, 1.0 - 0.87795304],
        meanstd = {"mean": [0.90949707, 0.8188697, 0.87795304],
                   "std": [0.36357649, 0.49984502, 0.40477625]}
    elif dataset_name == "br_256_256":
        # {"mean": [1.0 - 0.90949707, 1.0 - 0.8188697, 1.0 - 0.87795304],
        meanstd = {"mean": [0.90949707, 0.8188697, 0.87795304],
                   "std": [0.36357649, 0.49984502, 0.40477625]}
    elif dataset_name == "br_128_128":
        # {"mean": [1.0 - 0.90949707, 1.0 - 0.8188697, 1.0 - 0.87795304],
        meanstd = {"mean": [0.90949707, 0.8188697, 0.87795304],
                   "std": [0.36357649, 0.49984502, 0.40477625]}
    elif dataset_name == "br_256_2x":
        meanstd = {"mean": [0.90949707, 0.8188697, 0.87795304],
                   "std": [0.36357649, 0.49984502, 0.40477625]}
    elif dataset_name == "selected_10x":
        meanstd = {"mean": [0.772427, 0.539656, 0.693181],
                   "std": [0.147167, 0.187551, 0.136804]}
    elif dataset_name == "selected_10x_5x":
        meanstd = {"mean": [0.772427, 0.539656, 0.693181],
                   "std": [0.147167, 0.187551, 0.136804]}
        # cur_transform = T.Compose([
        #     T.ToTensor()
        # ])
        # dataset = trainval_slides.BiopsySlidesSelectedOTF(dataset_params, all_selected, cur_transform, None, None,
        #                                                   phase="meanstd")
        # meanstd = compute_meanstd(num_workers, dataset, batch_size=1)
        # print(meanstd)
    else:
        raise NotImplementedError(f"Mean and std for {dataset_name} not computed!!")

    return meanstd


# BATCH_DATASET = set(["dw_sample_16", "dw_sample_16v2", "16_128_128"])

def build_dataset_loader(batch_size, num_workers, dataset_params, split, phase, all_selected, fold=None, mil_arch=None,
                         has_drop_rate=0.0):
    """

    :param batch_size:
    :param num_workers:
    :param dataset_params:
    :param split: Which split of data to use
    :param phase: Whether to train/val on the split of data
    :param fold:
    :return:
    """
    if dataset_params.dataset in {"selected_10x"}:
        meanstd = get_meanstd(dataset_params.dataset, dataset_params, all_selected, num_workers)
    else:
        meanstd = get_meanstd(dataset_params.dataset)
    # Define different transformations
    normalize = [
        # T.Resize(dataset_params.input_size, interpolation=Image.ANTIALIAS),
        T.ToTensor(),
        T.Normalize(mean=meanstd['mean'], std=meanstd['std'])]
    augmentation = [
                       T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip(),
                       T.RandomAffine(degrees=180, fillcolor=(255, 255, 255)),

        # T.RandomResizedCrop(dataset_params.input_size, scale=(0.3, 1.0), ratio=(0.7, 1.4),
        #                     interpolation=INTERP),
        # T.RandomApply([
        #     T.ColorJitter(0.2, 0.2, 0.2, 0.1)  # not strengthened
        # ], p=0.8),
        # T.RandomGrayscale(p=0.2),
        # T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        # T.RandomHorizontalFlip(),
        ] + normalize
    if has_drop_rate > 0:
        assert split == "train"
    if split == "train":
        transform = T.Compose(augmentation)
    elif split == "val" or split == "test":
        transform = T.Compose(normalize)
    else:
        raise NotImplementedError(f"Not implemented for split {split}")
    # Multi-scale input dataset
    if dataset_params.dataset in {'multi'}:
        dataset = trainval_slides_multi.BiopsySlidesBatchMulti(dataset_params, transform, fold, split, phase=phase)
    elif dataset_params.dataset in {'selected_10x_5x'}:
        dataset = trainval_slides_multi.BiopsySlidesBatchMultiSelect(dataset_params, all_selected, transform, fold,
                                                                     split, phase=phase)
    elif dataset_params.dataset in {"dw_sample_16", "dw_sample_16v2"}:
        dataset = trainval_slides.BiopsySlidesBatch(dataset_params, transform, fold, split, phase=phase)
    elif dataset_params.dataset in {"16_128_128"}:
        dataset = trainval_slides.BiopsySlidesImage(dataset_params, transform, fold, split, phase=phase)
    elif dataset_params.dataset in {'br_256_256', 'br_128_128', 'br_256_2x'}:
        dataset = trainval_slides.BiopsySlidesBatchV2(dataset_params, transform, fold, split, phase=phase,
                                                      has_drop_rate=has_drop_rate)
    elif dataset_params.dataset in {"selected_10x"}:
        dataset = trainval_slides.BiopsySlidesSelectedOTF(dataset_params, all_selected, transform, fold, split,
                                                          phase=phase, has_drop_rate=has_drop_rate)

    else:
        dataset = trainval_slides.BiopsySlidesChunk(dataset_params, transform, fold, split, phase=phase)

    shuffle = True if split == "train" else False

    loader = \
        torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                                    num_workers=num_workers, pin_memory=True)
    return loader, dataset





