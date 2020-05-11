from prediction_models.att_mil.utils import config_params
import pickle
import os
from prediction_models.att_mil.datasets import trainval_slides, test_slides
from torchvision import transforms
from preprocessing.normalization import reinhard_bg
import torch


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def compute_meanstd(dataset_params, fold, num_workers, from_k_slides=1000):
    cur_transform = T.Compose([
        T.ToTensor()
    ])
    dataset = trainval_slides.BiopsySlides(dataset_params, cur_transform, fold, split="val", phase="meanstd")
    train_mean = torch.zeros(3)
    train_std = torch.zeros(3)
    loader = \
        torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False,
                                    num_workers=num_workers, pin_memory=True)
    loader_iter = iter(loader)

    cur_num, step = 0, 0
    while step < len(loader) and step < from_k_slides:
        tiles, _, _, _ = loader_iter.next()
        tiles = torch.squeeze(tiles, dim=0)
        train_mean += torch.sum(torch.mean(tiles.view(tiles.size(0), tiles.size(1), -1), dim=2), dim=0)
        train_std += torch.sum(torch.std(tiles.view(tiles.size(0), tiles.size(1), -1), dim=2), dim=0)
        cur_num += tiles.size(0)
        step += 1
    train_std /= cur_num
    train_mean /= cur_num
    meanstd = {
        "mean": float(train_mean),
        "std": float(train_std),
    }
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
    meanstd_file = f"{dataset_params.cache_dir}/{fold}/meanstd.pkl" if fold else f"{dataset_params.cache_dir}"

    if not os.path.isfile(meanstd_file):
        meanstd = compute_meanstd(dataset_params, fold, num_workers)
        pickle.dump(meanstd, open(meanstd_file, "wb"))
    meanstd = pickel.load(open(meanstd_file, "rb"))

    # Define different transformations
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean=meanstd['mean'],
                             std=meanstd['std'])]
    augmentation = [
        transforms.RandomResizedCrop(dataset_params.input_size, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                     interpolation=INTERP),
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        normalize]

    if split == "train":
        transform = transforms.Compose(augmentation)
    elif split == "val" or split == "test":
        transform = transforms.Compose([transforms.Resize(dataset_params.input_size)] + normalize)
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





