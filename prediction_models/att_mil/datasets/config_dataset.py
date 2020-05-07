from prediction_models.att_mil.utils import config_params
import pickle
import os
from prediction_models.att_mil.datasets import trainval_slides, test_slides
from torchvision import transforms


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def compute_meanstd():
    meanstd = dict()
    return meanstd


def build_dataset(batch_size, num_workers, dataset_params, split, phase, fold=None):
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
        meanstd = compute_meanstd()
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
        dataset = BiopsyTiles.BiopsySampleSlideDataset(data_dir, dataset_params, train_transform,
                                                             split='train')
    elif split == "val" or split == "test":
        transform = transforms.Compose([transforms.Resize(dataset_params.input_size)] + normalize)
    else:
        raise NotImplementedError(f"Not implemented for split {split}")

    if split == "train" or split == "val":
        dataset = trainval_slides.BiopsySlides(dataset_params, transform, fold, split, phase=phase)





