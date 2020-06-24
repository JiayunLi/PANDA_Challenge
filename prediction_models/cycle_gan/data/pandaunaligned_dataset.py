import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import pandas as pd
import numpy as np


class PANDAUnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_dir = opt.dataroot
        self.csv_folder = opt.csvfolder
        self.df_rad = pd.read_csv(os.path.join(self.csv_folder, "radboud_4_fold_train.csv"))
        self.df_kar = pd.read_csv(os.path.join(self.csv_folder, "karolinska_4_fold_train.csv"))
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        # self.A_size = len(self.A_paths)  # get the size of dataset A
        # self.B_size = len(self.B_paths)  # get the size of dataset B
        # self.A_size = np.sum(self.df_rad["tile_number"].tolist())
        # self.B_size = np.sum(self.df_kar["tile_number"].tolist())
        self.A_size = len(self.df_rad)
        self.B_size = len(self.df_kar)
        # print(self.A_size, self.B_size)
        # exit()

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_num = self.df_rad.loc[index % self.A_size, "tile_number"]
        A_num = np.random.choice(A_num, 1)[0]
        A_path = os.path.join(self.data_dir, self.df_rad.loc[index% self.A_size, "image_id"] + f"_{A_num}.png")
        # A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_num = self.df_kar.loc[index_B % self.B_size, "tile_number"]
        B_num = np.random.choice(B_num, 1)[0]
        B_path = os.path.join(self.data_dir, self.df_kar.loc[index_B, "image_id"] + f"_{B_num}.png")
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)