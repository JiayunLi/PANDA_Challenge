import os, sys
sys.path.append('../')
from input.tile_extraction_spine import write_2_zip_img
from utiles import utils
import pandas as pd
if __name__ == "__main__":
    TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'  ## train image folder
    OUT_TRAIN = '../input/panda-36x256x256-tiles-data-spine/val.zip'  ## output image folder
    csv_file = '../input/panda-36x256x256-tiles-data/wo_mask_val.csv'
    df = pd.read_csv(csv_file)
    names = list(df['image_id'])
    sz = 256  ## image patch size
    N = 36  ## how many patches selected from each slide
    print(len(names))  ## only images that have masks
    utils.check_folder_exists(os.path.dirname(OUT_TRAIN))
    Source_Folder = TRAIN
    Des_File = OUT_TRAIN
    mean, std, ratio, tile_number = write_2_zip_img(Source_Folder, Des_File, names, sz, N)