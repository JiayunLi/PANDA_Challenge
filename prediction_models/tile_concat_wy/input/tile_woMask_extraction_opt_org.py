import os, sys
sys.path.append('../')
from input.tile_extraction_opt_org import write_2_file_img
from utiles import utils
import pickle
import pandas as pd
if __name__ == "__main__":
    TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'  ## train image folder
    OUT_TRAIN = '../input/panda-36x256x256-tiles-data-opt/train/'  ## output image folder
    MARKER = "../input/prostate-cancer-grade-assessment/marker_images/"
    csv_file = '../input/panda-36x256x256-tiles-data-spine-loc/wo_mask_val.csv'
    df = pd.read_csv(csv_file)
    names = list(df['image_id'])
    pen_marked_images = [name[:-4] for name in os.listdir(MARKER)]
    sz = 256  ## image patch size
    N = 36  ## how many patches selected from each slide
    print(len(names))  ## only images that have masks
    utils.check_folder_exists(os.path.dirname(OUT_TRAIN))
    Source_Folder = TRAIN
    Des_File = OUT_TRAIN
    mean, std, ratio, tile_number, tile_location = write_2_file_img(Source_Folder, Des_File, names, pen_marked_images, sz, N)
    with open(os.path.join(os.path.dirname(OUT_TRAIN), 'tile_loc_{}.pkl'.format("val")), 'wb') as f:
        pickle.dump(tile_location, f)