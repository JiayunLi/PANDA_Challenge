import torch
import os
import argparse
from prediction_models.att_mil.utils import config_params


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def trainval(opts):





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attention MIL PANDA challenge')

    # File location
    parser.add_argument('--data_dir', type=str, default='/data/', help='Root directory for processed data')
    parser.add_argument('--info_dir', type=str, default='./info/', help='Directory for cross validation information')
    parser.add_argument('--cache_dir', type=str, default='./cache/', help='Directory to save trained models')

    parser.add_argument('--phase', type=str, default='trainval')
    # Cross validation
    parser.add_argument('--start_fold', type=int, default=0)
    parser.add_argument('--end_fold', type=int, default=-1)
    parser.add_argument('--n_folds', type=int, default=5)

    # Hardward
    parser.add_argument('--cuda', action='store_true', help='Whether to use cuda')
    parser.add_argument('--num_workers', type=int, default=0, help='How many dataloader workers to use')

    # Restart
    parser.add_argument('--ckp_dir', default=None)

    # Model options
    # Encoder
    parser.add_argument('--arch', default='vgg11_bn', help="Network architecture for the tile encoder")
    parser.add_argument('--arch', default='vgg11_bn', help="Network architecture for the tile encoder")
    parser.add_argument('--tile_classes', default=4, type=int, help="Number of prediction classes for tiles")
    # MIL
    parser.add_argument('--mil_f_size', default=4, type=int, help="Feature map size to the MIL part")
    parser.add_argument('--ins_embed', default=256, type=int, help="Instance embedding size")
    parser.add_argument('--bag_embed', default=512, type=int, help="Bag embedding size")
    parser.add_argument('--bag_hidden', default=256, type=int, help="Bag hidden size")
    parser.add_argument('--slide_classes', default=6, type=int, help="Number of prediction classes for slides")

    args = parser.parse_args()