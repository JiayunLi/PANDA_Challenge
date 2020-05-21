import torch
import os
import argparse
import pickle
import random
import numpy as np
from prediction_models.att_mil.datasets import config_dataset
from prediction_models.att_mil.mil_models import config_model
from prediction_models.att_mil import train_att_mil, config_params
from prediction_models.att_mil.utils import checkpoint_utils

"""
python -m prediction_models.att_mil.main_trainval --data_dir /data/slides_encode_128/ --info_dir ./info/ --cuda --num_workers 4 --im_size 128 --cache_dir ./cache/  --feat_ft 5 --loss_type se --arch resnext50_32x4d  --alpha 0 --exp resNeXt_50_slideonly
"""

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_binary_options(cur):
    if cur.isnumeric():
        cur = int(cur)
        if cur == 0 or cur < 0:
            return False
        else:
            return True
    else:
        cur = cur.lower()
        if cur == "t" or cur == "true":
            return True
        else:
            return False


def trainval(opts):
    # Use restarted model options and load model checkpoint
    if opts.ckp_dir:
        opts, ckp = checkpoint_utils.load_options(opts.ckp_dir, opts.load_best, opts.data_dir, opts.cuda,
                                                  opts.num_workers)
    else:
        ckp = None
    if not hasattr(opts, "schedule_type"):
        opts.schedule_type = "plateau"

    opts.data_dir = f"{opts.data_dir}/{opts.dataset}/"
    opts.cls_weighted = parse_binary_options(opts.cls_weighted)
    opts.slide_binary, opts.tile_binary = parse_binary_options(opts.slide_binary), parse_binary_options(opts.tile_binary)
    if opts.tile_binary:
        opts.tile_classes = 1
    if opts.slide_binary:
        opts.slide_classes = 1

    # Write options
    print(opts)

    # Generate tile-level labels
    if not os.path.isfile(f"{opts.data_dir}/tile_labels_{opts.dataset}.json"):
        from prediction_models.att_mil.utils import dataset_utils
        masks_ldmb_dir = f"{opts.data_dir}/label_masks/"

        dataset_utils.generate_tile_label_json(masks_ldmb_dir, opts.data_dir, mask_size=opts.im_size,
                                               trainval_file=f"{opts.data_dir}/train.csv", binary_label=False)

    # Generate cross validation file
    if not os.path.isfile(f"{opts.info_dir}/train_{opts.start_fold}.csv"):
        from prediction_models.att_mil.utils import dataset_utils
        dataset_utils.generate_cv_split(
            f"{opts.data_dir}/train.csv", opts.info_dir, opts.n_folds, opts.manual_seed, opts.re_split)

    pickle.dump(opts, open(f"{opts.exp_dir}/options.pkl", "wb"))

    dataset_params = config_params.DatasetParams(opts.im_size, opts.input_size, opts.info_dir,
                                                 opts.data_dir, opts.cache_dir, opts.exp_dir, opts.dataset,
                                                 opts.num_channels)
    mil_params = config_params.set_mil_params(opts.mil_f_size, opts.ins_embed, opts.bag_embed,
                                              opts.bag_hidden, opts.slide_classes, opts.tile_classes,
                                              opts.loss_type, opts.schedule_type)
    trainval_params = config_params.TrainvalParams(opts.lr, opts.feat_lr, opts.wd, opts.train_blocks,
                                                   opts.optim, opts.epochs, opts.feat_ft, opts.log_every, opts.alpha,
                                                   opts.loss_type, opts.cls_weighted, opts.schedule_type,
                                                   opts.slide_binary, opts.tile_binary)
    end_fold = opts.end_fold if opts.end_fold > 0 else opts.n_folds

    device = "cpu" if not opts.cuda else "cuda"
    # Train and validation
    for fold in range(opts.start_fold, end_fold):
        print(f"Start train and validation for fold {fold}!")
        # Only support batch size 1 for MIL with variable input size.
        train_loader, train_data = config_dataset.build_dataset_loader(1, opts.num_workers, dataset_params,
                                                                      split="train", phase="train", fold=fold)
        val_loader, val_data = config_dataset.build_dataset_loader(1, opts.num_workers, dataset_params,
                                                                   split="val", phase="val", fold=fold)
        model, optimizer, scheduler, start_epoch, iters, checkpointer = \
            config_model.config_model_optimizer(opts, ckp, fold, mil_params, steps_per_epoch=len(train_loader))
        exp_dir = f"{opts.exp_dir}/{fold}/"
        # Disable model loading for next fold.
        ckp = None
        train_att_mil.trainval(fold, exp_dir, start_epoch, iters, trainval_params, model, optimizer, scheduler,
                               checkpointer, train_loader, train_data, val_loader, device)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attention MIL PANDA challenge')

    # File location
    parser.add_argument('--data_dir', type=str, default='/data/', help='Root directory for processed data')
    parser.add_argument('--info_dir', type=str, default='.../info/', help='Directory for cross validation information')
    parser.add_argument('--cache_dir', type=str, default='.../cache/', help='Directory to save trained models')
    parser.add_argument('--dataset', type=str, default="dw_sample_16", help='Different types of processed tiles')

    # Cross validation
    parser.add_argument('--start_fold', type=int, default=0)
    parser.add_argument('--end_fold', type=int, default=-1)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--re_split', action='store_true', help="Re-generate cross validation split")

    # Hardward
    parser.add_argument('--cuda', action='store_true', help='Whether to use cuda')
    parser.add_argument('--num_workers', type=int, default=0, help='How many dataloader workers to use')
    parser.add_argument('--manual_seed', type=int, default=6)

    # Input data options
    parser.add_argument('--im_size', default=128, type=int, help="original extracted tile size")
    parser.add_argument('--input_size', default=128, type=int, help="input size to the network")
    parser.add_argument('--num_channels', default=3, type=int, help="# of input image channels")

    # Restart
    parser.add_argument('--ckp_dir', default=None)
    parser.add_argument('--load_best', action='store_true')

    # Model options
    # Encoder
    parser.add_argument('--arch', default='vgg11_bn', help="Network architecture for the tile encoder")
    parser.add_argument('--tile_classes', default=4, type=int, help="Number of prediction classes for tiles")
    parser.add_argument('--pretrained', default=True)
    # MIL
    parser.add_argument('--mil_f_size', default=4, type=int, help="Feature map size to the MIL part")
    parser.add_argument('--ins_embed', default=256, type=int, help="Instance embedding size")
    parser.add_argument('--bag_embed', default=512, type=int, help="Bag embedding size")
    parser.add_argument('--bag_hidden', default=256, type=int, help="Bag hidden size")
    parser.add_argument('--slide_classes', default=6, type=int, help="Number of prediction classes for slides")

    # Training options
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for classifier')
    parser.add_argument('--feat_lr', default=0.0005, type=float, help='learning rate for features')
    parser.add_argument('--wd', type=float, default=10e-5, metavar='R', help='weight decay')
    parser.add_argument('--train_blocks', default=4, type=int, help='Train How many blocks')
    parser.add_argument('--optim', default='adam', help="Optimizer used for model training")
    parser.add_argument('--schedule_type', default='plateau', help="options: plateau | cycle")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--feat_ft', default=0, type=int, help="Start finetune features, -1: start from epoch 0")
    parser.add_argument('--log_every', default=50, type=int, help='Log every n steps')
    parser.add_argument('--alpha', default=0.5, type=int, help='weighted factor for tile loss')
    parser.add_argument('--loss_type', default='ce', type=str,
                        help="Different types of loss functions; cross entropy, MSE")
    parser.add_argument('--cls_weighted', default='f', type=str, help='Whether to use weighted  loss')
    parser.add_argument('--slide_binary', default='f', type=str, help='Only predict cancer versus noon cancer for slide'
                                                                      'classification')
    parser.add_argument('--tile_binary', default='f', type=str, help='Only predict cancer versus noon cancer for slide'
                                                                      'classification')

    # Exp
    parser.add_argument('--exp', default='debug', help="Name of current experiment")

    args = parser.parse_args()

    args.exp_dir = f"{args.cache_dir}/{args.exp}/"
    seed_torch(args.manual_seed)
    if not os.path.isdir(args.exp_dir):
        os.mkdir(args.exp_dir)

    trainval(args)
