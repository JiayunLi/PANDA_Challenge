import os
import torch
import argparse


def trim_weights(model_dir, n_folds):
    if not os.path.isdir(f"{model_dir}/trimed_weights/"):
        os.mkdir(f"{model_dir}/trimed_weights/")
    for fold in range(n_folds):
        ckp_path = f"{model_dir}/{fold}/checkpoint_best.pth"
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
        hp = ckp['hyperparams']
        params = ckp['model']
        new_ckp = {"hyperparams": hp, "model": params}
        torch.save(new_ckp, f"{model_dir}/trimed_weights/checkpoint_best_{fold}.pth/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attention MIL PANDA challenge')

    # File location
    parser.add_argument('--model_dir', type=str, default='./cache/', help='Root directory for processed data')

    parser.add_argument('--n_folds', type=int, default=5)

    args = parser.parse_args()
    trim_weights(args.model_dir, args.n_folds)