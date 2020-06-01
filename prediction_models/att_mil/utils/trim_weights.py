import os
import torch
import argparse
import shutil


def trim_weights(model_dir, weight_dir, n_folds):
    tempt = model_dir.split("/")
    i = len(tempt) - 1
    while i > 0:
        cur = tempt[i]
        if len(cur) > 0 and cur != "/":
            model_name = cur
            break
        i -= 1

    out_dir = f"{weight_dir}/{model_name}/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    shutil.copyfile(f"{model_dir}/options.pkl", f"{out_dir}/options.pkl")
    for fold in range(n_folds):
        ckp_path = f"{model_dir}/{fold}/checkpoint_best.pth"
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
        hp = ckp['hyperparams']
        params = ckp['model']
        new_ckp = {"hyperparams": hp, "model": params}
        print(f"{out_dir}/checkpoint_best_{fold}.pth")
        torch.save(new_ckp, f"{out_dir}/checkpoint_best_{fold}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attention MIL PANDA challenge')

    # File location
    parser.add_argument('--model_dir', type=str, default='./cache/', help='Root directory for processed data')
    parser.add_argument('--weights_dir', default='/raid/jiayunli/data/PANDA_challenge/trimmed_weights/')
    parser.add_argument('--n_folds', type=int, default=5)

    args = parser.parse_args()
    if not os.path.isdir(args.weights_dir):
        os.mkdir(args.weights_dir)
    trim_weights(args.model_dir, args.weights_dir, args.n_folds)