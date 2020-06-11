import torchvision.transforms as T
from PIL import Image
from prediction_models.att_mil.datasets import test_slides
from prediction_models.att_mil.mil_models import mil
import tqdm
# from prediction_models.att_mil.utils import reinhard_fast as reinhard_bg
import numpy as np
import torch
import time
import pickle
import pandas as pd
import glob
import skimage.io
from prediction_models.att_mil.datasets import get_selected_locs, gen_selected_tiles
from prediction_models.att_mil.datasets.gen_selected_tiles import RATE_MAP


class TestParams:
    def __init__(self, test_slides_dir, im_size, input_size, loss_type,
                 dw_rate=None, ts_thres=None, overlap=None, top_n=40, lowest_im_size=32, level=-2,
                 num_channels=3):
        self.test_slides_dir = test_slides_dir
        self.im_size = im_size
        self.input_size = input_size
        self.dw_rate = dw_rate
        self.overlap = overlap
        self.ts_thres = ts_thres
        self.num_channels = num_channels
        self.loss_type = loss_type
        self.top_n = top_n
        self.level = level
        self.lowest_im_size = lowest_im_size


def load_opts(ckp_dir, data_dir, cuda, num_workers, batch_size):
    model_opts = pickle.load(open(f"{ckp_dir}/options.pkl", "rb"))
    model_opts.cuda = cuda
    model_opts.num_workers = num_workers
    model_opts.data_dir = data_dir
    model_opts.device = "gpu" if cuda else "cpu"
    model_opts.batch_size = batch_size
    return model_opts


def load_model(ckp_path, device):
    if device == 'cuda':
        ckp = torch.load(ckp_path)
    else:
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
    hp = ckp['hyperparams']
    params = ckp['model']
    if hp['mil_params']['mil_arch'] == 'pool_simple':
        model = mil.PoolSimpleInfer(hp['mil_params'])
    else:
        base_encoder, feature_dim = \
            mil.config_encoder_infer(hp['input_size'], hp['mil_params']['n_tile_classes'], hp['encoder_arch'], False)
        if hp['mil_params']['mil_arch'] == 'pool':
            model = mil.PoolMilBatch(base_encoder, hp['pretrained'], hp['encoder_arch'], hp['input_size'],
                                     feature_dim, hp['mil_params'])
        else:
            print("Use Attention MIL model")
            model = mil.AttMILBatch(base_encoder, hp['pretrained'], hp['encoder_arch'], hp['input_size'],
                                    feature_dim, hp['mil_params'])
    model.load_state_dict(params)
    return model


def load_models(exp_dir, device, load_best=True):
    exp_fold_paths = glob.glob(f"{exp_dir}/*.pth")
    models = []
    for cur_ckp_path in exp_fold_paths:
        cur_model = load_model(cur_ckp_path, device)
        models.append(cur_model)
        cur_model.eval()
    return models


def test_w_atts(all_models, meanstd, test_slides_df, test_params, num_workers, batch_size, device):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=meanstd['mean'], std=meanstd['std'])])
    dataset = test_slides.BiopsySlideSelected(test_params, test_slides_df, normalize, phase='w_atts')
    loader = \
        torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                    num_workers=num_workers, pin_memory=False)
    print("Start apply model")
    pred_data = []
    test_iter = iter(loader)
    start_time = time.time()
    all_atts, all_tile_ids, all_tile_pads = {}, {}, {}
    for cur_model in all_models:
        cur_model.to(device)
        cur_model.eval()
    with torch.no_grad():
        for step in tqdm.tqdm(range(len(loader))):
        # for step in range(len(loader)):
            tiles, image_ids, tile_ids, pad_tops, pad_lefts, nrows, ncols = test_iter.next()
            tiles = tiles.to(device)
            bs, N, C, h, w = tiles.shape
            # dihedral TTA
            tiles = torch.stack([tiles, tiles.flip(-1), tiles.flip(-2), tiles.flip(-1,-2),
                                 tiles.transpose(-1, -2), tiles.transpose(-1, -2).flip(-1),
                                 tiles.transpose(-1, -2).flip(-2), tiles.transpose(-1, -2).flip(-1, -2)], 1)
            tiles = tiles.view(-1, N, C, h, w) # bs*8, top_n, 3, sz, sz
            pooled_predicted, pooled_atts = [], []
            for cur_model in all_models:
                cur_preds, _, tiles_atts = cur_model(tiles)
                pooled_predicted.append(cur_preds.cpu())
                pooled_atts.append(tiles_atts.cpu())
            pooled_predicted = torch.stack(pooled_predicted, 1)  # [bs * 8, n_models, 6]
            pooled_atts = torch.stack(pooled_atts, 1)  # [bs * 8, top_n, n_models, 1]

            pooled_predicted = pooled_predicted.view(bs, 8 * len(all_models), -1) # [bs, 8(augmentation) * 4 (model), 6]
            pooled_atts = pooled_atts.view(bs, 8 * len(all_models), -1) # [bs, 8 * n_models, top_n]
            if test_params.loss_type == "mse":
                pooled_predicted = np.squeeze(pooled_predicted.mean(1).round().cpu().numpy()[:], axis=1)  # [bs]
            else:
                pooled_predicted = pooled_predicted.mean(1).argmax(-1).cpu().numpy()[:]  # [bs]
            pooled_atts = pooled_atts.mean(1).cpu().numpy()[:] # [bs, top_n]
            batch_idx = 0
            for image_id, cur_pred, cur_atts, pad_top, pad_left, nrow, ncol in \
                    zip(image_ids, pooled_predicted, pooled_atts, pad_tops, pad_lefts, nrows, ncols):
                pred_data.append({"image_id": str(image_id), "isup_grade": int(cur_pred)})
                all_atts[str(image_id)] = cur_atts
                # all_tile_ids[str(image_id)] = tile_ids[batch_idx, :, :].cpu().numpy()[:]
                all_tile_ids[str(image_id)] = tile_ids[batch_idx].cpu().numpy()[:]
                all_tile_pads[str(image_id)] = (int(pad_top), int(pad_left), int(nrow), int(ncol))
                batch_idx += 1
            del tiles
            del pooled_predicted
            del pooled_atts
    print(f"Total time to apply the model: {time.time() - start_time}")
    pred_df = pd.DataFrame(columns=["image_id", "isup_grade"], data=pred_data)
    return pred_df, all_atts, all_tile_ids, all_tile_pads


def test_helper(all_models, loader, device, w_atts):
    print("Start apply model")
    test_iter = iter(loader)
    start_time = time.time()
    predicted = []
    for cur_model in all_models:
        cur_model.to(device)
        cur_model.eval()
    att_results = {"atts": {}, "tile_ids": {}, "tile_pads": {}}
    # all_atts, all_tile_ids, all_tile_pads = {}, {}, {}
    with torch.no_grad():
        for _ in tqdm.tqdm(range(len(loader))):
            if w_atts:
                tiles, image_ids, tile_ids, pad_tops, pad_lefts, nrows, ncols = test_iter.next()
            else:
                tiles = test_iter.next()
            tiles = tiles.to(device)
            bs, N, C, h, w = tiles.shape
            # dihedral TTA
            tiles = torch.stack([tiles, tiles.flip(-1), tiles.flip(-2), tiles.flip(-1, -2),
                                 tiles.transpose(-1, -2), tiles.transpose(-1, -2).flip(-1),
                                 tiles.transpose(-1, -2).flip(-2), tiles.transpose(-1, -2).flip(-1, -2)], 1)
            tiles = tiles.view(-1, N, C, h, w)

            pooled_probs, pooled_atts = [], []
            # Apply each fold model to the data
            for cur_model in all_models:
                cur_probs, _, tiles_atts = cur_model(tiles)
                pooled_probs.append(cur_probs.cpu())
                if w_atts:
                    pooled_atts.append(tiles_atts.cpu())
            pooled_probs = torch.stack(pooled_probs, 1)  # [bs * 8, 4, 6]
            pooled_probs = pooled_probs.view(bs, 8 * len(all_models), -1)  # [bs, 8(augmentation) * 4 (model), 6]
            pooled_probs = torch.squeeze(pooled_probs.mean(1).cpu(), dim=1)
            predicted.append(pooled_probs)

            if w_atts:
                pooled_atts = torch.stack(pooled_atts, 1)  # [bs * 8, n_models, top_n, 1]
                pooled_atts = pooled_atts.view(bs, 8 * len(all_models), -1)  # [bs, 8 * n_models, top_n, 1]
                pooled_atts = pooled_atts.mean(1).cpu().numpy()[:]  # [bs, top_n]
                batch_idx = 0
                for image_id, cur_atts, pad_top, pad_left, nrow, ncol in \
                        zip(image_ids, pooled_atts, pad_tops, pad_lefts, nrows, ncols):
                    att_results["atts"][str(image_id)] = cur_atts
                    att_results["tile_ids"][str(image_id)] = tile_ids[batch_idx].cpu().numpy()[:]
                    att_results["tile_pads"][str(image_id)] = (int(pad_top), int(pad_left), int(nrow), int(ncol))
                    batch_idx += 1
            del tiles
            del pooled_atts
            del pooled_probs
    del all_models
    print(f"Finished one detection stage time: {time.time() - start_time}")
    return predicted, att_results


def selection(test_df, slides_dir, all_atts,  att_level, att_tile_size_lowest, att_n,
              select_sub_size_lowest, n_sub):
    all_selected = dict()
    select_sub_size = RATE_MAP[att_level] * select_sub_size_lowest
    for i in range(len(test_df)):
        slide_id = test_df.iloc[i].image_id
        tile_ids = all_atts["tile_ids"][slide_id]
        cur_atts = all_atts["atts"][slide_id]
        pad_top, pad_left, n_row, n_col = all_atts['tile_pads'][slide_id]
        max_atts_ids = np.argsort(cur_atts)[::-1]
        cur_tile_ids = tile_ids[max_atts_ids]
        cur_tile_ids_fix = []
        for tile_id in cur_tile_ids:
            if tile_id == "-1":
                continue
            cur_tile_ids_fix.append(int(tile_id))
        cur_tile_ids_fix = cur_tile_ids_fix[:att_n]
        orig = skimage.io.MultiImage(f"{slides_dir}/{slide_id}.tiff")
        results = gen_selected_tiles.get_highres_tiles(orig, cur_tile_ids_fix, pad_top, pad_left,
                                                       att_tile_size_lowest, None, att_level,
                                                       top_n=-1, n_row=n_row, n_col=n_col)
        cur_selected_locs = get_selected_locs.select_sub_simple_4x4(cur_tile_ids_fix, att_tile_size_lowest, None,
                                                                    pad_top, pad_left, select_sub_size_lowest,
                                                                    select_sub_size, results['tiles'], n_sub,
                                                                    n_row=n_row, n_col=n_col)
        all_selected[slide_id] = cur_selected_locs
    return all_selected


def get_model_name(model_dir):
    tempt = model_dir.split("/")
    for i in range(len(tempt)-1, -1, -1):
        if len(tempt[i]) > 0:
            return tempt[i]
    return tempt[0]


# Generate attention maps for cross validation dataset
# Here we only use trained model to apply on the validation fold versus during testing, we amy apply all models.
def get_cv_attentions(args):
    meanstd = {"mean": [0.90949707, 0.8188697, 0.87795304], "std": [0.36357649, 0.49984502, 0.40477625]}
    opts = load_opts(args.model_dir, args.data_dir, args.cuda, args.num_workers, args.batch_size)
    device = "cpu" if not args.cuda else "cuda"

    test_params = TestParams(opts.data_dir, opts.im_size, opts.input_size, opts.loss_type,
                             lowest_im_size=args.lowest_im_size, level=args.level, top_n=args.top_n)
    all_atts_info = {"atts": {}, "tile_ids": {}, "pad_infos": {}, "level":  args.level,
                     "lowest_im_size": args.lowest_im_size}
    model_name = get_model_name(args.model_dir)
    for fold in range(opts.n_folds):
        ckp_path = f"{args.model_dir}/checkpoint_best_{fold}.pth"
        print(f"start loading model from {fold}, {ckp_path}")
        predictor = load_model(ckp_path, device)
        cur_df = pd.read_csv(f"{args.info_dir}/val_{fold}.csv")
        _, cur_split_atts, cur_split_tile_ids, all_tile_pads = \
            test_w_atts([predictor], meanstd, cur_df, test_params, opts.num_workers, opts.batch_size, device)
        for slide_id, atts in cur_split_atts.items():
            all_atts_info['atts'][slide_id] = atts
            all_atts_info['tile_ids'][slide_id] = cur_split_tile_ids[slide_id]
            all_atts_info['pad_infos'][slide_id] = all_tile_pads[slide_id]
    np.save(f"{args.att_dir}/{model_name}_n_{args.top_n}_sz_{opts.im_size}.npy", all_atts_info)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/storage_slides/PANDA_challenge/16_128_128/")
    parser.add_argument('--info_dir', default='./info/16_128_128/')
    parser.add_argument('--model_dir',
                        default='/data/storage_slides/PANDA_challenge/trimmed_weights/resnext50_3e-4_bce_256/')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--top_n', type=int, default=36)
    parser.add_argument('--lowest_im_size', type=int, default=64)
    parser.add_argument('--level', type=int, default=-2)

    parser.add_argument('--cuda', action='store_true', help="whether to use GPU for training")

    parser.add_argument('--att_dir', default='../../cache/atts/')

    options = parser.parse_args()
    if not os.path.isdir(options.att_dir):
        os.mkdir(options.att_dir)

    get_cv_attentions(options)
    # opts = load_opts(options.model_dir, options.data_dir, options.cuda, 0, 2)
    # device = "cpu" if not options.cuda else "cuda"
    #
    # test_params = TestParams(opts.data_dir, opts.im_size, opts.input_size, opts.loss_type,
    #                          lowest_im_size=options.lowest_im_size, level=options.level, top_n=options.top_n)
    # ckp_path = f"{options.model_dir}/checkpoint_best_{0}.pth"
    # print(f"start loading model from {0}, {ckp_path}")
    # predictor = load_model(ckp_path, device)
    #
    # cur_df = pd.read_csv(f"{options.info_dir}/val_{0}.csv", index_col=0).head(2)
    # meanstd = {"mean": [0.90949707, 0.8188697, 0.87795304], "std": [0.36357649, 0.49984502, 0.40477625]}
    # normalize = T.Compose([
    #     T.ToTensor(),
    #     T.Normalize(mean=meanstd['mean'], std=meanstd['std'])])
    # dataset = test_slides.BiopsySlideSelected(test_params, cur_df, normalize, phase='w_atts')
    # loader = \
    #     torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=False, drop_last=False,
    #                                 num_workers=0, pin_memory=False)
    # cur_predicted_probs, att_results = test_helper([predictor], loader, device, True)
    # np.save("../../cache/debug_atts.npy", att_results)
    # # att_results = dict(np.load("../../cache/debug_atts.npy", allow_pickle=True).tolist())
    # selected_locs = selection(cur_df, opts.data_dir, att_results, options.level, options.lowest_im_size, options.top_n // 2,
    #                           select_sub_size_lowest=16, n_sub=2)
    # np.save("../../cache/debug_atts_locs.npy", selected_locs)


