import pandas as pd
import pickle
import sys
import glob
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
import time
import os
import torch.utils.data as data
from fastai.vision import *
from prediction_models.att_mil.test_model import load_model, load_opts, TestParams, selection, test_helper
from prediction_models.att_mil.datasets import test_slides


def run_mid_res(predictors, model_dir, model_name, slides_dir, test_slides_df, mode):
    # lowest_im_size, input_size, level, top_n, att_n = 64, 256, -2, 36, 9
    lowest_im_size, input_size, level, top_n, att_n = 64, 256, -2, 36, 18

    meanstd = {"mean": [0.90949707, 0.8188697, 0.87795304], "std": [0.36357649, 0.49984502, 0.40477625]}
    opts = load_opts(f"{model_dir}/{model_name}", slides_dir, cuda, num_workers, batch_size)
    # predictors = load_models(f"{model_dir}/{model_name}", opts.device)
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=meanstd['mean'], std=meanstd['std'])])

    test_params = TestParams(slides_dir, opts.im_size, opts.input_size, opts.loss_type, top_n=top_n,
                             lowest_im_size=lowest_im_size, level=level)
    dataset = test_slides.BiopsySlideSelected(test_params, test_slides_df, normalize, mode, phase='w_atts',
                                              select_method='4x4')
    loader = \
        torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                    num_workers=num_workers, pin_memory=False)
    start_time = time.time()
    cur_predicted_probs, att_results = test_helper(predictors, loader, device, True)
    all_selected = selection(test_slides_df, att_results, att_n)
    print(f"Total time to run stage 1 model: {time.time() - start_time}")
    del predictors
    return cur_predicted_probs, all_selected


def run_high_res(predictors, model_dir, model_name, slides_dir, test_slides_df, selected):
    # lowest_im_size, input_size, level, top_n = 64, 512, -3, 9
    lowest_im_size, input_size, level, top_n = 32, 256, -3, 36

    meanstd = {"mean": [0.772427, 0.539656, 0.693181], "std": [0.147167, 0.187551, 0.136804]}
    opts = load_opts(f"{model_dir}/{model_name}", slides_dir, cuda, num_workers, batch_size)
    # predictors = load_models(f"{model_dir}/{model_name}", opts.device)
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=meanstd['mean'], std=meanstd['std'])])

    test_params = TestParams(slides_dir, opts.im_size, opts.input_size, opts.loss_type, top_n=top_n,
                             lowest_im_size=lowest_im_size, level=level)
    dataset = test_slides.BiopsyHighresSelected(test_params, test_slides_df, selected, normalize)
    loader = \
        torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                    num_workers=num_workers, pin_memory=False)
    start_time = time.time()
    cur_predicted_probs, _ = test_helper(predictors, loader, device, False)

    print(f"Total time to run stage 2 model: {time.time() - start_time}")
    del predictors
    return cur_predicted_probs


if __name__ == "__main__":
    cuda = True
    num_workers = 4
    batch_size = 2
    device = 'cuda'
    model_name_mid = "resnext50_3e-4_bce_256"
    model_name_high = "resnet50_256_10x_consine_bce"
    loss_type = "bce"

    # data location
    slides_dir = "/slides_data//train_images/"
    model_dir = "/weights/"
    info_dir = "/PANDA_Challenge/info/"
    start_time = time.time()
    prob_data = []

    for fold in range(4):
        cur_df = pd.read_csv(f"{info_dir}/val_{fold}.csv")
        mid_res_ckp_path = f"{model_dir}/{model_name_mid}/checkpoint_best_{fold}.pth"
        print(f"start loading mid resolution model from {fold}, {mid_res_ckp_path}")
        mid_res_predictor = load_model(mid_res_ckp_path, device)
        mid_res_raw_probs, all_selected = run_mid_res([mid_res_predictor],
                                                      model_dir, model_name_mid, slides_dir, cur_df, mode=0)
        high_res_ckp_path = f"{model_dir}/{model_name_high}/checkpoint_best_{fold}.pth"
        print(f"start loading high resolution model from {fold}, {high_res_ckp_path}")
        high_res_predictor = load_model(high_res_ckp_path, device)
        high_res_raw_probs = run_high_res([high_res_predictor], model_dir, model_name_high, slides_dir,
                                          cur_df, all_selected)
        high_res_raw_probs = high_res_raw_probs.sigmoid().sum(1).detach().cpu().numpy()

        for i in range(len(cur_df)):
            slide_info = cur_df.iloc[i]
            slide_id = slide_info.image_id
            cur_high_res_prob = high_res_raw_probs[i]
            prob_data.append({"image_id": slide_id,
                              "prob": float(high_res_raw_probs[i])})
    prob_data_df = pd.DataFrame(data=prob_data, columns=["image_id", "prob"])
    prob_data_df.to_csv('/PANDA_Challenge/cache/cv_probs_jiayun.csv', index=False)
