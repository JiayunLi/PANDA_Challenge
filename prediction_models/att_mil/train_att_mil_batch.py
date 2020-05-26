import torch
from prediction_models.att_mil.utils import trainval_stats, model_utils
from prediction_models.att_mil.mil_models import config_model
import time
import sys
import gc
import numpy as np


def train_epoch(epoch, iteras, model, slide_criterion, tile_criterion, optimizer,
                train_loader, alpha, loss_type, log_every, logger, device):
    model.train()
    torch.cuda.empty_cache()
    # Quick check of status for log_every steps
    fast_stats = trainval_stats.AverageMeterSet()
    time_start = time.time()
    train_iter = iter(train_loader)
    for step in range(len(train_loader)):
        tiles, _, slide_label, _ = train_iter.next()
        if loss_type == "mse":
            slide_label = slide_label.float()
        slide_label = slide_label.to(device)
        tiles = tiles.to(device)
        slide_probs, _, _ = model(tiles)
        if loss_type == "mse":
            slide_loss = slide_criterion(slide_probs.view(-1), slide_label)
        else:
            slide_loss = slide_criterion(slide_probs, slide_label)
        # if len(tile_labels) > 0 and tile_labels[0][0] != -1:
        #     tile_labels = torch.stack(tile_labels).view(-1)
        #     if loss_type == "mse":
        #         tile_labels = tile_labels.float().to(device)
        #         # tile_loss = tile_criterion(tiles_probs.view(-1), tile_labels)
        #     else:
        #         tile_labels = tile_labels.to(device)
        #         # tile_loss = tile_criterion(tiles_probs, tile_labels)
        #     loss = alpha * tile_loss + (1 - alpha) * slide_loss
        # else:
        loss = slide_loss
        gc.collect()
        # backpropagate and take a step
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        cur_dict = {
            'slide_loss': slide_loss.item(),
            'tot_loss': loss.item(),
        }
        # if len(tile_labels) > 0 and tile_labels[0] != -1:
        #     cur_dict['tile_loss'] = tile_loss.item()
        fast_stats.update_dict(cur_dict, n=1)
        iteras += 1
        if step % log_every == 0 and step != 0:
            torch.cuda.empty_cache()
            time_stop = time.time()
            spu = (time_stop - time_start) / 100.
            cur_stats = fast_stats.pretty_string()
            print(f"Epoch {epoch}, Updates {step}/{len(train_loader)}, {cur_stats}, {spu:.4f}/update")
            logger.record_stats(fast_stats.averages(iteras, prefix='train/'))
            time_start = time.time()
            fast_stats = trainval_stats.AverageMeterSet()
    return iteras


def configure_criterion(loss_type, cls_weighted, use_binary, label_weights):
    if use_binary:
        if cls_weighted:
            tile_criterion = torch.nn.BCEWithLogitsLoss(label_weights)
        else:
            tile_criterion = torch.nn.BCEWithLogitsLoss()
        return tile_criterion

    if loss_type == 'mse':
        tile_criterion = torch.nn.MSELoss()
    elif loss_type == 'ce':
        if cls_weighted:
            tile_criterion = torch.nn.CrossEntropyLoss(label_weights)
        else:
            tile_criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Criterion not implemented {loss_type}")
    return tile_criterion


def val(epoch, model, val_loader, slide_criterion, loss_type, logger, slide_binary, device):
    model.eval()
    torch.cuda.empty_cache()
    # Quick check of status for log_every steps
    val_stats = trainval_stats.AverageMeterSet()
    time_start = time.time()
    val_iter = iter(val_loader)
    all_labels, all_preds = [], []
    # optimized_rounder = config_model.OptimizedRounder()
    with torch.no_grad():
        for step in range(len(val_loader)):
            tiles, tile_labels, slide_label, tile_names = val_iter.next()
            tiles = tiles.to(device)
            slide_probs, _, _ = model(tiles, phase='val')
            if loss_type == "mse":
                slide_label = slide_label.float()
            slide_label = slide_label.to(device)
            if loss_type == "mse":
                slide_loss = slide_criterion(slide_probs.view(-1), slide_label)
            else:
                slide_loss = slide_criterion(slide_probs, slide_label)
            cur_dict = {
                'slide_loss': slide_loss.item(),
            }
            val_stats.update_dict(cur_dict, n=1)
            if slide_binary:
                normalized_probs = torch.nn.Sigmoid(slide_probs)
                predicted = torch.ge(normalized_probs, 0.5).cpu().numpy()
            elif loss_type == "mse":
                predicted = np.squeeze(slide_probs.cpu().round().numpy()[:], axis=1)
            else:
                _, predicted = torch.max(slide_probs.data, 1)
                predicted = predicted.cpu().numpy()
            all_labels.append(slide_label.cpu().numpy())
            all_preds.append(predicted)

        print(f"Validation step {step}/{len(val_loader)}")
    # if loss_type == "mse":
    #     optimized_rounder.fit(all_preds, all_labels)
    #     coefficients = optimized_rounder.coefficients()
    #     all_preds = optimized_rounder.predict(all_preds, coefficients)
    #     quadratic_kappa = trainval_stats.compute_kappa(all_preds, all_labels)
    # else:
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    quadratic_kappa = trainval_stats.compute_kappa(all_preds, all_labels)
    val_stats.update_dict({"kappa": quadratic_kappa}, 1)

    cur_stats = val_stats.pretty_string()
    print(f"Epoch {epoch}, {cur_stats}, time: {time.time() - time_start}")
    sys.stdout.flush()
    logger.record_stats(val_stats.averages(epoch, prefix="val/"))
    torch.cuda.empty_cache()
    return quadratic_kappa, val_stats.avgs['slide_loss']