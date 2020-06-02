import torch
from prediction_models.att_mil.utils import trainval_stats, model_utils
from prediction_models.att_mil.mil_models import config_model
import time
import sys
import gc
import numpy as np


def train_epoch(epoch, fold, iteras, model, slide_criterion, tile_criterion, optimizer,
                train_loader, alpha, loss_type, log_every, logger, device, schedule):
    model.train()
    torch.cuda.empty_cache()
    # Quick check of status for log_every steps
    fast_stats = trainval_stats.AverageMeterSet()
    time_start = time.time()
    train_iter = iter(train_loader)
    if model.mil_params['mil_arch'] != 'pool' and alpha == 1:
        tile_loss_only = True
    else:
        tile_loss_only = False
    for step in range(len(train_loader)):
        tiles, tiles_labels, slide_label, _ = train_iter.next()
        tiles = tiles.to(device)

        # If only use tile-level loss
        if tile_loss_only:
            tiles_labels = torch.stack(tiles_labels, dim=0)
            tiles_labels = tiles_labels.view(tiles_labels.size(1), tiles_labels.size(0)).view(-1)
            # tiles: bs, n, c, w, h
            tiles = tiles.reshape(-1, tiles.size(2), tiles.size(3), tiles.size(4))
            _, tiles_probs, _ = model(tiles, phase="tiles_only")
            # if loss_type == "mse":
            #     tiles_labels = tiles_labels.float().to(device)
            #     cur_loss = tile_criterion(tiles_probs.view(-1), tiles_labels)
            # else:
            tiles_labels = tiles_labels.to(device)
            cur_loss = tile_criterion(tiles_probs, tiles_labels)
        else:
            slide_probs, _, _ = model(tiles)
            if loss_type == "mse":
                slide_label = slide_label.float().to(device)
                cur_loss = slide_criterion(slide_probs.view(-1), slide_label)
            else:
                slide_label = slide_label.to(device)
                cur_loss = slide_criterion(slide_probs, slide_label)

        gc.collect()
        # backpropagate and take a step
        optimizer.zero_grad()
        cur_loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        if tile_loss_only:
            cur_dict = {
                'tile_loss': cur_loss.item(),
            }
        else:
            cur_dict = {
                'slide_loss': cur_loss.item(),
            }

        fast_stats.update_dict(cur_dict, n=1)
        iteras += 1
        if step % log_every == 0 and step != 0:
            torch.cuda.empty_cache()
            time_stop = time.time()
            spu = (time_stop - time_start) / 100.
            cur_stats = fast_stats.pretty_string()
            print(f"Fold {fold}, Epoch {epoch}, Updates {step}/{len(train_loader)}, {cur_stats}, alpha is {alpha},"
                  f"{spu:.4f}/update")
            logger.record_stats(fast_stats.averages(iteras, prefix='train/'))
            time_start = time.time()
            fast_stats = trainval_stats.AverageMeterSet()
        # schedule.step()
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


def val(epoch, fold, model, val_loader, slide_criterion, tile_criterion, alpha, loss_type, logger, slide_binary, device):
    model.eval()
    torch.cuda.empty_cache()
    # Quick check of status for log_every steps
    val_stats = trainval_stats.AverageMeterSet()
    time_start = time.time()
    val_iter = iter(val_loader)
    all_labels, all_preds = [], []
    if model.mil_params['mil_arch'] != 'pool' and alpha == 1:
        tile_loss_only = True
    else:
        tile_loss_only = False
    # optimized_rounder = config_model.OptimizedRounder()
    with torch.no_grad():
        for step in range(len(val_loader)):
            tiles, tiles_labels, slide_label, tile_names = val_iter.next()
            tiles = tiles.to(device)
            # If only use tile-level loss
            if tile_loss_only:
                labels = torch.stack(tiles_labels, dim=0)
                labels = labels.view(labels.size(1), labels.size(0)).view(-1)
                # tiles: bs, n, c, w, h
                tiles = tiles.reshape(-1, tiles.size(2), tiles.size(3), tiles.size(4))
                _, probs, _ = model(tiles, phase="tiles_only")
                # if loss_type == "mse":
                #     labels = labels.float().to(device)
                #     cur_loss = tile_criterion(probs.view(-1), labels)
                # else:
                labels = labels.to(device)
                cur_loss = tile_criterion(probs, labels)
                cur_dict = {
                    'tile_loss': cur_loss.item(),
                }
            else:
                probs, _, _ = model(tiles)
                if loss_type == "mse":
                    labels = slide_label.float().to(device)
                    cur_loss = slide_criterion(probs.view(-1), labels)
                else:
                    labels = slide_label.to(device)
                    cur_loss = slide_criterion(probs, labels)
                cur_dict = {
                    'slide_loss': cur_loss.item(),
                }

            val_stats.update_dict(cur_dict, n=1)

            if slide_binary:
                normalized_probs = torch.nn.Sigmoid(probs)
                predicted = torch.ge(normalized_probs, 0.5).cpu().numpy()
            elif loss_type == "mse":
                predicted = np.squeeze(probs.cpu().round().numpy()[:], axis=1)
            else:
                _, predicted = torch.max(probs.data, 1)
                predicted = predicted.cpu().numpy()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted)

        print(f"Validation step {step}/{len(val_loader)}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    quadratic_kappa = trainval_stats.compute_kappa(all_preds, all_labels)
    val_stats.update_dict({"kappa": quadratic_kappa}, 1)

    cur_stats = val_stats.pretty_string()
    print(f"Fold {fold}, Epoch {epoch}, {cur_stats}, time: {time.time() - time_start}")
    sys.stdout.flush()
    prefix = "val_tiles/" if tile_loss_only else "val/"
    logger.record_stats(val_stats.averages(epoch, prefix=prefix))
    torch.cuda.empty_cache()
    if tile_loss_only:
        avg_loss = val_stats.avgs['tile_loss']
    else:
        avg_loss = val_stats.avgs['slide_loss']

    return quadratic_kappa, avg_loss
