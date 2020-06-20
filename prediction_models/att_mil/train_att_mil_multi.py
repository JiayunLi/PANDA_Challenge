import torch
from prediction_models.att_mil.utils import trainval_stats, model_utils
import time
import sys
import gc
import numpy as np


def train_epoch(epoch, fold, iteras, model, slide_criterion, optimizer,
                train_loader, alpha, loss_type, log_every, logger, device, schedule):
    model.train()
    torch.cuda.empty_cache()
    # Quick check of status for log_every steps
    fast_stats = trainval_stats.AverageMeterSet()
    time_start = time.time()
    train_iter = iter(train_loader)

    for step in range(len(train_loader)):
        tiles_low, tiles_high, slide_label = train_iter.next()
        tiles_low = tiles_low.to(device)
        tiles_high = tiles_high.to(device)
        # If only use tile-level loss
        slide_probs, _, _ = model(tiles_low, tiles_high)
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

        cur_dict = {
            'slide_loss': cur_loss.item(),
        }

        del tiles_high
        del tiles_low
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


def val(epoch, fold, model, val_loader, slide_criterion, alpha, loss_type, logger, slide_binary, device):
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
            tiles_low, tiles_high, slide_label = val_iter.next()
            tiles_low = tiles_low.to(device)
            tiles_high = tiles_high.to(device)
            probs, _, _ = model(tiles_low, tiles_high)
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
                # elif (loss_type == "mse") and (not tile_loss_only):
                predicted = np.squeeze(probs.cpu().round().numpy()[:], axis=1)
            elif loss_type == "bce":
                predicted = probs.sigmoid().sum(1).detach().round().cpu().numpy()
            else:
                _, predicted = torch.max(probs.data, 1)
                predicted = predicted.cpu().numpy()
            if loss_type == 'bce':
                all_labels.append(labels.sum(1).cpu().numpy())
            else:
                all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted)
            del tiles_low
            del tiles_high
            gc.collect()

        print(f"Validation step {step}/{len(val_loader)}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    quadratic_kappa = trainval_stats.compute_kappa(all_preds, all_labels)
    val_stats.update_dict({"kappa": quadratic_kappa}, 1)

    cur_stats = val_stats.pretty_string()
    print(f"Fold {fold}, Epoch {epoch}, {cur_stats}, time: {time.time() - time_start}")
    sys.stdout.flush()
    logger.record_stats(val_stats.averages(epoch, prefix="val/"))
    torch.cuda.empty_cache()
    avg_loss = val_stats.avgs['slide_loss']

    return quadratic_kappa, avg_loss
