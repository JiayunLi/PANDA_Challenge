import torch
from prediction_models.att_mil.utils import trainval_stats, model_utils
from prediction_models.att_mil.mil_models import config_model
import time
import sys
import gc


def configure_criterion(loss_type, cls_weighted, label_weights, is_tile):
    reduce_method = 'none' if is_tile else 'mean'
    if loss_type == 'mse':
        criterion = torch.nn.MSELoss(reduction=reduce_method)
    elif loss_type == 'ce':
        if cls_weighted:
            criterion = torch.nn.CrossEntropyLoss(label_weights, reduction=reduce_method)
        else:
            criterion = torch.nn.CrossEntropyLoss(reduce_method)
    elif loss_type == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss(reduce_method)
    else:
        raise NotImplementedError(f"Criterion not implemented {loss_type}")
    return criterion


def train_epoch(epoch, fold, iteras, model, slide_criterion, tile_criterion, optimizer,
                train_loader, alpha, loss_type, log_every, logger, device):
    model.train()
    torch.cuda.empty_cache()
    # Quick check of status for log_every steps
    fast_stats = trainval_stats.AverageMeterSet()
    time_start = time.time()
    train_iter = iter(train_loader)
    for step in range(len(train_loader)):
        tiles, tile_labels, slide_label, tile_names = train_iter.next()
        tiles = torch.squeeze(tiles, dim=0)

        # Reshape from n, bs to bs * n
        tile_labels = torch.stack(tile_labels, dim=0)
        tile_labels = tile_labels.view(tile_labels.size(1), tile_labels.size(0)).view(-1)

        if loss_type == "mse":
            slide_label = slide_label.float()
            tile_labels = tile_labels.float()

        tiles = tiles.to(device)
        slide_label = slide_label.to(device)
        tile_labels = tile_labels.to(device)

        slide_probs, tiles_probs, _ = model(tiles)
        if loss_type == "mse":
            slide_loss = slide_criterion(slide_probs.view(-1), slide_label)
            tile_loss = tile_criterion(tiles_probs.view(-1), tile_labels)
        else:
            slide_loss = slide_criterion(slide_probs, slide_label)
            tile_loss = tile_criterion(tiles_probs, tile_labels)
        tile_loss_mask = tile_labels > 0
        tile_loss *= tile_loss_mask
        tile_loss = tile_loss.mean()
        loss = alpha * tile_loss + (1 - alpha) * slide_loss

        gc.collect()
        # backpropagate and take a step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_dict = {
            'slide_loss': slide_loss.item(),
            'tot_loss': loss.item(),
            'tile_loss': tile_loss.item()
        }

        fast_stats.update_dict(cur_dict, n=1)
        iteras += 1
        if step % log_every == 0 and step != 0:
            torch.cuda.empty_cache()
            time_stop = time.time()
            spu = (time_stop - time_start) / 100.
            cur_stats = fast_stats.pretty_string()
            print(f"Fold {fold}, Epoch {epoch}, Updates {step}/{len(train_loader)}, {cur_stats}, {spu:.4f}/update")
            logger.record_stats(fast_stats.averages(iteras, prefix='train/'))
            time_start = time.time()
            fast_stats = trainval_stats.AverageMeterSet()
    return iteras


def val_epoch(epoch, fold, model, val_loader, slide_criterion, tile_criterion, loss_type, alpha, logger, device):
    model.eval()
    torch.cuda.empty_cache()
    # Quick check of status for log_every steps
    val_stats = trainval_stats.AverageMeterSet()
    time_start = time.time()
    val_iter = iter(val_loader)
    all_labels, all_preds = [], []
    optimized_rounder = config_model.OptimizedRounder()
    with torch.no_grad():
        for step in range(len(val_loader)):
            tiles, tile_labels, slide_label, tile_names = val_iter.next()
            tiles = torch.squeeze(tiles, dim=0)
            tiles = tiles.to(device)
            slide_probs, tiles_probs, _ = model(tiles)
            tile_labels = tile_labels.view(tile_labels.size(1), tile_labels.size(0)).view(-1)
            if loss_type == "mse":
                slide_label = slide_label.float()
                tile_labels = tile_labels.float()

            slide_label = slide_label.to(device)
            tile_labels = tile_labels.to(device)

            if loss_type == "mse":
                slide_loss = slide_criterion(slide_probs.view(-1), slide_label)
                tile_loss = tile_criterion(tiles_probs.view(-1), tile_labels)
            else:
                slide_loss = slide_criterion(slide_probs, slide_label)
                tile_loss = tile_criterion(tiles_probs, tile_labels)
            tile_loss_mask = tile_labels > 0
            tile_loss *= tile_loss_mask
            tile_loss = tile_loss.mean()
            loss = alpha * tile_loss + (1 - alpha) * slide_loss
            cur_dict = {
                'slide_loss': slide_loss.item(),
                'tile_loss': tile_loss.item(),
                'tot_loss': loss.item()
            }
            val_stats.update_dict(cur_dict, n=1)

            if loss_type == "mse":
                predicted = int(slide_probs.cpu().item())
            else:
                _, predicted = torch.max(slide_probs.data, 1)
                predicted = int(predicted.cpu().item())
            all_labels.append(int(slide_label[0]))
            all_preds.append(predicted)

        print(f"Validation step {step}/{len(val_loader)}")
    if loss_type == "mse":
        optimized_rounder.fit(all_preds, all_labels)
        coefficients = optimized_rounder.coefficients()
        all_preds = optimized_rounder.predict(all_preds, coefficients)
        quadratic_kappa = trainval_stats.compute_kappa(all_preds, all_labels)
    else:
        quadratic_kappa = trainval_stats.compute_kappa(all_preds, all_labels)
    val_stats.update_dict({"kappa": quadratic_kappa}, 1)

    cur_stats = val_stats.pretty_string()
    print(f"Fold {fold}, Epoch {epoch}, {cur_stats}, time: {time.time() - time_start}")
    sys.stdout.flush()
    logger.record_stats(val_stats.averages(epoch, prefix="val/"))
    torch.cuda.empty_cache()
    return quadratic_kappa, val_stats.avgs['slide_loss']