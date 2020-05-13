import torch
from prediction_models.att_mil.utils import trainval_stats
from prediction_models.att_mil.mil_models import config_model
import time
import sys
import gc


def train_epoch(epoch, iteras, model, slide_criterion, tile_criterion, optimizer,
                train_loader, alpha, log_every, logger, device):
    model.train()
    torch.cuda.empty_cache()
    # Quick check of status for log_every steps
    fast_stats = trainval_stats.AverageMeterSet()
    time_start = time.time()
    train_iter = iter(train_loader)
    for step in range(len(train_loader)):
        tiles, tile_labels, slide_label, tile_names = train_iter.next()
        tiles = torch.squeeze(tiles, dim=0)

        slide_label = slide_label.to(device)
        tiles = tiles.to(device)
        slide_probs, tiles_probs, _ = model(tiles)
        slide_loss = slide_criterion(slide_probs, slide_label)
        if len(tile_labels) > 0 and tile_labels[0][0] != -1:
            tile_labels = torch.squeeze(torch.stack(tile_labels), dim=1)
            tile_labels = tile_labels.to(device)
            tile_loss = tile_criterion(tiles_probs, tile_labels)
            loss = alpha * tile_loss + (1 - alpha) * slide_loss
        else:
            loss = slide_loss
        gc.collect()
        # backpropagate and take a step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_dict = {
            'slide_loss': slide_loss.item(),
            'tot_loss': loss.item(),
        }
        if len(tile_labels) > 0 and tile_labels[0] != -1:
            cur_dict['tile_loss'] = tile_loss.item()
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


def trainval(fold, exp_dir, start_epoch, iters, trainval_params, model, optimizer, scheduler,
             checkpointer, train_loader, val_loader, device):
    logger = trainval_stats.StatTracker(log_dir=f"{exp_dir}/")
    slide_criterion = torch.nn.CrossEntropyLoss()
    tile_criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(start_epoch, trainval_params.tot_epochs):
        print(f"Start training for Fold {fold}\t Epoch: {epoch}/{trainval_params.tot_epochs}")
        if epoch == trainval_params.feat_ft and epoch > 0:
            optimizer, scheduler = \
                config_model.config_optimizer(model, epoch, model.hp["arch"], trainval_params.optim,
                                              trainval_params.feat_lr, trainval_params.feat_ft,
                                              trainval_params.lr, trainval_params.wd, trainval_params.train_blocks)

        iters = train_epoch(epoch, iters, model, slide_criterion, tile_criterion, optimizer, train_loader,
                            trainval_params.alpha, trainval_params.log_every, logger, device)
        kappa, loss = val(epoch, model, val_loader, slide_criterion, logger, device)
        checkpointer.update(epoch, iters, kappa)
        scheduler.step(loss)
    return


def val(epoch, model, val_loader, slide_criterion, logger, device):
    model.eval()
    torch.cuda.empty_cache()
    # Quick check of status for log_every steps
    val_stats = trainval_stats.AverageMeterSet()
    time_start = time.time()
    val_iter = iter(val_loader)
    all_labels, all_preds = [], []
    with torch.no_grad():
        for step in range(len(val_loader)):
            tiles, tile_labels, slide_label, tile_names = val_iter.next()
            tiles = torch.squeeze(tiles, dim=0)
            tiles = tiles.to(device)
            slide_probs, tiles_probs, _ = model(tiles)
            slide_label = slide_label.to(device)
            slide_loss = slide_criterion(slide_probs, slide_label)
            cur_dict = {
                'slide_loss': slide_loss.item(),
            }
            val_stats.update_dict(cur_dict, n=1)
            _, predicted = torch.max(slide_probs.data, 1)
            all_labels.append(int(slide_label[0]))
            all_preds.append(int(predicted.item()))
        print(f"Validation step {step}/{len(val_loader)}")
    quadratic_kappa = trainval_stats.compute_kappa(all_preds, all_labels)
    val_stats.update_dict({"kappa": quadratic_kappa}, 1)

    cur_stats = val_stats.pretty_string()
    print(f"Epoch {epoch}, {cur_stats}, time: {time.time() - time_start}")
    sys.stdout.flush()
    logger.record_stats(val_stats.averages(epoch, prefix="val/"))
    torch.cuda.empty_cache()
    return quadratic_kappa, val_stats.avgs['slide_loss']