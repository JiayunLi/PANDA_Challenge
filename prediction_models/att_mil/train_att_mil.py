import torch
from prediction_models.att_mil.utils import trainval_stats, model_utils
from prediction_models.att_mil.mil_models import config_model
import time
import sys
import gc
from prediction_models.att_mil.datasets import config_dataset
from prediction_models.att_mil import train_att_mil_batch as batch_train
from prediction_models.att_mil import train_att_mil_multi as multi_train


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

        if loss_type == "mse":
            slide_label = slide_label.float()
        slide_label = slide_label.to(device)
        tiles = tiles.to(device)
        slide_probs, tiles_probs, _ = model(tiles)
        if loss_type == "mse":
            slide_loss = slide_criterion(slide_probs.view(-1), slide_label)
        else:
            slide_loss = slide_criterion(slide_probs, slide_label)
        if len(tile_labels) > 0 and tile_labels[0][0] != -1:
            tile_labels = torch.squeeze(torch.stack(tile_labels), dim=1)
            if loss_type == "mse":
                tile_labels = tile_labels.float().to(device)
                tile_loss = tile_criterion(tiles_probs.view(-1), tile_labels)
            else:
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
            print(f"Fold {fold}, Epoch {epoch}, Updates {step}/{len(train_loader)}, {cur_stats}, {spu:.4f}/update")
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
    elif loss_type == 'bce':
        tile_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Criterion not implemented {loss_type}")
    return tile_criterion


def trainval(fold, exp_dir, start_epoch, iters, trainval_params, dataset_params, model, optimizer, scheduler,
             checkpointer, train_loader, train_data, val_loader, device):
    logger = trainval_stats.StatTracker(log_dir=f"{exp_dir}/")
    # tile_label_weights, slide_label_weights = \
    #     model_utils.compute_class_frequency(train_data.slides_df, train_data.tile_labels, binary_only=False)
    # print("Tile weights")
    # print(tile_label_weights)

    if trainval_params.loss_type == "ce" and trainval_params.cls_weighted:
        tile_label_weights, slide_label_weights = \
            model_utils.compute_class_frequency(train_data.slides_df, train_data.tile_labels, binary_only=False)
        print(tile_label_weights)
        print(slide_label_weights)
        tile_label_weights = torch.FloatTensor(tile_label_weights).to(device)
        slide_label_weights = torch.FloatTensor(slide_label_weights).to(device)
    else:
        tile_label_weights, slide_label_weights = None, None
    slide_criterion = configure_criterion(trainval_params.loss_type, trainval_params.cls_weighted,
                                          trainval_params.slide_binary, slide_label_weights)
    # tile_criterion = configure_criterion("ce", True,
    #                                      trainval_params.tile_binary, tile_label_weights)
    tile_criterion = configure_criterion(trainval_params.loss_type, trainval_params.cls_weighted,
                                         trainval_params.tile_binary, tile_label_weights)
    # if start_epoch < trainval_params.tile_ft:
    #     # Train network with tile-level only. Here we need to make sure all data comes from Radbound dataset.
    #     tile_alpha = 1
    #     if trainval_params.schedule_type == "plateau":
    #         tile_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #     else:
    #         tile_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=trainval_params.lr,
    #                                                              total_steps=trainval_params.tile_ft,
    #                                                              pct_start=0.0, div_factor=100)
    #     tiles_train_loader, _ = \
    #         config_dataset.build_dataset_loader(trainval_params.batch_size, trainval_params.num_workers,
    #                                             dataset_params, split="train", phase="train_tiles", fold=fold,
    #                                             mil_arch=trainval_params.mil_arch, has_drop_rate=0)
    #     tiles_val_loader, _ = \
    #         config_dataset.build_dataset_loader(trainval_params.batch_size, trainval_params.num_workers,
    #                                             dataset_params, split="val", phase="val_tiles", fold=fold,
    #                                             mil_arch=trainval_params.mil_arch)
    #     for epoch in range(start_epoch, trainval_params.tile_ft):
    #         print("Start training for tiles")
    #         if model.mil_params['mil_arch'] in {"pool_simple", "pool", 'att_batch'}:
    #             iters = batch_train.train_epoch(epoch, fold, iters, model, slide_criterion, tile_criterion, optimizer,
    #                                             tiles_train_loader, tile_alpha, trainval_params.loss_type,
    #                                             trainval_params.log_every, logger, device, tile_scheduler)
    #             kappa, loss = batch_train.val(epoch, fold, model, tiles_val_loader, slide_criterion, tile_criterion,
    #                                           tile_alpha, trainval_params.loss_type,
    #                                           logger, trainval_params.slide_binary, device)
    #         else:
    #             iters = train_epoch(epoch, fold, iters, model, slide_criterion, tile_criterion, optimizer,
    #                                 tiles_train_loader, tile_alpha, trainval_params.loss_type,
    #                                 trainval_params.log_every, logger, device)
    #             kappa, loss = val(epoch, fold, model, tiles_val_loader, slide_criterion, trainval_params.loss_type,
    #                               logger, trainval_params.slide_binary, device)
    #         checkpointer.update(epoch, iters, 0)
    #         if trainval_params.schedule_type == "plateau":
    #             print("Take one Plateau step")
    #             tile_scheduler.step(loss)
    #         elif trainval_params.schedule_type == "cycle":
    #             print("Take one cycle step")
    #             # print("Take step per batch")
    #             tile_scheduler.step()
    #         else:
    #             raise NotImplementedError(f"{trainval_params.schedule_type} Not implemented!!")
    #     start_epoch = trainval_params.tile_ft
    # Start with slide-level loss only training
    alpha = 0
    for epoch in range(start_epoch, trainval_params.tot_epochs):
        print(f"Start training for Fold {fold}\t Epoch: {epoch}/{trainval_params.tot_epochs}")
        if epoch == trainval_params.feat_ft and epoch > 0:
            optimizer, scheduler = \
                config_model.config_optimizer(model, epoch, model.hp["arch"], trainval_params.optim,
                                              trainval_params.feat_lr, trainval_params.feat_ft,
                                              trainval_params.lr, trainval_params.wd, trainval_params.train_blocks)
        if model.mil_params['mil_arch'] in {"pool_simple", "pool", 'att_batch', }:
            iters = batch_train.train_epoch(epoch, fold, iters, model, slide_criterion, tile_criterion, optimizer,
                                            train_loader, alpha, trainval_params.loss_type,
                                            trainval_params.log_every, logger, device, scheduler)
            kappa, loss = batch_train.val(epoch, fold, model, val_loader, slide_criterion, tile_criterion,
                                          alpha, trainval_params.loss_type, logger, trainval_params.slide_binary,
                                          device)
        elif model.mil_params['mil_arch'] in {'att_batch_multi'}:
            iters = multi_train.train_epoch(epoch, fold, iters, model, slide_criterion, optimizer,
                                            train_loader, alpha, trainval_params.loss_type,
                                            trainval_params.log_every, logger, device, scheduler)
            kappa, loss = multi_train.val(epoch, fold, model, val_loader, slide_criterion, alpha,
                                          trainval_params.loss_type, logger, trainval_params.slide_binary, device)
        else:
            iters = train_epoch(epoch, fold, iters, model, slide_criterion, tile_criterion, optimizer, train_loader,
                                alpha, trainval_params.loss_type, trainval_params.log_every,
                                logger, device)
            kappa, loss = val(epoch, fold, model, val_loader, slide_criterion, trainval_params.loss_type,
                              logger, trainval_params.slide_binary, device)
        checkpointer.update(epoch, iters, kappa)
        if trainval_params.schedule_type == "plateau":
            print("Take one Plateau step")
            scheduler.step(loss)
        elif trainval_params.schedule_type in {"cycle", "cosine"}:
            print("Take one cycle step")
            # print("Take step per batch")
            scheduler.step()
        else:
            raise NotImplementedError(f"{trainval_params.schedule_type} Not implemented!!")
        # if trainval_params.smooth_alpha:
        #     alpha -= alpha_reduce_rate
        #     alpha = max(0, alpha)

    return


def val(epoch, fold, model, val_loader, slide_criterion, loss_type, logger, slide_binary, device):
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
                normalized_prob = torch.nn.Sigmoid(slide_probs)
                predicted = int(torch.ge(normalized_prob, 0.5).cpu().item())
            elif loss_type == "mse":
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