from prediction_models.att_mil.mil_models import mil
from prediction_models.att_mil.utils import model_utils, checkpoint_utils
import torch
import torch.optim as optim


def config_model_optimizer(opts, ckp, fold, mil_params):
    checkpointer = checkpoint_utils.Checkpointer(fold, opts.exp_dir)
    if ckp:
        model = checkpointer.restore_model_from_checkpoint(ckp)
        epoch, step = checkpointer.get_current_position()
    else:
        # Start a new model
        epoch, step = 0, 0
        base_encoder, feature_dim = mil.config_encoder(opts.input_size, mil_params["n_tile_classes"],
                                                       opts.arch, opts.pretrained)
        model = mil.AttMIL(base_encoder, opts.pretrained, opts.arch, opts.input_size, feature_dim, mil_params)

    train_encoder_params = \
        model_utils.set_train_parameters_tile_encoder(model, opts.arch, opts.optim, opts.feat_lr,
                                                      opts.lr, opts.wd, train_layer=opts.train_blocks,
                                                      fix=opts.feat_ft < epoch)

    train_mil_params = model_utils.set_train_parameters_mil(model, opts.lr, opts.optim, opts.wd)
    train_params = []
    for cur_param in train_encoder_params:
        train_params.append(cur_param)
    for cur_param in train_mil_params:
        train_params.append(cur_param)

    if opts.optim == 'sgd':
        optimizer = optim.SGD(train_params)
    else:
        optimizer = optim.Adam(train_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    if ckp:
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['scheduler'])
    else:
        checkpointer.track_new_model(model, optimizer, scheduler)
    device = "cpu" if not opts.cuda else "cuda"
    model.to(device)

    return model, optimizer, scheduler, epoch, step, checkpointer


def config_optimizer(model, epoch, arch, optim_type, feat_lr, feat_ft, lr, wd, train_blocks):
    train_params = \
        model_utils.set_train_parameters_tile_encoder(model, arch, optim_type, feat_lr,
                                                      lr, wd, train_layer=train_blocks,
                                                      fix=feat_ft <= epoch)
    train_params += model_utils.set_train_parameters_mil(model, lr, optim_type, wd)

    if optim_type == 'sgd':
        optimizer = optim.SGD(train_params)
    else:
        optimizer = optim.Adam(train_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    return optimizer, scheduler
