from prediction_models.att_mil.mil_models import mil
from prediction_models.att_mil.utils import model_utils, checkpoint_utils
import torch
import torch.optim as optim
import numpy as np
import scipy as sp
from functools import partial
from prediction_models.tile_concat_wy.utiles import radam
from sklearn.metrics import cohen_kappa_score

MAX_LR_RATE = 5


def config_model_optimizer_all(opts, ckp, fold, mil_params, steps_per_epoch):
    checkpointer = checkpoint_utils.Checkpointer(fold, opts.exp_dir)
    if ckp:
        model = checkpointer.restore_model_from_checkpoint(ckp)
        epoch, step = checkpointer.get_current_position()
    else:
        # Start a new model
        epoch, step = 0, 0
        base_encoder, feature_dim = mil.config_encoder(opts.input_size, mil_params["n_tile_classes"],
                                                       opts.arch, opts.pretrained)
        if mil_params['mil_arch'] == "att_batch":
            model = mil.AttMILBatch(base_encoder, opts.pretrained, opts.arch, opts.input_size, feature_dim, mil_params)
        elif mil_params['mil_arch'] == "pool":
            model = mil.PoolMilBatch(base_encoder, opts.pretrained, opts.arch, opts.input_size, feature_dim, mil_params)
        else:
            model = mil.AttMIL(base_encoder, opts.pretrained, opts.arch, opts.input_size, feature_dim, mil_params)

    if opts.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9)
    elif opts.optim == "aug_adam":
        optimizer = radam.Over9000(model.parameters())
    else:
        optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.wd, betas=(0.9, 0.999))
    if mil_params['schedule_type'] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif mil_params['schedule_type'] == "cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opts.lr, steps_per_epoch=steps_per_epoch,
                                                        epochs=opts.epochs, pct_start=0.3, div_factor=100)
    else:
        raise NotImplementedError(f"{mil_params['schedule_type']} Not implemented!!")

    if ckp:
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['scheduler'])
    else:
        checkpointer.track_new_model(model, optimizer, scheduler)
    device = "cpu" if not opts.cuda else "cuda"
    model.to(device)

    return model, optimizer, scheduler, epoch, step, checkpointer


def config_model_optimizer(opts, ckp, fold, mil_params, steps_per_epoch):

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
    if mil_params['schedule_type'] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif mil_params['schedule_type'] == "cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opts.lr, steps_per_epoch=steps_per_epoch,
                                                        epochs=opts.epochs)
    else:
        raise NotImplementedError(f"{mil_params['schedule_type']} Not implemented!!")

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


class OptimizedRounder:
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_['x']