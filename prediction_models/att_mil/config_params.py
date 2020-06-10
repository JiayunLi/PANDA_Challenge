
class DatasetParams:
    def __init__(self, im_size, input_size, info_dir, data_dir, cache_dir, exp_dir, dataset, normalized,
                 loss_type, num_channels=3, top_n=-1):
        self.im_size = im_size
        self.input_size = input_size
        self.num_channels = num_channels
        self.info_dir = info_dir
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.dataset = dataset
        self.exp_dir = exp_dir
        self.normalized = normalized
        self.top_n = top_n
        self.loss_type = loss_type


class SelectedDatasetParams:
    def __init__(self, level, lowest_im_size, input_size, info_dir, data_dir, cache_dir, exp_dir, dataset, normalized,
                 loss_type, num_channels=3, top_n=-1):
        self.level = level
        self.lowest_im_size = lowest_im_size
        self.input_size = input_size
        self.num_channels = num_channels
        self.info_dir = info_dir
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.dataset = dataset
        self.exp_dir = exp_dir
        self.normalized = normalized
        self.top_n = top_n
        self.loss_type = loss_type


class DatasetParamsMulti:
    def __init__(self, im_size_low, im_size_high, input_size, info_dir, data_dir_low, data_dir_high, cache_dir,
                 exp_dir, normalized, num_channels=3, top_n_low=-1, top_n_high=-1):
        self.im_size_low, self.im_size_high = im_size_low, im_size_high
        self.input_size = input_size
        self.num_channels = num_channels
        self.info_dir = info_dir
        self.data_dir_low, self.data_dir_high = data_dir_low, data_dir_high
        self.cache_dir = cache_dir
        self.exp_dir = exp_dir
        self.normalized = normalized
        self.top_n_low = top_n_low
        self.top_n_high = top_n_high
        self.dataset = "multi"


class DatasetTest:
    def __init__(self, im_size, input_size, num_classes, overlap, ts_thres, num_channels=3):
        self.im_size = im_size
        self.input_size = input_size
        self.dw_rate = im_size / input_size
        self.num_classes = num_classes
        self.overlap = overlap
        self.ts_thres = ts_thres
        self.test_file = self.test_file
        self.test_slides_dir = self.test_slides_dir
        self.num_channels = num_channels
        # self.top_n = top_n


class TrainvalParams:
    def __init__(self, lr, feat_lr, wd, train_blocks, optim, tot_epochs, feat_ft, log_every, alpha, loss_type,
                 cls_weighted, schedule_type, slide_binary, tile_binary, tile_ft, batch_size, num_workers, mil_arch):
        self.lr = lr
        self.feat_lr = feat_lr
        self.wd = wd
        self.train_blocks = train_blocks
        self.optim = optim
        self.tot_epochs = tot_epochs
        self.feat_ft = feat_ft
        self.log_every = log_every
        self.alpha = alpha
        self.loss_type = loss_type
        self.cls_weighted = cls_weighted
        self.schedule_type = schedule_type
        self.slide_binary = slide_binary
        self.tile_binary = tile_binary
        self.tile_ft = tile_ft
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mil_arch = mil_arch


def set_mil_params(mil_in_feat_size, instance_embed_dim, bag_embed_dim, bag_hidden_dim, slide_n_classes,
                   tile_classes, loss_type, schedule_type, mil_arch):
    if loss_type == "mse":
        slide_n_classes = 1
        tile_classes = 1
    elif loss_type == "bce":
        slide_n_classes = 5
    params = {
        "mil_in_feat_size": mil_in_feat_size,
        "instance_embed_dim": instance_embed_dim,
        "bag_embed_dim": bag_embed_dim,
        "bag_hidden_dim": bag_hidden_dim,
        "n_slide_classes": slide_n_classes,
        "n_tile_classes": tile_classes,
        "schedule_type": schedule_type,
        "mil_arch": mil_arch
    }
    return params


class TrainvalParamsMultiTask:
    def __init__(self, lr, feat_lr, wd, train_blocks, optim, tot_epochs, log_every, alpha, loss_type,
                 cls_weighted, schedule_type, batch_size, num_workers, mil_arch):
        self.lr = lr
        self.feat_lr = feat_lr
        self.wd = wd
        self.train_blocks = train_blocks
        self.optim = optim
        self.tot_epochs = tot_epochs
        self.log_every = log_every
        self.alpha = alpha
        self.loss_type = loss_type
        self.cls_weighted = cls_weighted
        self.schedule_type = schedule_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mil_arch = mil_arch