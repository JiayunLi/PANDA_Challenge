import os
import torch
import shutil
import pickle


def load_ckp(load_best, ckp_dir, device):
    if load_best:
        ckp_path = f"{ckp_dir}/checkpoint_best.pth"
    else:
        ckp_path = f"{ckp_dir}/checkpoint.pth"

    if device == 'cuda':
        ckp = torch.load(ckp_path)
    else:
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
    return ckp


class Checkpointer:
    def __init__(self, fold, exp_dir):
        # set output dir will this checkpoint will save itself
        self.output_dir = f"{exp_dir}/{fold}/"
        self.exp_dir = exp_dir
        self.epoch = 0
        self.step = 0
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.fold = fold
        self.best_score = 0

    def track_new_model(self, model, optimizer, scheduler):
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def restore_model_from_checkpoint(self, ckp):

        self.epoch = ckp['cursor']['epoch'] + 1
        self.step = ckp['cursor']['step'] + 1
        self.fold = ckp['cursor']['fold']
        self.output_dir = f"{self.exp_dir}/{self.fold}/"

        hp = ckp['hyperparams']
        params = ckp['model']
        self.best_score = ckp['best_score']

        from prediction_models.att_mil.mil_models import mil

        base_encoder, feature_dim = \
            mil.config_encoder(hp['input_size'], hp['mil_params']['n_tile_classes'],
                               hp['encoder_arch'], False)
        if hp['mil_params'].aug_mil:
            self.model = mil.AttMILBatch(base_encoder, hp['pretrained'], hp['encoder_arch'], hp['input_size'],
                                        feature_dim, hp['mil_params'])
        else:
            self.model = mil.AttMIL(base_encoder, hp['pretrained'], hp['encoder_arch'], hp['input_size'],
                                    feature_dim, hp['mil_params'])
        self.model.load_state_dict(params)

        print(f"***** CHECKPOINTING *****\n Model restored from checkpoint.\n MIL training epoch {self.epoch}\n")

        return self.model

    def _get_state(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'hyperparams': self.model.hp,
            'best_score': self.best_score,
            'cursor': {
                'epoch': self.epoch,
                'step': self.step,
                'fold': self.fold,
            }
        }

    def _save_cpt(self, cur_score):
        cpt_path = f"{self.output_dir}/checkpoint.pth"
        # write updated checkpoint to the desired path
        torch.save(self._get_state(), cpt_path)
        if cur_score > self.best_score:
            self.best_score = cur_score
            shutil.copyfile(cpt_path, f"{self.output_dir}/checkpoint_best.pth")
        return

    def update(self, epoch, step, cur_score):
        self.epoch = epoch
        self.step = step
        self._save_cpt(cur_score)

    def get_current_position(self):
        return self.epoch, self.step


def load_options(ckp_dir, load_best, data_dir, cuda, num_workers):
    ckp = load_ckp(load_best, ckp_dir, cuda)
    fold = ckp['cursor']['fold']
    options_dir = ckp_dir.replace(f"{fold}/", "")
    model_opts = pickle.load(open(f"{options_dir}/options.pkl", "rb"))
    model_opts.cuda = cuda
    model_opts.num_workers = num_workers
    model_opts.data_dir = data_dir
    model_opts.start_fold = fold
    model_opts.load_best = load_best

    return model_opts, ckp
