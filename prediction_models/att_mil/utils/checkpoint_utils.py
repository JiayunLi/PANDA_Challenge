import os
import torch
import shutil


class Checkpointer:
    def __init__(self, fold, output_dir=None):
        # set output dir will this checkpoint will save itself
        self.output_dir = output_dir
        self.epoch = 0
        self.step = 0
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.fold = fold
        self.best_score = 0

    def track_new_model(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def restore_model_from_checkpoint(self, cpt_path, device='cpu'):
        if device == 'cuda':
            ckp = torch.load(cpt_path)
        else:
            ckp = torch.load(cpt_path, map_location=lambda storage, loc: storage)

        self.epoch = ckp['cursor']['epoch'] + 1
        self.step = ckp['cursor']['step'] + 1
        self.fold = ckp['cursor']['fold']

        hp = ckp['hyperparams']
        params = ckp['model']
        self.best_score = ckp['best_score']

        from prediction_models.att_mil.mil_models import mil
        base_encoder = mil.config_encoder(hp['input_size'], hp['n_tile_classes'], hp['encoder_arch'], hp['pretrained'])
        self.model = mil.AttMIL(base_encoder, hp['pretrained'], hp['encoder_arch'], hp['input_size'],
                                hp['n_tile_classes'], hp['feature_dim'], hp['mil_params'])
        self.model.load_state_dict(params)

        self.optimizer.load_state_dict(ckp['optimizer'])
        self.scheduler.load_state_dict(ckp['scheduler'])

        print(f"***** CHECKPOINTING *****\n Model restored from checkpoint.\n MIL training epoch {epoch}\n")
        return self.model, self.optimizer, self.scheduler

    def _get_state(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'hyperparams': self.model.hp,
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
