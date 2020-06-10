import torch
import torch.nn as nn

# import Models.layers as model_layers
# from Models import mix_precision
# from Models import helper
from prediction_models.att_mil.mil_models import costs


class AttMILSemi(nn.Module):
    def __init__(self, base_encoder, pretrained, arch, input_size, feature_dim, mil_params, device):
        super(AttMILSemi, self).__init__()
        self.tile_encoder = base_encoder
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self.hp = {"input_size": input_size, "encoder_arch": arch,
                   "feature_dim": feature_dim, "pretrained": pretrained,
                   "mil_params": mil_params, "arch": arch}
        self.mil_params = mil_params

        self.instance_embed = self._config_instance_embed()

        self.embed_bag_feat, self.attention = self._config_attention()
        self.slide_classifier = self._config_classifier()
        self._initialize()

        self.tasks, rkhs_1 = self._config_encoder()
        # rkhs_1, _ = self.encode(dummy_batch)
        self.rkhs_dim = rkhs_1.size(1)

        # configure hacky multi-gpu module for infomax costs
        self.tclip, self.device = self.mil_params['tclip'], device
        self.g2l_loss = self._config_nce()

        # # gather lists of self-supervised and classifier modules
        # # self.info_modules = [self.encoder.module, self.g2l_loss] # use this for multi-gpu
        # self.info_modules = [self.encoder, self.g2l_loss]
        # self.class_modules = [self.evaluator]

    def _config_nce(self):
        g2l_loss = costs.LossNCE(tclip=self.tclip, device=self.device)
        return g2l_loss

    def _config_encoder(self):
        tasks = ('1t5',)
        hp = self.hyperparams
        dummy_batch = torch.zeros((2, 3, hp["encoder_size"], hp["encoder_size"]))
        encoder = Encoder(dummy_batch, num_channels=3, ndf=hp["ndf"], n_rkhs=hp["n_rkhs"], n_depth=hp["n_depth"],
                          encoder_size=hp["encoder_size"], use_bn=hp["use_bn"])
        self.encoder = encoder
        rkhs_1, _ = self.encode(dummy_batch)
        return tasks, rkhs_1

    def init_weights(self, init_scale=1.):
        # self.encoder.module.init_weights(init_scale) # for multi-gpu
        self.encoder.init_weights(init_scale)

    def encode(self, x, no_grad=True, use_eval=False):
        '''
        Encode the images in x, with or without grads detached.
        '''
        if use_eval:
            self.eval()
        x = mix_precision.maybe_half(x)
        if no_grad:
            with torch.no_grad():
                rkhs_1, rkhs_5 = self.encoder(x)
        else:
            rkhs_1, rkhs_5 = self.encoder(x)
        if use_eval:
            self.train()
        return mix_precision.maybe_half(rkhs_1), mix_precision.maybe_half(rkhs_5)

    def reset_evaluator(self, n_classes=None):
        '''
        Reset the evaluator module, e.g. to apply encoder on new data.
        - evaluator is reset to have n_classes classes (if given)
        '''
        dim_1 = self.evaluator.dim_1
        if n_classes is None:
            n_classes = self.evaluator.n_classes
        self.evaluator = Evaluator(n_classes, dim_1=dim_1)
        self.class_modules = [self.evaluator]
        return self.evaluator

    def forward(self, x1, x2, class_only=False, no_cls=False):
        '''
        Input:
          x1 : images from which to extract features -- x1 ~ A(x)
          x2 : images from which to extract features -- x2 ~ A(x)
          class_only : whether we want all outputs for infomax training
        Output:
          res_dict : various outputs depending on the task
        '''
        # dict for returning various values
        res_dict = {}
        if class_only:
            # shortcut to encode one image and evaluate classifier
            rkhs_1, _, = self.encode(x1, no_grad=True)
            lgt_glb_mlp, lgt_glb_lin = self.evaluator(rkhs_1)
            res_dict['class'] = [lgt_glb_mlp, lgt_glb_lin]
            res_dict['rkhs_glb'] = helper.flatten(rkhs_1)
            return res_dict

        # run augmented image pairs through the encoder
        r1_x1, r5_x1 = self.encoder(x1)
        r1_x2, r5_x2 = self.encoder(x2)

        # compute NCE infomax objective at multiple scales
        loss_1t5, lgt_reg = self.g2l_loss(r1_x1, r5_x1, r1_x2, r5_x2)
        res_dict['g2l_1t5'] = loss_1t5
        res_dict['lgt_reg'] = lgt_reg
        # grab global features for use elsewhere
        res_dict['rkhs_glb'] = helper.flatten(r1_x1)

        # compute classifier logits for online eval during infomax training
        # - we do this for both images in each augmented pair...
        if not no_cls:
            lgt_glb_mlp, lgt_glb_lin = self.evaluator(ftr_1=torch.cat([r1_x1, r1_x2]))
            res_dict['class'] = [lgt_glb_mlp, lgt_glb_lin]
        return res_dict