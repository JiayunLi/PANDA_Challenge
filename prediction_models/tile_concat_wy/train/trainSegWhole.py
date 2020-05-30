## system package
import os, sys, shutil
sys.path.append('../')
from pathlib import Path
from datetime import datetime
from pytz import timezone
import warnings
warnings.filterwarnings("ignore")

## general package
from fastai.vision import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from sklearn.metrics import cohen_kappa_score
## custom package
from input.inputPipeline_seg import PandaPatchDatasetSeg
from input.inputPipeline_seg import dataloader_collte_fn as dataloader_collte_fn_seg
from input.inputPipeline import dataloader_collte_fn as dataloader_collte_fn_val
from input.inputPipeline import PandaPatchDataset, data_transform
# from model.deeplabv3 import *
from model.resnext_seg import *
from utiles.radam import *
from utiles.utils import *

class Train(object):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.segcoef = 0.1
    def train_epoch(self,trainloader, valloader, criterion):
        ## train
        self.model.train()
        train_loss = []
        with torch.set_grad_enabled(True):
            for i, data in enumerate(tqdm(trainloader, desc='trainIter'), start=0):
                # if i >= 5:
                #     break
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, grade = data
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs.cuda())
                # outputs = outputs.squeeze(dim = 1) # for regression
                h,w = labels.shape[-2:]
                labels = labels.view(-1, h, w).long().cuda()
                loss1 = criterion[0](outputs['out'], labels)
                # loss2 = criterion[0](outputs['aux'], labels)
                loss3 = criterion[1](outputs['isup_grade'].squeeze(dim=1), grade.float().cuda())
                # loss = self.segcoef * (loss1 + 0.4 * loss2) + loss3
                loss = self.segcoef * loss1 + loss3
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
        ## val
        model.eval()
        val_label, val_preds = [], []
        with torch.no_grad():
            for i, data in enumerate(tqdm(valloader, desc='valIter'), start=0):
                # if i > 5:
                #     break
                # get the inputs; data is a list of [inputs, labels]
                inputs, grade = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs.cuda())
                # outputs = outputs.squeeze(dim=1)  # for regression
                val_label.append(grade.cpu())
                val_preds.append(outputs['isup_grade'].squeeze(dim=1).cpu())

        val_preds = torch.cat(val_preds, 0).round() # for regression
        val_label = torch.cat(val_label)
        # val_preds = torch.cat(val_preds, 0).round()
        kappa = cohen_kappa_score(val_label.view(-1), val_preds.view(-1), weights='quadratic')
        self.scheduler.step()
        self.segcoef *= 0.1
        return np.mean(train_loss), kappa


def save_checkpoint(state, is_best, fname):
    torch.save(state, '{}_ckpt.pth.tar'.format(fname))
    if is_best:
        # shutil.copyfile('{}_ckpt.pth.tar'.format(fname), '{}_best.pth.tar'.format(fname))
        state = state['state_dict']
        torch.save(state, '{}_best.pth.tar'.format(fname)) ## only save weights for best model

if __name__ == "__main__":
    center = "radboud" ## radboud or karolinska
    fname = "Resnext50_12patch_multitask_whole_30_epoch_" + center
    nfolds = 4
    num_classes = 6 if center == 'radboud' else 3
    bs = 8
    epochs = 30
    csv_file_train = '../input/panda-32x256x256-tiles-data/{}_{}_fold_train.csv'.format(center,nfolds)
    csv_file_val = '../input/panda-32x256x256-tiles-data/wo_mask_val.csv'
    image_dir_train = '../input/panda-32x256x256-tiles-data/train/'
    image_dir_val = '../input/panda-32x256x256-tiles-data/val/'
    mask_dir = '../input/panda-32x256x256-tiles-data/masks/'
    ## image statistics
    mean = torch.tensor([0.90949707, 0.8188697, 0.87795304])
    std = torch.tensor([0.36357649, 0.49984502, 0.40477625])
    # mean = torch.tensor([0.5, 0.5, 0.5])
    # std = torch.tensor([0.5, 0.5, 0.5])
    ## dataset, can fetch data by dataset[idx]
    dataset_train = PandaPatchDatasetSeg(csv_file_train, image_dir_train, mask_dir, [mean, std], N = 12)
    tsfm = data_transform(mean, std)
    dataset_val = PandaPatchDataset(csv_file_val, image_dir_val, transform = tsfm, N = 12)
    ## dataloader
    # crossValData = crossValDataloader(csv_file, dataset, bs)
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True, num_workers=4,
                                                  collate_fn=dataloader_collte_fn_seg, pin_memory=True,
                                                  drop_last=True)
    valloader = torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=4,
                                            collate_fn=dataloader_collte_fn_val, pin_memory=True,
                                            drop_last=True)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    ## tensorboard writer
    writerDir = './runs'
    check_folder_exists(writerDir)
    timeStamp = datetime.now(timezone('US/Pacific')).strftime("%m_%d_%H_%M_%S")
    writer = SummaryWriter('{}/{}_{}'.format(writerDir,fname,timeStamp))
    ## weight saving
    weightsDir = './weights/{}'.format(fname)
    check_folder_exists(weightsDir)

    model = Model().cuda()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 1)
    optimizer = Over9000(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, total_steps = epochs,
                                              pct_start = 0.3, div_factor = 100)
    Training = Train(model, optimizer, scheduler)
    best_kappa = 0
    weightsPath = os.path.join(weightsDir, '{}'.format(fname))
    for epoch in trange(epochs, desc='epoch'):
        train_loss, kappa = Training.train_epoch(trainloader,valloader,
                                                           [criterion1, criterion2])
        tqdm.write("Epoch {}, train loss: {:.4f}, kappa-score: {:.4f}.\n".format(epoch,
                                                                                train_loss,
                                                                                kappa))
        writer.add_scalar('RadboudWholeTrain/train_loss', train_loss, epoch)
        writer.add_scalar('RadboudWholeTrain/kappa_score', kappa, epoch)
        writer.flush()
        ## save the checkpoints and best model
        is_best = kappa > best_kappa
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'kappa': kappa,
            'optimizer': optimizer.state_dict(),
        }, is_best, weightsPath)
        best_kappa = kappa if is_best else best_kappa
    del model
    del optimizer
    del Training
    del scheduler
    writer.close()