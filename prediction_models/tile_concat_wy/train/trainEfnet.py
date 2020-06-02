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
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from tqdm import trange, tqdm
from sklearn.metrics import cohen_kappa_score
## custom package
from input.inputPipeline_org import *
# from model.deeplabv3_finetune import *
from model.evnet import *
from utiles.radam import *
from utiles.utils import *

class Train(object):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
    def train_epoch(self,trainloader, valloader, criterion):
        ## train
        self.model.train()
        train_loss = []
        for i, data in enumerate(tqdm(trainloader, desc='trainIter'), start=0):
            # if i >= 5:
            #     break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs.cuda())
            outputs_main = outputs['out'] # for regression
            # outputs_aux = outputs['aux'].squeeze(dim=1)  # for regression
            loss1 = criterion(outputs_main, labels.float().cuda())
            # loss2 = criterion(outputs_aux, labels.float().cuda())
            # loss = loss1 + 0.4 * loss2
            loss = loss1
            train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
        ## val
        model.eval()
        val_loss, val_label, val_preds = [], [], []
        with torch.no_grad():
            for i, data in enumerate(tqdm(valloader, desc='valIter'), start=0):
                # if i > 5:
                #     break
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs.cuda())
                outputs_main = outputs['out']
                # outputs_aux = outputs['aux'].squeeze(dim=1)  # for regression
                loss1 = criterion(outputs_main, labels.float().cuda())
                # loss2 = criterion(outputs_aux, labels.float().cuda())
                # loss = loss1 + 0.4 * loss2
                loss = loss1
                val_loss.append(loss.item())
                val_label.append(labels.sum(1).cpu())
                val_preds.append(outputs['out'].sum(1).cpu())
        # val_preds = torch.argmax(torch.cat(val_preds, 0), 1) # for classification
        val_label = torch.cat(val_label, 0)
        val_preds = torch.cat(val_preds, 0).round()
        kappa = cohen_kappa_score(val_label, val_preds, weights='quadratic')
        self.scheduler.step()
        return np.mean(train_loss), np.mean(val_loss), kappa


def save_checkpoint(state, is_best, fname):
    torch.save(state, '{}_ckpt.pth.tar'.format(fname))
    if is_best:
        # shutil.copyfile('{}_ckpt.pth.tar'.format(fname), '{}_best.pth.tar'.format(fname))
        state = state['state_dict']
        torch.save(state, '{}_best.pth.tar'.format(fname)) ## only save weights for best model

if __name__ == "__main__":
    # fname = "Dlv3_ft_reg_medreso_12patch_aux"
    fname = "Evnet_medreso_36patch"
    nfolds = 4
    bs = 6
    enet_type = 'efficientnet-b0'
    epochs = 30
    csv_file = '../input/panda-16x128x128-tiles-data/{}_fold_whole_train.csv'.format(nfolds)
    image_dir = '../input/prostate-cancer-grade-assessment/train_images/'

    ## image transformation
    # tsfm = data_transform()
    tsfm = None
    ## dataset, can fetch data by dataset[idx]
    dataset = PandaPatchDataset(csv_file, image_dir, 256, transform=tsfm, N = 36, rand=True)
    ## dataloader
    crossValData = crossValDataloader(csv_file, dataset, bs)
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()

    ## tensorboard writer
    writerDir = './runs'
    check_folder_exists(writerDir)
    timeStamp = datetime.now(timezone('US/Pacific')).strftime("%m_%d_%H_%M_%S")
    writer = SummaryWriter('{}/{}_{}'.format(writerDir,fname,timeStamp))
    ## weight saving
    weightsDir = './weights/{}'.format(fname)
    check_folder_exists(weightsDir)
    # for fold in trange(nfolds - 1, nfolds, desc='fold'):
    for fold in range(nfolds):
        trainloader, valloader = crossValData(fold)
        model = Model(enet_type, out_dim=5).cuda()
        optimizer = Over9000(model.parameters())
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, total_steps = epochs,
                                                  pct_start = 0.3, div_factor = 100)
        Training = Train(model, optimizer, scheduler)
        best_kappa = 0
        weightsPath = os.path.join(weightsDir, '{}_{}'.format(fname, fold))
        for epoch in trange(epochs, desc='epoch'):
            train_loss, val_loss, kappa = Training.train_epoch(trainloader,valloader,criterion)
            tqdm.write("Epoch {}, train loss: {:.4f}, val loss: {:.4f}, kappa-score: {:.4f}.\n".format(epoch,
                                                                                               train_loss,
                                                                                               val_loss,
                                                                                               kappa))
            writer.add_scalar('Fold:{}/train_loss'.format(fold), train_loss, epoch)
            writer.add_scalar('Fold:{}/val_loss'.format(fold), val_loss, epoch)
            writer.add_scalar('Fold:{}/kappa_score'.format(fold), kappa, epoch)
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