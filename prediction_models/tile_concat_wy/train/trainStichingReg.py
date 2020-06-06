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
from collections import OrderedDict
## custom package
from input.inputPipeline_stiching_reg import *
# from model.evnet import *
from model.resnext_ssl_stiching import *
from utiles.radam import *
from utiles.utils import *

class Train(object):
    def __init__(self, model, optimizer, scheduler, GLS = False, mltLoss = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mltLoss = mltLoss
        self.GLS = GLS

    def train_epoch(self,trainloader, criterion):
        ## train
        self.model.train()
        train_loss = []
        bar = tqdm(trainloader, desc='trainIter')
        result = OrderedDict()
        for i, data in enumerate(bar, start=0):
            # if i >= 5:
            #     break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['img'], data['isup_grade']
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs.cuda())
            outputs_main = outputs['out'].squeeze(dim = 1) # for regression
            # outputs_aux = outputs['aux'].squeeze(dim=1)  # for regression
            loss1 = criterion(outputs_main, labels.float().cuda())
            if self.GLS:
                primary_gls, secondary_gls = data['primary_gls'], data['secondary_gls']
                outputs_prim = outputs['primary_gls'].squeeze(dim=1)
                outputs_sec = outputs['secondary_gls'].squeeze(dim=1)
                loss2 = criterion(outputs_prim, primary_gls.float().cuda())
                loss3 = criterion(outputs_sec, secondary_gls.float().cuda())
                if self.mltLoss is not None:
                    loss = self.mltLoss(torch.Tensor([loss1, loss2, loss3]).cuda())
                else:
                    loss = loss1 + 0.5 * (loss2 + 0.5 * loss3)
            else:
                loss = loss1
            # loss2 = criterion(outputs_aux, labels.float().cuda())
            # loss = loss1 + 0.4 * loss2
            train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            bar.set_description('loss: %.5f, smth: %.5f' % (loss.item(), smooth_loss))
        result['train_loss'] = np.mean(train_loss)
        return result

    def val_epoch(self, valloader, criterion):   ## val
        model.eval()
        val_loss, val_label, val_preds, val_provider = [], [], [], []
        result = OrderedDict()
        with torch.no_grad():
            for i, data in enumerate(tqdm(valloader, desc='valIter'), start=0):
                # if i > 50:
                #     break
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, provider = data['img'], data['isup_grade'], data['datacenter']
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs.cuda())
                outputs_main = outputs['out'].squeeze(dim=1)
                # outputs_aux = outputs['aux'].squeeze(dim=1)  # for regression
                loss1 = criterion(outputs_main, labels.float().cuda())
                # loss2 = criterion(outputs_aux, labels.float().cuda())
                # loss = loss1 + 0.4 * loss2
                loss = loss1
                # print("output_main", outputs_main.shape)
                # print("labels", labels.shape)
                val_loss.append(loss.item())
                val_label.append(labels.cpu())
                val_preds.append(outputs_main.cpu())
                val_provider += provider
                # print("postval_label", labels.sum(1))
                # print("postval_label", outputs_main.sigmoid().sum(1).round())

        val_label = torch.cat(val_label)
        val_preds = torch.cat(val_preds, 0).round()
        # print(val_label.shape, val_preds.shape)
        self.scheduler.step()
        index_r = [i for i, x in enumerate(val_provider) if x == "radboud"]
        index_k = [i for i, x in enumerate(val_provider) if x == "karolinska"]
        kappa = cohen_kappa_score(val_label, val_preds, weights='quadratic')
        kappa_r = cohen_kappa_score(val_label[index_r], val_preds[index_r], weights='quadratic')
        kappa_k = cohen_kappa_score(val_label[index_k], val_preds[index_k], weights='quadratic')
        result['val_loss'] = np.mean(val_loss)
        result['kappa'] = kappa
        result['kappa_r'] = kappa_r
        result['kappa_k'] = kappa_k
        # result['val_label'] = val_label
        # result['val_preds'] = val_preds
        return result


def save_checkpoint(state, is_best, fname):
    torch.save(state, '{}_ckpt.pth.tar'.format(fname))
    if is_best:
        # shutil.copyfile('{}_ckpt.pth.tar'.format(fname), '{}_best.pth.tar'.format(fname))
        state = state['state_dict']
        torch.save(state, '{}_best.pth.tar'.format(fname)) ## only save weights for best model

if __name__ == "__main__":
    fname = "Resnext50_medreso_16patch_overlook_cosine_reg_gls_mltloss"
    nfolds = 4
    bs = 6
    enet_type = 'efficientnet-b0'
    epochs = 30
    GLS = True
    csv_file = '../input/panda-16x128x128-tiles-data/{}_fold_whole_train.csv'.format(nfolds)
    image_dir = '../input/panda-36x256x256-tiles-data/train/'

    ## image transformation
    tsfm = data_transform()
    # tsfm = None
    ## dataset, can fetch data by dataset[idx]
    dataset = PandaPatchDataset(csv_file, image_dir, 256, transform=tsfm, N = 16, rand=True)
    ## dataloader
    crossValData = crossValDataloader(csv_file, dataset, bs)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    ## tensorboard writer
    writerDir = './runs'
    check_folder_exists(writerDir)
    timeStamp = datetime.now(timezone('US/Pacific')).strftime("%m_%d_%H_%M_%S")
    writer = SummaryWriter('{}/{}_{}'.format(writerDir,fname,timeStamp))
    ## weight saving
    weightsDir = './weights/{}'.format(fname)
    check_folder_exists(weightsDir)
    for fold in range(nfolds):
        trainloader, valloader = crossValData(fold)
        # model = Model(enet_type, out_dim=5).cuda()
        model = Model(n = 1, GleasonScore = GLS).cuda()
        optimizer = Over9000(model.parameters())
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, total_steps = epochs,
                                                  pct_start = 0.3, div_factor = 100)
        # optimizer = optim.Adam(model.parameters(), lr=0.00003)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        if GLS:
            mltLoss = MultiTaskLoss(3).cuda()
            Training = Train(model, optimizer, scheduler, GLS=GLS, mltLoss=mltLoss)
        else:
            Training = Train(model, optimizer, scheduler, GLS=GLS)
        best_kappa = 0
        weightsPath = os.path.join(weightsDir, '{}_{}'.format(fname, fold))
        for epoch in trange(epochs, desc='epoch'):
            train = Training.train_epoch(trainloader,criterion)
            writer.add_scalar('Fold:{}/train_loss'.format(fold), train['train_loss'], epoch)
            val = Training.val_epoch(valloader, criterion)
            writer.add_scalar('Fold:{}/val_loss'.format(fold), val['val_loss'], epoch)
            writer.add_scalar('Fold:{}/kappa_score'.format(fold), val['kappa'], epoch)
            writer.add_scalar('Fold:{}/kappa_score_r'.format(fold), val['kappa_r'], epoch)
            writer.add_scalar('Fold:{}/kappa_score_k'.format(fold), val['kappa_k'], epoch)
            writer.flush()
            # print(val['val_preds'], val['val_label'])
            # print(val['val_preds'].shape, val['val_label'].shape, val['kappa'])
            tqdm.write("Epoch {}, train loss: {:.4f}, val loss: {:.4f}, kappa-score: {:.4f}.\n".format(epoch,
                                                                                               train['train_loss'],
                                                                                               val['val_loss'],
                                                                                               val['kappa']))
            ## save the checkpoints and best model
            is_best = val['kappa'] > best_kappa
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'kappa': val['kappa'],
                'optimizer': optimizer.state_dict(),
            }, is_best, weightsPath)
            best_kappa = val['kappa'] if is_best else best_kappa
        del model
        del optimizer
        del Training
        del scheduler
    writer.close()