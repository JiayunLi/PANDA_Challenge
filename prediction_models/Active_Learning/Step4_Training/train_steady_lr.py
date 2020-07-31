## system package
import os, sys
sys.path.append('../')
from datetime import datetime
from pytz import timezone
import warnings
warnings.filterwarnings("ignore")

## general package
from fastai.vision import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from collections import OrderedDict
import argparse
import torch.distributed as dist
## custom package
from input.inputPipeline_stiching import data_transform, PandaPatchDataset, crossValDataloader
from utiles.radam import *
from utiles.utils import *
from utiles.flatten_cosanneal import *
from Model.efficientnet.model import EfficientNet as Model

class Train(object):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_epoch(self,trainloader, criterion):
        ## train
        self.model.train()
        train_loss, train_sample_loss, train_idx = [], [], []
        bar = tqdm(trainloader, desc='trainIter')
        result = OrderedDict()
        for i, data in enumerate(bar, start=0):
            if i >= 20:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, img_idx = data['img'], data['isup_grade'], data['idx']
            # print(labels)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            outputs_main = model(inputs.cuda().float())
            loss = criterion(outputs_main, labels.cuda().float())
            train_idx.append(img_idx)
            loss = torch.sum(loss, 1)
            train_sample_loss.append(loss)
            loss = torch.mean(loss)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            bar.set_description('loss: %.5f, smth: %.5f' % (loss.detach().cpu(), smooth_loss))
        train_sample_loss = np.concatenate(train_sample_loss, 0)
        result['train_loss'] = np.mean(train_loss)
        result['train_sample_loss'] = np.stack(np.asarray(train_sample_loss).reshape(-1,1), np.asarray(train_idx).reshape(-1,1),
                                               1)
        return result

    def val_epoch(self, valloader, criterion):   ## val
        model.eval()
        val_loss, val_label, val_preds, val_provider, val_idx = [], [], [], [], []
        result = OrderedDict()
        with torch.no_grad():
            for i, data in enumerate(tqdm(valloader, desc='valIter'), start=0):
                if i > 20:
                    break
                get the inputs; data is a list of [inputs, labels]
                inputs, labels, provider, img_idx = data['img'], data['isup_grade'], data['datacenter'], data['idx']
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs_main = model(inputs.cuda().float())
                # outputs_aux = outputs['aux'].squeeze(dim=1)  # for regression
                loss = criterion(outputs_main, labels.float().cuda())
                loss = torch.sum(loss, 1)
                # print("output_main", outputs_main.shape)
                # print("labels", labels.shape)
                val_loss.append(loss.detach().cup().numpy())
                val_idx.append(img_idx)
                val_label.append(labels.sum(1).cpu())
                val_preds.append(outputs_main.sigmoid().sum(1).round().cpu())
                val_provider += provider
                # print("postval_label", labels.sum(1))
                # print("postval_label", outputs_main.sigmoid().sum(1).round())
        if self.scheduler:
            self.scheduler.step()
        val_label = torch.cat(val_label, 0)
        val_preds = torch.cat(val_preds, 0)
        # print(val_label.shape, val_preds.shape)
        index_r = [i for i, x in enumerate(val_provider) if x == "radboud"]
        index_k = [i for i, x in enumerate(val_provider) if x == "karolinska"]
        kappa = cohen_kappa_score(val_label, val_preds, weights='quadratic')
        kappa_r = cohen_kappa_score(val_label[index_r], val_preds[index_r], weights='quadratic')
        kappa_k = cohen_kappa_score(val_label[index_k], val_preds[index_k], weights='quadratic')
        val_loss = np.concatenate(val_loss, 0)
        result['val_loss'] = np.mean(val_loss)
        result['val_sample_loss'] = np.stack(np.asarray(val_loss).reshape(-1,1), np.asarray(val_idx).reshape(-1,1),
                                             1)
        result['kappa'] = kappa
        result['kappa_r'] = kappa_r
        result['kappa_k'] = kappa_k
        return result

def save_checkpoint(state, is_best, fname):
    torch.save(state, '{}_ckpt.pth.tar'.format(fname))
    if is_best:
        # shutil.copyfile('{}_ckpt.pth.tar'.format(fname), '{}_best.pth.tar'.format(fname))
        state = state['state_dict']
        torch.save(state, '{}_best.pth.tar'.format(fname)) ## only save weights for best model


if __name__ == "__main__":
    """Define Your Input"""
    parser = argparse.ArgumentParser(description='Optional arguments')
    parser.add_argument('--fold', type=str, default="0,1,2,3", help='which fold to train.')
    parser.add_argument('--mode', type=str, default="curriculum_easy_10idx", help='train idx file')
    parser.add_argument('--provider', type=str, default="whole", help='which dataset to train.')
    parser.add_argument('--patch', default=36, type=int,
                        help='number of patches used for training')
    parser.add_argument('--bs', default=6, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='epochs for training')
    # parser.add_argument('--local_rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    args = parser.parse_args()
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    folds = args.fold
    mode = args.mode
    folds = folds.split(',')
    folds = [int(i) for i in folds]
    provider = args.provider
    nfolds = 4
    N = args.patch ## number of patches
    fname = f'Resnext50_{N}patch_constant_lr_{mode}'

    if provider == "rad":
        csv_file = '../input/csv_pkl_files/radboud_{}_fold_train_wo_sus.csv'.format(nfolds)
    elif provider == 'kar':
        csv_file = '../input/csv_pkl_files/karolinska_{}_fold_train_wo_sus.csv'.format(nfolds)
    else:
        csv_file = '../Train_Data.csv'
    image_dir = '../../tile_concat_wy/input/panda-36x256x256-tiles-data-opt/train/'
    bs = args.bs
    epochs = args.epochs
    Pre_Train = False
    start_epoch = 0

    ## image transformation
    tsfm = data_transform()
    # tsfm = None
    ## dataset, can fetch data by dataset[idx]
    dataset = PandaPatchDataset(csv_file, image_dir, 256, transform=tsfm, N = N, rand=True)
    ## dataloader
    df = pd.read_csv(csv_file)
    crossValData = crossValDataloader(dataset, bs)
    criterion = nn.BCEWithLogitsLoss(reduction = 'none')
    ## tensorboard writer
    writerDir = './runs'
    ## weight saving
    weightsDir = './weights/{}'.format(fname)
    # if args.local_rank == 0:
    timeStamp = datetime.now(timezone('US/Pacific')).strftime("%m_%d_%H_%M_%S")
    writer = SummaryWriter('{}/{}_{}'.format(writerDir, fname, timeStamp))
    check_folder_exists(weightsDir)
    check_folder_exists(writerDir)

    for fold in folds:
        print(f"training fold {fold}!")
        train_idx = list(np.load(f"../Idxs/{mode}_{fold}.npy"))
        val_idx = df.index[df['split'] == fold].tolist()
        trainloader, valloader = crossValData(train_idx, val_idx)
        model = Model.from_pretrained('efficientnet-b0', num_classes = 5).cuda()
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank])
        optimizer = optim.Adam(model.parameters(), lr=0.00003)  # current best 0.00003
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        best_kappa = 0
        best_kappa_k = 0
        best_kappa_k = 0
        best_kappa_r = 0
        if Pre_Train:
            model_path = './weights/Resnext50_36patch_adam_cos_spine_{}/Resnext50_36patch_adam_cos_spine_{}_{}_best.pth.tar'.format(provider,provider,fold)
            map_location = {"cuda:0": "cuda:{}".format(args.local_rank)}
            state = torch.load(model_path, map_location = map_location)
            pretrained_dict = state['state_dict']
            start_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer'])
            # model_dict = model.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # model_dict.update(pretrained_dict)
            model.load_state_dict(pretrained_dict)
            # best_kappa = state['kappa']
            print(f"Load pre-trained weights for model, start epoch {start_epoch}.")

        Training = Train(model, optimizer, None)
        weightsPath = os.path.join(weightsDir, '{}_{}'.format(fname, fold))
        for epoch in tqdm(range(start_epoch,epochs), desc='epoch'):
            # trainSampler.set_epoch(epoch) ## have to use this for shuffle the image
            train = Training.train_epoch(trainloader,criterion)
            np.save(os.path.join(writerDir, f"train_sample_loss_{fold}_{epoch}_{mode}.npy"))
            ## TODO: unlabeled data monitoring
            # if args.local_rank == 0:
            val = Training.val_epoch(valloader, criterion)
            writer.add_scalar('Fold:{}/train_loss'.format(fold), train['train_loss'], epoch)
            writer.add_scalar('Fold:{}/val_loss'.format(fold), val['val_loss'], epoch)
            writer.add_scalar('Fold:{}/kappa_score'.format(fold), val['kappa'], epoch)
            if provider == "rad":
                writer.add_scalar('Fold:{}/kappa_score_r'.format(fold), val['kappa_r'], epoch)
            elif provider == "kar":
                writer.add_scalar('Fold:{}/kappa_score_k'.format(fold), val['kappa_k'], epoch)
            else:
                writer.add_scalar('Fold:{}/kappa_score_r'.format(fold), val['kappa_r'], epoch)
                writer.add_scalar('Fold:{}/kappa_score_k'.format(fold), val['kappa_k'], epoch)
            writer.flush()
            tqdm.write("Epoch {}, train loss: {:.4f}, val loss: {:.4f}, kappa-score: {:.4f}.\n".format(epoch,
                                                                                           train['train_loss'],
                                                                                           val['val_loss'],
                                                                                           val['kappa']))
            ## save the checkpoints and best model
            # if args.local_rank == 0:
            is_best = val['kappa'] > best_kappa
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'kappa': val['kappa'],
                'optimizer': optimizer.state_dict(),
            }, is_best, weightsPath)
            best_kappa = val['kappa'] if is_best else best_kappa

            is_best = val['kappa_r'] > best_kappa_r
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'kappa': val['kappa'],
                'optimizer': optimizer.state_dict(),
            }, is_best, weightsPath + "_rad")
            best_kappa_r = val['kappa_r'] if is_best else best_kappa_r

            is_best = val['kappa_k'] > best_kappa_k
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'kappa': val['kappa'],
                'optimizer': optimizer.state_dict(),
            }, is_best, weightsPath + "_kar")
            best_kappa_k = val['kappa_k'] if is_best else best_kappa_k

        del model
        del optimizer
        del Training
    writer.close()