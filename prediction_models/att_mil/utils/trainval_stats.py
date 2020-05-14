import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn import metrics

class AverageMeterSet:
    def __init__(self):
        self.sums = {}
        self.counts = {}
        self.avgs = {}

    def _compute_avgs(self):
        for name in self.sums:
            self.avgs[name] = float(self.sums[name]) / float(self.counts[name])

    def update_dict(self, name_val_dict, n=1):
        for name, val in name_val_dict.items():
            self.update(name, val, n)

    def update(self, name, value, n=1):
        if name not in self.sums:
            self.sums[name] = value
            self.counts[name] = n
        else:
            self.sums[name] = self.sums[name] + value
            self.counts[name] = self.counts[name] + n

    def pretty_string(self, ignore=('zzz')):
        self._compute_avgs()
        s = []
        for name, avg in self.avgs.items():
            keep = True
            for ign in ignore:
                if ign in name:
                    keep = False
            if keep:
                s.append('{0:s}: {1:.3f}'.format(name, avg))
        s = ', '.join(s)
        return s

    def averages(self, idx, prefix=''):
        self._compute_avgs()
        return {prefix + name: (avg, idx) for name, avg in self.avgs.items()}


class StatTracker:
    '''
    Helper class for collecting per-episode rewards and other stats during
    training.
    '''

    def __init__(self, log_name=None, log_dir=None):
        assert((log_name is None) or (log_dir is None))
        if log_dir is None:
            self.writer = SummaryWriter(comment=log_name)
        else:
            print('log_dir: {}'.format(str(log_dir)))
            try:
                self.writer = SummaryWriter(logdir=log_dir)
            except:
                self.writer = SummaryWriter(log_dir=log_dir)

    def close(self):
        self.writer.close()

    def record_stats(self, stat_dict):
        '''
        Record some named stats in the underlying tensorboard log.
        '''
        for stat_name, stat_vals in stat_dict.items():
            self.writer.add_scalar(stat_name, stat_vals[0], stat_vals[1])

    def add_image(self, label, image, number):
        '''
        Add an image to the tensorboard log.
        '''
        self.writer.add_image(label, image, number)


def compute_kappa(all_preds, all_labels):
    score = metrics.cohen_kappa_score(all_preds, all_labels, weights='quadratic')
    return score