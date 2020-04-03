""" Class that represents the problem to solve (i.e. target & loss function)
"""

from .data.starcraft import StarcraftDataset, parse_frame
from .ext.loss import cham_dist_2
import kaolin
import kaolin.transforms as tfs
from kaolin.datasets.modelnet import ModelNetPointCloud
from kaolin.metrics.point import batch_chamfer_distance

from torch.utils.data import random_split
import torch
import torch.nn.functional as F


def split_dataset(dataset, valid_amt=0.1):
    valid_len = int(len(dataset)*valid_amt)
    train_len = len(dataset) - valid_len
    train, valid = random_split(dataset, [train_len, valid_len])
    return train, valid


class Autoencoder:
    def __init__(self, data_path, rotate, classes=None, jitter=None):
        t = tfs.Compose([tfs.UnitSpherePointCloud(),
                         tfs.RandomRotatePointCloud(type=rotate)])
        self.train_dataset = ModelNetPointCloud(basedir=data_path,
                                                split='train',
                                                categories=classes,
                                                transform=t,
                                                num_points=2**10,
                                                sample_points=2**12)
        self.valid_dataset = ModelNetPointCloud(basedir=data_path,
                                                split='test',
                                                categories=classes,
                                                transform=t,
                                                num_points=2**10,
                                                sample_points=2**12)
        self.dataset = self.train_dataset

    def get_target(self, item):
        points, class_ = item
        return points

    def loss(self, item, pred):
        tgt = self.get_target(item)
        return cham_dist_2(pred, tgt)


class StarcraftScene:
    def __init__(self, data_path, max_units=36, hide_type=False,
                 max_files=None, num_workers=0, pred_dist=1):
        # Load training & test dataset
        self.train_dataset = StarcraftDataset(data_path,
                                              max_units=max_units,
                                              hide_type=hide_type,
                                              max_files=max_files,
                                              num_workers=num_workers,
                                              pred_dist=pred_dist)
        self.valid_dataset = StarcraftDataset(data_path+'/test',
                                              max_units=max_units,
                                              hide_type=hide_type,
                                              max_files=max_files,
                                              num_workers=num_workers,
                                              pred_dist=pred_dist)

    def get_target(self, item):
        return item['post']

    def loss(self, item, pred):
        l = sc2_loss(item, pred)
        avg_loss = l.sum() / item['mask'].sum()
        return avg_loss

def sc2_loss(item, pred):
    pre = parse_frame(item['pre'])
    target = parse_frame(item['post'])
    # Health loss
    h_loss = F.mse_loss(pred['health'], target['health'], reduction='none')
    # Shield loss
    s_loss = F.mse_loss(pred['shields'], target['shields'], reduction='none')
    # Position loss
    delta = target['pos'] - pre['pos']
    sqr_diffs = (delta - pred['pos'])**2
    p_loss = sqr_diffs.sum(dim=1)
    # Orientation loss
    _, idx_target = target['ori'].max(dim=1)
    idx_target = idx_target - (1-item['mask'])
    o_loss = F.cross_entropy(pred['ori'], idx_target, reduction='none',
                             ignore_index=-1)
    total_loss = torch.stack((p_loss, h_loss, s_loss, o_loss), dim=0)
    total_loss *= item['mask'].unsqueeze(0)
    return total_loss


def make_problem(problem_name, **kwargs):
    _problem_map = {'autoencoder': Autoencoder, 'starcraft': StarcraftScene}
    return _problem_map[problem_name](**kwargs)
