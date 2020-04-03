""" Training harness & helpers """

import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from torch.utils.data import DataLoader

from . import problem
from .measure import Measure
from pathlib import Path
from uuid import uuid4

import pickle

# TODO: surely this means I'm doing something wrong
def data_to_cuda(d):
    if isinstance(d, dict):
        return {k:v.cuda() for k,v in d.items()}
    elif isinstance(d, list):
        return [v.cuda() for v in d]


class Trainer:
    """ Main object that defines and tracks the state of a training run. """
    def __init__(self, net, problem, out_path, report_freq=5,
                 optim='adam', batch_size=4, lr=1e-3, weight_decay=1e-4,
                 momentum=0.95, period=1, num_workers=0):
        # 'Macro' parameters
        self.net = net
        self.problem = problem
        self.out_path = out_path
        # Training parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        # UI parameters
        self.report_freq = report_freq
        # Set up tools
        self.measure = Measure()
        self.init_tools(lr, weight_decay, momentum, period, optim)

    def init_tools(self, lr, wd, mom, period, optim_):
        if optim_ == 'adam':
            self.optim = optim.AdamW(self.net.parameters(), lr=lr,
                                     weight_decay=wd)
        elif optim_ == 'sgd':
            self.optim = optim.SGD(self.net.parameters(), lr=lr,
                                   momentum=mom, nesterov=True,
                                   weight_decay=wd)
        # LR Schedule
        self.sched = schedule.CosineAnnealingWarmRestarts(self.optim, period, 2)
        self.batch_sched_step = lambda x: self.sched.step(x)
        self.epoch_sched_step = lambda: None

    def train(self, epochs):
        self.start_time = None
        self.net = self.net.cuda()
        loader = DataLoader(self.problem.train_dataset, shuffle=True,
                            batch_size=self.batch_size, drop_last=True,
                            num_workers=self.num_workers, pin_memory=True)
        total_batches = len(loader)
        report_every = total_batches // self.report_freq
        valid_loader = DataLoader(self.problem.valid_dataset, shuffle=True,
                                  batch_size=self.batch_size, drop_last=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)
        for self.epoch in range(epochs):
            if self.problem.valid_dataset is not None:
                valid_loss = self.validation(valid_loader)
                wall_time = self.runtime()
                self.measure.valid_stats(self.epoch, wall_time, valid_loss)
            total_loss = 0
            for i, data in enumerate(loader):
                data = data_to_cuda(data)
                self.batch_sched_step(self.epoch + i/total_batches)
                self.zero_grads()
                net_args = self.net.get_args(data)
                pred = self.net.forward(*net_args)
                loss = self.problem.loss(data, pred)
                loss.backward()
                total_loss += loss.item()
                self.optim_step()
                if (i+1) % report_every == 0:
                    avg_loss = total_loss / report_every
                    epoch_time = self.epoch + (i+1) / total_batches
                    wall_time = self.runtime()
                    self.measure.training_loss(epoch_time, wall_time,
                                               avg_loss)
                    total_loss = 0
            self.epoch_sched_step()
        # Final validation
        if self.problem.valid_dataset is not None:
            wall_time = self.runtime()
            valid_loss = self.validation(valid_loader)
            self.measure.valid_stats(self.epoch+1, wall_time, valid_loss)

    def zero_grads(self):
        self.optim.zero_grad()

    def optim_step(self):
        self.optim.step()

    def dump_metadata(self):
        """ TODO: Merge this info into Measure instead of putting it alongside
        """
        o = self.out_path.with_suffix('.pkl')
        data = {'measure': self.measure,
                'net_type': type(self.net).__name__,
                'net_args': self.net.args}
        with o.open('wb') as fd:
            pickle.dump(data, fd)

    def dump_net(self):
        """ Save the network to disk.
        """
        o = self.out_path.with_suffix('.net')
        torch.save(self.net.cpu().state_dict(), o)

    def runtime(self):
        """ Returns time elapsed since this function was first called. """
        t = time.time()
        if self.start_time is None:
            self.start_time = t
        return t - self.start_time

    def validation(self, loader):
        """ Returns the validation score of the network """
        self.net.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                data = data_to_cuda(data)
                net_args = self.net.get_args(data)
                pred = self.net(*net_args)
                loss = self.problem.loss(data, pred)
                total_loss += loss.item()
        loss = total_loss/len(loader)
        self.net.train()
        return loss


def train_single(net, name, problem_args, train_args, epochs, out_dir,
                 gpu_id=None):
    """ Fully train and instrument a single network instantiation.
    net: Network to be trained
    name: Readable name of experiment
    problem_args: Description of the problem to solve
    train_args: Description of the training environment/setup
    epochs: Number of epochs to train for
    out_dir: Output directory for stats and final network
    gpu_id: ID of the CUDA device to use for training
    """
    if gpu_id is not None:
        print(f'GPU ID: {gpu_id}')
        torch.cuda.set_device(gpu_id)
    p_name, p_args = problem_args
    prob = problem.make_problem(p_name, **p_args)
    uid = uuid4().hex
    out_path = Path(out_dir) / f'{name}:{uid}'
    print(out_path)
    trainer = Trainer(net, prob, out_path, **train_args)
    trainer.train(epochs)
    trainer.dump_metadata()
    trainer.dump_net()
