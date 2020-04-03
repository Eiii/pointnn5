from . import train
from . import common

from time import sleep
import random
import argparse
import torch
import multiprocessing


def run_parallel(all_args):
    """ Use all the available CUDA devices to train the desired networks

    TODO: Update logging output

    all_args: List of network descriptions to be trained
    """
    num_gpus = torch.cuda.device_count()
    multiprocessing.set_start_method('spawn')
    available_gpus = list(range(num_gpus))
    running_procs = []
    random.shuffle(all_args)
    while len(all_args) > 0 or len(running_procs) > 0:
        while len(available_gpus) > 0 and len(all_args) > 0:
            # Get GPU
            gpu_id = available_gpus.pop(0)
            # Get next set of args
            args = all_args.pop(0) + (gpu_id,)
            p = multiprocessing.Process(target=train.train_single, args=args)
            p.reserved_gpu = gpu_id  # Why not?
            p.start()
            running_procs.append(p)
            print(f'Started w/ GPU {gpu_id}:')
            sleep(15)  # Long sleep to avoid dumb file access limits on startup
        finished_procs = []
        for p in running_procs:
            p.join(1)
            if p.exitcode is not None:
                gpu = p.reserved_gpu
                print(f'Finished {gpu}!')
                finished_procs.append(p)
                available_gpus.append(gpu)
        for p in finished_procs:
            running_procs.remove(p)


def run_single(all_args):
    for args in all_args:
        train.train_single(*args)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    exp_args = common.read_experiment_json(args.config)
    run_single(exp_args)
