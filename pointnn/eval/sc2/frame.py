from ...data.starcraft import StarcraftDataset, parse_frame
from .. import common
import argparse
import torch

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as lines

colors = ('red', 'blue')
lightcolors = ('pink', 'lightblue')
markers = 'os^'

def extract_input(scene, mask):
    mask_idxs = mask.nonzero()[:, 0]
    pos = scene[13:15, mask_idxs]
    _, type_ = scene[1:4, mask_idxs].max(dim=0)
    team = scene[0, mask_idxs]
    return pos, team, type_

def draw_scene(ax, scene, mask, colors=lightcolors):
    pos, team, type_ = extract_input(scene, mask)
    for t, color in zip((0, 1), colors):
        for y, ms in zip((0, 1, 2), markers):
            sel = (type_==y)&(team==t)
            p = pos[:, sel]
            r = ax.scatter(*p, marker=ms, alpha=0.5, c=color)

def draw_deltas(ax, prev_scene, next_scene, mask):
    pre_pos, team, _ = extract_input(prev_scene, mask)
    next_pos, team, _ = extract_input(next_scene, mask)
    for t, color in zip((0, 1), colors):
        sel = (team==t)
        pp = pre_pos[:, sel].detach()
        np = next_pos[:, sel].detach()
        draw_deltas_(ax, pp, np, color)

def draw_deltas_(ax, prev, next_, color):
    for i in range(prev.size(1)):
        p = prev[:, i].tolist()
        n = next_[:, i].tolist()
        xs, ys = zip(p, n)
        l = lines.Line2D(xs, ys, color=color, linewidth=1)
        ax.add_line(l)

def draw_scene_pred(ax, scene, pred, mask, colors=colors):
    _, team, type_ = extract_input(scene, mask)
    pre = parse_frame(scene.unsqueeze(0))
    mask_idxs = mask.nonzero()[:, 0]
    prev_pos = pre['pos'].squeeze(0)[:, mask_idxs]
    offset = pred['pos'].squeeze(0)[:, mask_idxs]
    pos = (prev_pos + offset).detach()
    off = offset.detach()
    for t, color in zip((0, 1), colors):
        for y, ms in zip((0, 1, 2), markers):
            sel = (type_==y)&(team==t)
            p = pos[:, sel]
            r = ax.scatter(*p, marker=ms, alpha=0.5, c=color)
        sel = (team==t)
        prev = prev_pos[:, sel]
        next_ = pos[:, sel]
        draw_deltas_(ax, prev, next_, color)

def plot_value_pred(ax, actual, pred, mask, value):
    mask_idxs = mask.nonzero()[:, 0]
    actual = parse_frame(actual.unsqueeze(0))[value][0, mask_idxs].tolist()
    pred = pred[value][0, mask_idxs].tolist()
    xs = list(range(mask.sum().item()))
    ax.scatter(actual, pred, marker='+')
    ax.plot([0,1], [0,1], alpha=0.25, color='black')
    ax.set_xlabel('Actual value')
    ax.set_ylabel('Predicted value')

def pred(scene, net):
    net = net.eval()
    args = [a.unsqueeze(0) for a in net.get_args(scene)]
    p = net(*args)
    return p


def plot_pred(item, net, num, title=None):
    rows = 2
    cols = 2
    fig = plt.figure(figsize=(4*cols, 4*rows), dpi=200)
    ax1 = fig.add_subplot(rows, cols, 1)
    d = 2
    ax1.set_xlim(-d, d)
    ax1.set_ylim(-d, d)
    ax1.set_title('Actual transition')
    draw_scene(ax1, item['pre'], item['mask'])
    draw_scene(ax1, item['post'], item['mask'], colors=colors)
    draw_deltas(ax1, item['pre'], item['post'], item['mask'])

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.set_xlim(*ax1.get_xlim())
    ax2.set_ylim(*ax1.get_ylim())
    ax2.set_title('Predicted transition')
    draw_scene(ax2, item['pre'], item['mask'])
    pr = pred(item, net)
    draw_scene_pred(ax2, item['pre'], pr, item['mask'])

    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.set_title('Predicted unit health')
    plot_value_pred(ax3, item['post'], pr, item['mask'], 'health')

    ax3 = fig.add_subplot(rows, cols, 4)
    ax3.set_title('Predicted unit shields')
    plot_value_pred(ax3, item['post'], pr, item['mask'], 'shields')
    if title:
        fig.suptitle(title)
    fig.savefig(f'{num:03d}.png')


def _onehot(arr, max):
    return torch.eye(max)[arr].T


def update(prev, pred):
    pre = parse_frame(prev['pre'].unsqueeze(0))
    new_health = pred['health'].squeeze(0)
    new_s = pred['shields'].squeeze(0)
    _, new_ori = pred['ori'].max(dim=1)
    new_ori = _onehot(new_ori.squeeze(0), 7)
    new_pos = (pre['pos'] + pred['pos']).squeeze(0)
    n = prev['pre'].clone()
    n[4, :] = new_health
    n[5, :] = new_s
    n[6:13, :] = new_ori
    n[13:15, :] = new_pos
    n, n_mask = remove_dead(n, prev['mask'])
    prev['pre'] = n.detach()
    prev['mask'] = n_mask
    return prev


def remove_dead(f, mask):
    h = f[4, :]
    dead_idxs = h<=0.05
    mask[dead_idxs] = 0
    return f, mask

def remove_team(pretend):
    team_idxs = (pretend['pre'][0, :]==0)
    mask = pretend['mask']
    mask[team_idxs] = 0
    pretend['mask'] = mask
    return pretend

def plot_pred_pretend(item, pretend, net, num):
    rows = 2
    cols = 2
    fig = plt.figure(figsize=(4*cols, 4*rows), dpi=200)
    ax1 = fig.add_subplot(rows, cols, 1)
    d = 2
    ax1.set_xlim(-d, d)
    ax1.set_ylim(-d, d)
    ax1.set_title('Actual transition')
    draw_scene(ax1, item['pre'], item['mask'])
    draw_scene(ax1, item['post'], item['mask'], colors=colors)
    draw_deltas(ax1, item['pre'], item['post'], item['mask'])

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.set_xlim(*ax1.get_xlim())
    ax2.set_ylim(*ax1.get_ylim())
    ax2.set_title('Predicted transition')
    draw_scene(ax2, pretend['pre'], pretend['mask'])
    pr = pred(pretend, net)
    draw_scene_pred(ax2, pretend['pre'], pr, pretend['mask'])

    ax3 = fig.add_subplot(rows, cols, 3)
    plot_value_pred(ax3, pretend['pre'], pr, pretend['mask'], 'health')
    ax3.set_title('Predicted unit health')
    ax3.set_xlabel('Previous value')

    ax4 = fig.add_subplot(rows, cols, 4)
    plot_value_pred(ax4, pretend['pre'], pr, pretend['mask'], 'shields')
    ax4.set_title('Predicted unit shields')
    ax4.set_xlabel('Previous value')
    fig.savefig(f'{num:03d}.png')
    return update(pretend, pr)

def plot_scene_simple(item, num):
    rows = 1
    cols = 1
    fig = plt.figure(figsize=(4*cols, 4*rows), dpi=200)
    ax1 = fig.add_subplot(rows, cols, 1)
    draw_scene(ax1, item['pre'], item['mask'], colors=colors)
    draw_deltas(ax1, item['pre'], item['post'], item['mask'])
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    fig.savefig(f'{num:03d}.png')
    plt.close(fig)

def main_single(net_path, data_path):
    if net_path.is_dir():
        net_path = list(net_path.glob('*.net'))[0]
        print(net_path)
    measure = common.load_measure(net_path.with_suffix('.pkl'))
    net = common.load_net(net_path.with_suffix('.net'), measure)
    ds = StarcraftDataset(data_path, max_units=70, max_files=1, hide_type=False)
    ep_idx = 0
    for i, item in enumerate(ds.episode_frames(ep_idx)):
        plot_pred(item, net, i)

def main_multistep(net_path, data_path):
    if net_path.is_dir():
        net_path = list(net_path.glob('*.net'))[0]
        print(net_path)
    measure = common.load_measure(net_path.with_suffix('.pkl'))
    net = common.load_net(net_path.with_suffix('.net'), measure)
    ds = StarcraftDataset(data_path, max_units=70, max_files=1, hide_type=False,
                          pred_dist=5)
    ep_idx = 8
    for i, (d, item) in enumerate(ds.episode_frames(ep_idx, include_delta=True)):
        s = f'Time delta = {d}'
        plot_pred(item, net, i, s)

def main_simple():
    data_path = 'data/zerglots'
    ds = StarcraftDataset(data_path, device='cpu', max_units=30)
    for i in range(0, len(ds), 10):
        item = ds[i]
        plot_scene_simple(item, i)

def main_long(net_path, data_path):
    if net_path.is_dir():
        net_path = list(net_path.glob('*.net'))[0]
        print(net_path)
    measure = common.load_measure(net_path.with_suffix('.pkl'))
    net = common.load_net(net_path.with_suffix('.net'), measure)
    ds = StarcraftDataset(data_path, max_units=70, max_files=1, hide_type=False)
    ep_idx = 75
    fs = ds.episode_frames(ep_idx)
    pretend = fs[0]
    for i, item in enumerate(fs):
        pretend = plot_pred_pretend(item, pretend, net, i)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=Path)
    parser.add_argument('--data', type=Path)
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    main_single(args.net, args.data)
