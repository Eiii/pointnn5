import torch
import math
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class StarcraftDataset(Dataset):
    def __init__(self, base, max_units=None, pred_dist=1,
                 hide_type=False, max_files=None, include_anom=False,
                 num_workers=0):
        print(f'Loading SC dataset @ {base}..')
        self.base_dir = Path(base)
        self.max_units = max_units
        if isinstance(pred_dist, int):
            self.pred_dists = list(range(1, pred_dist+1))
        elif isinstance(pred_dist, list):
            self.pred_dists = pred_dist
        self.hide_type = hide_type
        self.include_anom = include_anom
        raw_episodes = self.load_episodes(max_files)
        print('Raw episodes loaded.')
        if num_workers > 0:
            import multiprocessing
            p = multiprocessing.Pool(processes=num_workers) #TODO: with syntax
            self.episodes = p.map(self.convert_episode, raw_episodes)
            p.terminate()
        else:
            self.episodes = [self.convert_episode(e) for e in raw_episodes]
        print('Episodes processed.')
        self.all_entries = self.calc_dataset()
        self.anoms = []
        print(f'Done ({len(self.episodes)} episodes, {len(self.all_entries)} entries).')

    def load_episodes(self, max_files=None):
        files = self.base_dir.glob('*.pkl')
        all_episodes = []
        from itertools import islice
        if max_files is not None:
            print(f'WARNING: ONLY LOADING PARTIAL DATASET')
        for f in (islice(files, max_files) if max_files else files):
            with f.open('rb') as fd:
                d = pickle.load(fd)
                all_episodes += d
        return all_episodes

    def calc_dataset(self):
        all_entries = []
        for ep_idx, ep in enumerate(self.episodes):
            for prev_idx in range(len(ep)-max(self.pred_dists)):
                for offset in self.pred_dists:
                    pred_idx = prev_idx+offset
                    all_entries.append((ep_idx, prev_idx, pred_idx))
        return all_entries

    def convert_episode(self, ep):
        return [self.convert_frame(f) for f in ep]

    def add_time(self, frame, t):
        ys = frame[-1, :]
        zs = t * np.ones_like(ys)
        mod_frame = np.vstack((frame, zs[np.newaxis, :]))
        return mod_frame

    def convert_frame(self, f):
        # ID - No change
        ids = f[0, :]
        # Owner - [1, 16] to onehot(1)
        @np.vectorize
        def _owner(x):
            return {1: 0, 16: 1}[x]
        owner = _owner(f[1, :])
        # Type - [48, 73, 105] to onehot(3)
        # NOTE - 1970 = fast zergling

        @np.vectorize
        def _type(x):
            return {48: 0, 73: 1, 105: 2, 1970: 2}[x]

        def _onehot(arr, max):
            return np.eye(max)[arr].T
        type = _onehot(_type(f[2, :]), 3)
        anoms = np.where(f[2, :] == 1970)[0]
        if self.hide_type:
            type = np.zeros_like(type)
        # Health - [0 - 100] to [0 - 1]
        @np.vectorize
        def _health(x):
            return x/100
        health = _health(f[3, :])
        # Sheilds - [0 - 50] to [0 - 0.5]
        sheilds = _health(f[4, :])
        # Orientation - [0 - 6] to onehot(7)
        ori = _onehot(f[5, :], 7)
        # X - [20 - 47] to [-1 - 1]
        # Y - [20 - 47] to [-1 - 1]
        @np.vectorize
        def _coord(x):
            return (x-33.5)/13.5
        x = _coord(f[6, :])
        y = _coord(f[7, :])
        t = np.zeros_like(y)
        c = np.vstack((owner[np.newaxis, :],
                       type,
                       health[np.newaxis, :],
                       sheilds[np.newaxis, :],
                       ori,
                       x[np.newaxis, :],
                       y[np.newaxis, :],
                       t[np.newaxis, :]))
        # LAYOUT:
        #  0: Owner
        #  1-3: Type onehot (?, ?, ?)
        #  4: Health
        #  5: Shields
        #  6-12: Orientation onehot
        #  13-14: X, Y
        #  15: Z
        return ids, c, anoms

    def build_item(self, prev, pred, offset):
        # Find dead units
        # Match units by tag
        #TODO
        pre_idx, pre_dat, pre_anoms = prev
        post_idx, post_dat, _ = pred
        # Ensure each entity in pre has a corresponding entry in post
        # Basically, add back in units that have died
        dead = np.setdiff1d(pre_idx, post_idx)
        if dead.size > 0:
            post_idx = np.append(post_idx, dead)
            for d in dead:
                idx, = (pre_idx == d).nonzero()
                assert(idx.size == 1)
                # Dead unit's properties are it's previous properties w/
                # health zeroed
                row = pre_dat[:, idx].copy()
                row[3, 0] = 0 #Set health to 0
                post_dat = np.hstack((post_dat, row))
        # To ensure pre and post entity lists match up, sort them according to
        # the unit indices
        sort_pre = np.argsort(pre_idx)
        sort_post = np.argsort(post_idx)
        num_units = pre_dat.shape[1]
        assert(num_units == post_dat.shape[1])
        pre_dat = self.pad(pre_dat[:, sort_pre], self.max_units)
        post_dat = self.pad(post_dat[:, sort_post], self.max_units)
        pre = torch.tensor(pre_dat, dtype=torch.float)
        post = torch.tensor(post_dat, dtype=torch.float)
        mask = self.make_mask(num_units, pre)
        np_tag = pre_idx[sort_pre]
        np_tag = np.pad(pre_idx, (0, self.max_units-len(np_tag)), 'constant')
        tag = torch.tensor(np_tag, dtype=torch.int)
        off = torch.tensor(offset, dtype=torch.int)
        d = {'pre': pre,
             'post': post,
             'tag': tag,
             'mask': mask,
             'offset': off}
        if self.include_anom:
            d['anom_idxs'] = torch.tensor(pre_anoms)
        return d

    def pad(self, d, max_):
        if d.shape[1] < max_:
            diff = max_ - d.shape[1]
            pad = np.zeros((d.shape[0], diff))
            d = np.hstack((d, pad))
        return d

    def make_mask(self, num, ref):
        max_len = ref.shape[1]
        x = torch.arange(max_len) < num
        return x.int()

    def num_episodes(self):
        return len(set(ep_idx for ep_idx, _ in self.all_entries))

    def episode_frames(self, ep_idx, include_delta=False):
        ep = self.episodes[ep_idx]
        def make_entry(prev_idx, pred_idx):
            delta = pred_idx - prev_idx
            fr = self.build_item(ep[prev_idx], ep[pred_idx], delta)
            if include_delta:
                return (delta, fr)
            else:
                return fr
        return [make_entry(prev_idx, pred_idx) for e, prev_idx, pred_idx in self.all_entries if e == ep_idx]

    def __len__(self):
        return len(self.all_entries)

    def __getitem__(self, idx):
        ep_idx, prev_idx, pred_idx = self.all_entries[idx]
        offset = pred_idx - prev_idx
        prev = self.episodes[ep_idx][prev_idx]
        pred = self.episodes[ep_idx][pred_idx]
        return self.build_item(prev, pred, offset)


def parse_frame(frame):
    # Owner
    owner = frame[:, 0, :]
    # Type
    t_idxs = range(1, 1+3)
    type_ = frame[:, t_idxs, :]
    # Health
    health = frame[:, 4, :]
    # Shields
    shields = frame[:, 5, :]
    # Orientation
    o_idxs = range(6, 6+7)
    ori = frame[:, o_idxs, :]
    # X, Y, Z
    tp_idxs = range(13, 13+3)
    timepos = frame[:, tp_idxs, :]
    # X, Y
    p_idxs = range(13, 13+2)
    pos = frame[:, p_idxs, :]
    # Z
    t_idxs = [15]
    t = frame[:, t_idxs, :]
    return {'owner': owner,
            'type': type_,
            'health': health,
            'shields': shields,
            'ori': ori,
            'timepos': timepos,
            'time': t,
            'pos': pos}


if __name__ == '__main__':
    ds = StarcraftDataset('/home/eiii/StarCraftII/pysc2_output', max_units=36)
    print(len(ds))
