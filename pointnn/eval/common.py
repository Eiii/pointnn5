from ..data.loader import ModelnetDataset
import pickle
from pathlib import Path
import torch
from .. import nets


def load_measure(path):
    with open(path, 'rb') as fd:
        data = pickle.load(fd)
    if data['measure'].name is None:
        data['measure'].name = path.name.split(':')[0]
    return data


def load_net(path, measure):
    net = nets.make_net_args(measure['net_type'], measure['net_args'])
    pms = torch.load(path)
    net.load_state_dict(pms)
    return net


def load_pairs(pairs):
    def _l(p):
        m_path, n_path = p
        m = load_measure(m_path)
        return m, load_net(n_path, m)
    return [_l(p) for p in pairs]


def all_results_path(folder):
    return Path(folder).glob('*.pkl')


def add_nets_path(results):
    def to_pair(r):
        net = r.with_suffix('.net')
        return (r, net)
    return [to_pair(r) for r in results]


def load_all(folder):
    rs = all_results_path(folder)
    measures = [load_measure(p) for p in rs]
    return measures


def all_net_names(results):
    return {m['measure'].name for m, _ in results}


def nets_by_name(results, name):
    return [(m, n) for m, n in results if m['measure'].name == name]


def filter_best(results):
    all_nets = list({m['measure'].name for m, n in results})
    bests = {}
    for entry in results:
        m, nets = entry
        name = m['measure'].name
        last_valid = get_last_valid(m)
        if name not in bests or last_valid < get_last_valid(bests[name][0]):
            bests[name] = entry
    return [v for _, v in bests.items()]


def filter_type(results, name):
    if not name:
        return results
    return [(m, n) for m, n in results if m['net_type'] == name]


def filter_name(results, name):
    if not name:
        return results
    return [(m, n) for m, n in results if name in m['measure'].name]


def filter_out_name(results, name):
    if not name:
        return results
    return [(m, n) for m, n in results if name not in m['measure'].name]


def get_last_valid(m):
    return m['measure']._valid_stats[-1][2]


def get_dataset_classes(path10, path40=None):
    ds = ModelnetDataset(path10, type='test')
    classes10 = list(ds.classes)
    if path40 is not None:
        ds = ModelnetDataset(path40, type='test')
        classes40 = [x for x in ds.classes if x not in classes10]
        return classes10, classes40
    else:
        return classes10
