import argparse

from . import common

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(measures, out_path, show_training=False, logy=False):
    size = (8, 5)
    fig, ax = plt.subplots(figsize=size, dpi=200)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if logy:
        ax.set_yscale('log')
    #ax.set_ylim([0, 0.05])
    #ax.set_xlim([40, 75])
    all_names = list({m.name for m in measures if m.name is not None})
    print(f'Plotting {all_names}...')
    for name in all_names:
        print(f'Plotting {name}')
        # Valid loss
        vls = [m._valid_stats for m in measures if m.name == name]
        xs = None
        ys = None
        ts = None
        for vl in vls:
            x = [v[0] for v in vl]
            y = [v[2] for v in vl]
            t = vl[-1][1]
            print(t/60)
            y_array = np.array([y])
            if np.isnan(y_array).any():
                print('NaN loss - skipping')
                continue
            xs = np.vstack((xs, x)) if xs is not None else np.array([x])
            ys = np.vstack((ys, y)) if ys is not None else y_array
            ts = np.vstack((ts, t)) if ts is not None else np.array([t])
        assert((xs == xs[0]).all())
        xs = xs[0]
        errs = np.std(ys, axis=0)/np.sqrt(ys.shape[0])
        ys = np.mean(ys, axis=0)
        print(name, np.min(ts)/60)
        main_plot = ax.plot(xs, ys, label=name)
        main_color = main_plot[-1].get_color()

        # Errors
        if len(xs) > 15:
            i = int(len(xs)/15)
            xs = xs[::i]
            ys = ys[::i]
            errs = errs[::i]
            ax.errorbar(xs, ys, yerr=errs, fmt='none', color=main_color)

        # Training loss
        if show_training:
            tls = [m._training_loss for m in measures if m.name == name]
            xs = ys = None
            for tl in tls:
                x = [t[0] for t in tl]
                y = [t[2] for t in tl]
                y_array = np.array([y])
                if np.isnan(y_array).any():
                    continue
                xs = np.vstack((xs, x)) if xs is not None else np.array([x])
                ys = np.vstack((ys, y)) if ys is not None else y_array
            assert((xs == xs[0]).all())
            xs = xs[0]
            ys = np.mean(ys, axis=0)
            ax.plot(xs, ys, linestyle='--', color=main_color, alpha=0.5)
    ax.legend()


def make_plots(folder, out_path, logy=False, filter=None, training=False):
    measures = [m['measure'] for m in common.load_all(folder)]
    if filter:
        measures = [m for m in measures if filter in m.name]
    plot_training_curves(measures, out_path, training, logy)
    plt.tight_layout()
    plt.savefig(out_path)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('output', nargs='?', default='out.png')
    parser.add_argument('--filter', default=None)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--logy', action='store_true')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    make_plots(args.folder, args.output, args.logy, args.filter, args.training)

