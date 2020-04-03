import json
from pathlib import Path

from . import nets


def read_experiment_json(path):
    """
    path: Path to JSON experiment description

    Returns:
        List of network descriptions involved in the experiment
    """
    with open(path, 'rb') as fd:
        desc = json.load(fd)
    run_args = []
    out_path = Path(desc['experiment_path'])
    prob_args = (desc['problem_type'], desc['problem_args'])
    if not out_path.exists():
        out_path.mkdir(parents=True)
    for entry in desc['entries']:
        net_args = entry.get('net_args', dict())
        train_args = entry.get('train_args', dict())
        print(f"Experiment {entry['name']}")
        print(f'Train args: {train_args}')
        print(f'Net args: {net_args}')
        net = nets.make_net(entry['net'], **net_args)
        args = (net, entry['name'], prob_args, train_args, desc['epochs'],
                desc['experiment_path'])
        run_args += [args]*entry['repeat']
    return run_args


