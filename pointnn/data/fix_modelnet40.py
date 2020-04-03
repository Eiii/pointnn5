from pathlib import Path
from argparse import ArgumentParser
from kaolin.rep.TriangleMesh import TriangleMesh
import re

def needs_header_fix(c):
    match = re.match(r'^OFF\S', c)
    return match is not None

def header_fix(c):
    return re.sub(r'^OFF(\S)', r'OFF\n\1', c)

def read_model(path):
    with path.open('r') as fd:
        return fd.read()

def write_model(path, contents):
    with path.open('w') as fd:
        fd.write(contents)

bad_models = ['curtain/train/curtain_0066.off', 'cone/train/cone_0117.off']

def convert_all(base):
    all_models = base.glob('**/*.off')
    for path in all_models:
        model = read_model(path)
        if needs_header_fix(model):
            print(path)
            changed = True
            model = header_fix(model)
            write_model(path, model)
        if str(path.relative_to(base)) in bad_models:
            print(path)
            path.rename(path.with_suffix('.off.bad'))

def load_all():
    base = Path('data/ModelNet40')
    all_models = list(base.glob('**/*.off'))
    bad = []
    for model in all_models:
        try:
            data = TriangleMesh.from_off(model)
            samp = data.sample(16)
        except:
            print(model)
            bad.append(model)
            raise
    print(bad)

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('base', type=Path)
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    convert_all(args.base)
