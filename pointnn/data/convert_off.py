""" Tools to convert OFF models into point clouds """
import argparse
import numpy as np
import multiprocessing
from pathlib import Path
from itertools import islice


class OFFMesh:
    def __init__(self, in_):
        self.in_fd = self._skip_empty(in_)
        self._load()

    def _load(self):
        num_verts, num_faces = self._read_header()
        self.verts = np.zeros((num_verts, 3))
        self.faces = np.zeros((num_faces, 3), dtype=np.int)
        for idx, line in enumerate(islice(self.in_fd, num_verts)):
            self.verts[idx] = [float(v) for v in line.split()]
        for idx, line in enumerate(islice(self.in_fd, num_faces)):
            self.faces[idx] = [int(v) for v in line.split()[1:]]

    def _read_header(self):
        line = next(self.in_fd)
        assert('OFF' in line)
        line = line.replace('OFF', '')
        if line == '':
            line = next(self.in_fd)
        verts, faces, _ = [int(v) for v in line.split()]
        return verts, faces

    @property
    def num_faces(self):
        return self.faces.shape[0]

    def face_verts(self, i):
        return [self.verts[idx] for idx in self.faces[i]]

    @staticmethod
    def _skip_empty(fd):
        for line in fd:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            yield line


def triangle_area(a, b, c):
    return np.linalg.norm(np.cross(b-a, c-a)) / 2


def sample_points(mesh, num_points):
    face_weights = np.zeros(mesh.num_faces)
    for i in range(mesh.num_faces):
        verts = mesh.face_verts(i)
        face_weights[i] = triangle_area(*verts)
    face_weights /= sum(face_weights)
    points = np.zeros((num_points, 3))
    for i in range(num_points):
        face_idx = np.random.choice(mesh.num_faces, p=face_weights)
        a, b, c = mesh.face_verts(face_idx)
        r1, r2 = np.random.rand(2)
        p = (1-np.sqrt(r1))*a + np.sqrt(r1)*(1-r2)*b + np.sqrt(r1)*r2*c
        points[i] = p
    return points


def correct_points(points):
    # Center point cloud
    points -= np.mean(points, axis=0)
    # Scale to unit sphere
    max_point = np.max(np.linalg.norm(points, axis=1))
    points /= max_point
    return points


def to_channels(points):
    print(points.shape)
    return points


def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    sb = subparsers.add_parser('convert')
    sb.set_defaults(fn=convert)
    sb.add_argument('input', nargs='?', type=argparse.FileType('r'))
    sb.add_argument('output', nargs='?', type=argparse.FileType('wb'))
    sb.add_argument('--points', type=int, default=1024)
    sb = subparsers.add_parser('convert_many')
    sb.set_defaults(fn=convert_many)
    sb.add_argument('base', nargs='?', type=Path)
    sb.add_argument('--points', type=int, default=1024)
    return parser


def convert(input, output, points, **kwargs):
    mesh = OFFMesh(input)
    points = sample_points(mesh, points)
    points = correct_points(points)
    points = points.transpose()
    if output is not None:
        np.save(output, points)
    else:
        print(points)


def _convert_wrapper(file, target, points):
    if target.exists():
        #print(f'{file.name} -> {target.name} [DONE]')
        return
    try:
        with open(file, 'r') as in_fd:
            with open(target, 'wb') as out_fd:
                convert(in_fd, out_fd, points)
        print(f'{file.name} -> {target.name}')
    except Exception as e:
        print(f'BAD: {file}')
        target.unlink()


def convert_many(base, points, **kwargs):
    def _calc_args(file):
        target = file.with_suffix('.npy')
        return file, target, points
    all_files = list(base.glob('**/*.off'))
    args = [_calc_args(f) for f in all_files]
    with multiprocessing.Pool() as p:
        p.starmap(_convert_wrapper, args, chunksize=16)


if __name__ == '__main__':
    args = make_parser().parse_args()
    if 'fn' not in args:
        make_parser().print_help()
    else:
        args.fn(**vars(args))
