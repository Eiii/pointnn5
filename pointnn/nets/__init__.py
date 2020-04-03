from ..utils import argument_defaults
from .aenc import MLP, NoiseAppend, PointConvSample
from .sc2 import SC2Scene, SC2SceneSample, SC2SceneMultiStep

_aenc = [
    MLP,
    NoiseAppend,
    PointConvSample
]

_sc2 = [
    SC2Scene,
    SC2SceneSample,
    SC2SceneMultiStep
]

_available = _aenc + _sc2


def make_net(class_name, **kwargs):
    """ Instiantiate a network by name """
    class_ = {n.__name__: n for n in _available}[class_name]
    params = argument_defaults(class_.__init__)
    params.update(kwargs)
    net = class_(**params)
    net.args = params
    return net


def make_net_args(class_name, args):
    class_ = {n.__name__: n for n in _available}[class_name]
    return class_(**args)
