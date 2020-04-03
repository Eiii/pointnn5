""" Autoencoder networks """
import torch
import torch.nn as nn
import itertools

from kaolin.models.PointNet import PointNetFeatureExtractor as Pointnet

from ..utils import pairwise
from . import pointconv
from . import mlp

DEFAULT_FEAT = 2**9
DEFAULT_ENCODE = [2**6, 2**6, 2**8, 2**8, 2**9]
DEFAULT_DECODE = [2**10, 2**9, 2**9, 2**8, 2**8, 2**7]
DEFAULT_DIST = [2**9, 2**8, 2**8, 2**9]
DEFAULT_MLP_DECODE = [2**9, 2**10, 2**9, 2**8, 2**7]


class Autoencoder(nn.Module):
    def get_args(self, item):
        points, class_ = item
        return [points]


class MLP(Autoencoder):
    def __init__(self,
                 name='MLP',
                 encode_sizes=DEFAULT_ENCODE,
                 feat_size=DEFAULT_FEAT,
                 decode_sizes=DEFAULT_MLP_DECODE,
                 batch_norm=True,
                 out_count=2**10):
        super().__init__()
        self.name = name
        self.encode = Pointnet(feat_size=feat_size, layer_dims=encode_sizes)
        self.decode = mlp.create_decode(decode_sizes, out_count, batch_norm)

    def forward(self, points, out_count=2**10):
        glob_feat = self.encode(points)
        out_points = self.decode(glob_feat)
        out_points = out_points.reshape(out_points.shape[0], -1, 3)
        return out_points


class NoiseAppend(Autoencoder):
    def __init__(self,
                 name='NoiseAppend',
                 encode_sizes=DEFAULT_ENCODE,
                 feat_size=DEFAULT_FEAT,
                 decode_sizes=DEFAULT_DECODE,
                 noise_size=2**5):
        super().__init__()
        self.name = name
        self.noise_size = noise_size
        lcl_encode_sizes = list(encode_sizes)
        lcl_decode_sizes = list(decode_sizes)
        lcl_decode_sizes[0] += self.noise_size
        self.encode = Pointnet(feat_size=feat_size,
                               layer_dims=lcl_encode_sizes)
        self.decode = Pointnet(in_channels=feat_size+noise_size, feat_size=3,
                               layer_dims=lcl_decode_sizes,
                               output_feat='pointwise', transposed_input=True,
                               final_norm=False)

    def gen_noise(self, glob_feat, out_count):
        # Sample noise vector
        noise = torch.randn(glob_feat.shape[0], self.noise_size, out_count,
                            device=glob_feat.device)
        return noise

    def forward(self, points, out_count=2**10):
        glob_feat = self.encode(points)
        # Add noise - First duplicate the reduced global feature vector
        # `out_count` times, once per point to be decoded.
        samp_feat = glob_feat.unsqueeze(2).repeat(1, 1, out_count)
        # Append noise to duplicated global features, giving us samples from
        # the encoded point fetaure distribution.
        noise = self.gen_noise(glob_feat, out_count)
        samp_feat = torch.cat([samp_feat, noise], dim=1)
        out_points = self.decode(samp_feat)
        out_points = out_points.transpose(1, 2)
        return out_points


DEF_DECODE_HIDDEN = [2**10, 2**10, 2**9, 2**9, 2**8, 2**8, 2**7]
DEF_PC_PARAMS = {'neighbors': 32, 'weight_hidden': [32, 64, 128],
                 'final_hidden': [256, 256, 256], 'norm': True}
LAYER_PC_PARAMS = [
    {'c_in':  1, 'c_mid': 128, 'c_out':  8, 'final_hidden': [64, 32]},
    {'c_in':  8, 'c_mid': 64, 'c_out': 32},
    {'neighbors': 256, 'c_in': 32, 'c_mid': 64,
     'final_hidden': [1024, 512, 512]},
]
class PointConvSample(Autoencoder):
    def __init__(self,
                 name="PointConvSample",
                 latent_size=256,
                 key_counts=[-1, 256, 0],
                 default_params=DEF_PC_PARAMS,
                 layer_params=LAYER_PC_PARAMS,
                 final_type='mean',
                 decode_hidden=DEF_DECODE_HIDDEN,
                 noise_size=32):
        super().__init__()
        self.name = name
        self.key_counts = list(key_counts)
        self.final_type = final_type
        self.noise_size = 32
        # Check params
        assert len(key_counts) == len(layer_params), "Layer/key mismatch"
        #assert 'c_out' not in layer_params[-1], "Last layer has latent size set"
        lcl_layer_params = list(layer_params)
        lcl_layer_params[-1]['c_out'] = latent_size
        # Set up network
        self.pointconvs = nn.ModuleList()
        for p in lcl_layer_params:
            _p = dict(default_params)
            _p.update(p)
            pc = pointconv.GlobalPointConv(**_p)
            self.pointconvs.append(pc)

        lcl_decode_hidden = list(decode_hidden)
        self.decode = Pointnet(in_channels=latent_size+self.noise_size,
                               feat_size=3, layer_dims=lcl_decode_hidden,
                               output_feat='pointwise', transposed_input=True,
                               final_norm=False)

    def encode(self, points):
        if self.final_type == 'mean':
            mean_pt = points.mean(dim=1, keepdim=True)
        feats = pointconv.const_feats(points)
        for next_count, pc_net in zip(self.key_counts, self.pointconvs):
            if next_count > 0:
                next_points = pointconv.downsample_points(points, next_count)
            elif next_count == -1:
                next_points = points
            else:
                if self.final_type == 'origin':
                    next_points = pointconv.origin_point(points)
                elif self.final_type == 'mean':
                    next_points = mean_pt
            next_feats = pc_net(next_points, points, feats)
            points = next_points
            feats = next_feats
        return feats.squeeze(1)

    def gen_noise(self, feat, out_count):
        # Sample noise vector
        noise = torch.randn(feat.shape[0], self.noise_size, out_count,
                            device=feat.device)
        return noise

    def sample(self, mean, logvar, out_count):
        samp = mean.unsqueeze(2).repeat(1, 1, out_count)
        n = torch.randn_like(samp) * torch.exp(0.5 * logvar).unsqueeze(2)
        return samp + n

    def forward(self, points, out_count=2**10):
        shape_feat = self.encode(points)
        samp_feat = shape_feat.unsqueeze(2).repeat(1, 1, out_count)
        noise = self.gen_noise(samp_feat, out_count)
        samp_feat = torch.cat([samp_feat, noise], dim=1)
        out_points = self.decode(samp_feat)
        out_points = out_points.transpose(1, 2)
        return out_points
