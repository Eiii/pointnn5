import torch
import torch.nn as nn
import torch.nn.functional as F
from kaolin.models.PointNet import PointNetFeatureExtractor as Pointnet
from ..utils import RealToOnehot
from math import pi


def downsample_points(points, count, feats=None):
    idxs = torch.randperm(points.size(1))[:count]
    keys = points[:, idxs, :]
    if feats is None:
        return keys
    else:
        key_feats = feats[:, :, idxs]
        return keys, key_feats


def origin_point(ref):
    size = [ref.size(0), 1, 3]
    return torch.zeros(size, dtype=torch.float, device=ref.device)


def const_feats(points):
    f_size = list(points.shape)
    f_size[2] = 1
    return torch.ones(f_size, dtype=torch.float, device=points.device)


def closest_pts_to_keys(keys, points, pts_mask, neighbors):
    num_keys = keys.size(1)
    num_points = points.size(1)
    # Create matrices for 'cross product' - keys x points
    keys_expand = keys.unsqueeze(2).expand(-1, -1, num_points, -1)
    points_expand = points.unsqueeze(1).expand(-1, num_keys, -1, -1)
    # Calculate square distances
    diff = points_expand - keys_expand
    sqr_dist = (diff**2).sum(dim=3)
    # Add large value to masked out entries
    if pts_mask is not None:
        dup_mask = pts_mask.unsqueeze(1).expand(-1, num_keys, -1)
        big, _ = sqr_dist.max(dim=2, keepdim=True)
        mask_extra = 100.0*big*(1-dup_mask)
        sqr_dist += mask_extra
    _, idxs = sqr_dist.topk(neighbors, dim=2, largest=False, sorted=False)
    return idxs, diff


def group_rel(groups, points, pts_mask, neighbors):
    all_groups = torch.unique(groups)
    num_groups = len(all_groups)
    num_points = points.size(1)
    groups_x = groups.unsqueeze(2).expand(-1, -1, num_points)
    groups_y = groups.unsqueeze(1).expand(-1, num_points, -1)
    matches = (groups_x == groups_y)
    # Making some assumptions here
    _, _, idxs = matches.nonzero(as_tuple=True)
    idxs = idxs.reshape(groups.size(0), groups.size(1), neighbors)
    points_x = points.unsqueeze(2).expand(-1, -1, num_points, -1)
    points_y = points.unsqueeze(1).expand(-1, num_points, -1, -1)
    diff = points_x - points_y
    return idxs, diff


def idx_tensor(m, d=0):
    s = [-1 if i==d else 1 for i in range(m.dim())]
    b = torch.zeros_like(m) + torch.arange(m.size(d), device=m.device).view(*s)
    return b

def idx_tensor_manual(size, d, ref):
    m = torch.zeros(*size, device=ref.device)
    s = [-1 if i==d else 1 for i in range(m.dim())]
    b = m + torch.arange(m.size(d), device=ref.device).view(*s)
    return b


class GlobalPointConv(nn.Module):
    def __init__(self,
                 neighbors,
                 c_in,
                 weight_hidden,
                 c_mid,
                 final_hidden,
                 c_out,
                 norm=True,
                 relu=True,
                 dim=3,
                 pos_encoding='xy',
                 norm_type='batch',
                 residual=False):
        super().__init__()
        self.neighbor_count = neighbors
        self.relu = relu
        if pos_encoding == 'xy':
            weight_in = dim
        elif pos_encoding.startswith('xyoh'):
            min_, max_, steps = [int(s) for s in pos_encoding.split(':')[1:]]
            self.pos_encoder = RealToOnehot(min_, max_, steps, True)
            weight_in = dim*steps
        elif pos_encoding == 'rad':
            weight_in = dim
        elif pos_encoding.startswith('radoh'):
            max_, dist_steps, angle_steps = [int(s) for s in pos_encoding.split(':')[1:]]
            self.phi_encoder = RealToOnehot(-pi, pi, angle_steps, loop=True)
            self.theta_encoder = RealToOnehot(0, pi, angle_steps//2)
            self.r_encoder = RealToOnehot(0, max_, dist_steps)
            weight_in = angle_steps + angle_steps//2 + dist_steps
        else:
            raise ValueError(f'Unknown encoding `{pos_encoding}`')
        self.pos_encoding = pos_encoding.split(':')[0]
        self.weight_conv = Pointnet(in_channels=weight_in, feat_size=c_mid,
                                    layer_dims=list(weight_hidden),
                                    output_feat='pointwise',
                                    norm=norm, norm_type=norm_type,
                                    residual=residual)
        self.final_conv = Pointnet(in_channels=c_in*c_mid, feat_size=c_out,
                                   layer_dims=list(final_hidden),
                                   output_feat='pointwise',
                                   norm=norm, norm_type=norm_type,
                                   residual=residual)

    def encode_pos(self, pos):
        if self.pos_encoding == 'xy':
            return pos
        elif self.pos_encoding == 'xyoh':
            return self.pos_encoder(pos)
        elif self.pos_encoding.startswith('rad'):
            r = ((pos**2).sum(dim=-1))**0.5
            x = pos.select(-1, 0)
            y = pos.select(-1, 1)
            z = pos.select(-1, 2)
            phi = torch.atan2(x, y)
            theta = torch.atan2((x**2+y**2)**0.5, z)
            if self.pos_encoding == 'rad':
                return torch.stack((r, phi, theta), dim=-1)
            elif self.pos_encoding == 'radoh':
                roh = self.r_encoder(r.unsqueeze(-1))
                phioh = self.phi_encoder(phi.unsqueeze(-1))
                thetaoh = self.theta_encoder(theta.unsqueeze(-1))
                return torch.cat((roh, phioh, thetaoh), dim=-1)


    def forward(self, keys, points, feats, point_mask=None):
        pt_idxs, rel_pos = \
            closest_pts_to_keys(keys, points, point_mask, self.neighbor_count)
        # Shitty index calculation, TODO
        batch_idxs = idx_tensor(pt_idxs, 0)
        key_idxs = idx_tensor(pt_idxs, 1)
        # Get closest points, features per key
        closest_rel = rel_pos[batch_idxs, key_idxs, pt_idxs, :]
        closest_feats = feats[batch_idxs, pt_idxs, :]
        # These are still grouped in an extra dimension by keys
        # Since each key shares the same convolution, the extra dimension
        # can just be flattened and reconstructed later.
        cps = closest_rel.size()
        flat_rel = closest_rel.view(cps[0], cps[1]*cps[2], cps[3])
        flat_rel = self.encode_pos(flat_rel)
        flat_m = self.weight_conv(flat_rel)
        if self.relu:
            flat_m = F.relu(flat_m)
        m = flat_m.view(cps[0], -1, cps[1], cps[2])
        if point_mask is not None:
            temp_mask = point_mask[batch_idxs, pt_idxs]
            m *= temp_mask.unsqueeze(1)
            closest_feats *= temp_mask.unsqueeze(3)
        # Transpose for matrix multiplication
        mp = m.permute(0, 2, 1, 3)
        e = torch.matmul(mp, closest_feats)
        e = e.view(e.size(0), e.size(1), -1)
        final = self.final_conv(e).transpose(1, 2)
        return final

class GroupPointConv(nn.Module):
    def __init__(self,
                 neighbors,
                 c_in,
                 weight_hidden,
                 c_mid,
                 final_hidden,
                 c_out,
                 batchnorm=True,
                 relu=True,
                 dim=3,
                 pos_encoding='xy'):
        super().__init__()
        self.neighbor_count = neighbors
        self.relu = relu
        if pos_encoding == 'xy':
            weight_in = dim
        elif pos_encoding.startswith('xyoh'):
            min_, max_, steps = [int(s) for s in pos_encoding.split(':')[1:]]
            self.pos_encoder = RealToOnehot(min_, max_, steps, True)
            weight_in = dim*steps
        elif pos_encoding == 'rad':
            weight_in = dim
        elif pos_encoding.startswith('radoh'):
            max_, dist_steps, angle_steps = [int(s) for s in pos_encoding.split(':')[1:]]
            self.phi_encoder = RealToOnehot(-pi, pi, angle_steps, loop=True)
            self.theta_encoder = RealToOnehot(0, pi, angle_steps//2)
            self.r_encoder = RealToOnehot(0, max_, dist_steps)
            weight_in = angle_steps + angle_steps//2 + dist_steps
        else:
            raise ValueError(f'Unknown encoding `{pos_encoding}`')
        self.pos_encoding = pos_encoding.split(':')[0]
        self.weight_conv = Pointnet(in_channels=weight_in, feat_size=c_mid,
                                    layer_dims=list(weight_hidden),
                                    output_feat='pointwise',
                                    batchnorm=batchnorm)
        self.final_conv = Pointnet(in_channels=c_in*c_mid, feat_size=c_out,
                                   layer_dims=list(final_hidden),
                                   output_feat='pointwise',
                                   batchnorm=batchnorm)

    def encode_pos(self, pos):
        if self.pos_encoding == 'xy':
            return pos
        elif self.pos_encoding == 'xyoh':
            return self.pos_encoder(pos)
        elif self.pos_encoding.startswith('rad'):
            r = ((pos**2).sum(dim=-1))**0.5
            x = pos.select(-1, 0)
            y = pos.select(-1, 1)
            z = pos.select(-1, 2)
            phi = torch.atan2(x, y)
            theta = torch.atan2((x**2+y**2)**0.5, z)
            if self.pos_encoding == 'rad':
                return torch.stack((r, phi, theta), dim=-1)
            elif self.pos_encoding == 'radoh':
                roh = self.r_encoder(r.unsqueeze(-1))
                phioh = self.phi_encoder(phi.unsqueeze(-1))
                thetaoh = self.theta_encoder(theta.unsqueeze(-1))
                return torch.cat((roh, phioh, thetaoh), dim=-1)


    def forward(self, groups, points, feats, point_mask=None):
        pt_idxs, rel_pos = group_rel(groups, points, point_mask, self.neighbor_count)
        # Shitty index calculation, TODO
        batch_idxs = idx_tensor(pt_idxs, 0)
        key_idxs = idx_tensor(pt_idxs, 1)
        # Get closest points, features per key
        closest_rel = rel_pos[batch_idxs, key_idxs, pt_idxs, :]
        closest_feats = feats[batch_idxs, pt_idxs, :]
        # These are still grouped in an extra dimension by keys
        # Since each key shares the same convolution, the extra dimension
        # can just be flattened and reconstructed later.
        cps = closest_rel.size()
        flat_rel = closest_rel.view(cps[0], cps[1]*cps[2], cps[3])
        flat_rel = self.encode_pos(flat_rel)
        flat_m = self.weight_conv(flat_rel)
        if self.relu:
            flat_m = F.relu(flat_m)
        m = flat_m.view(cps[0], -1, cps[1], cps[2])
        flat_mask = point_mask.view(point_mask.size(0), -1)
        if point_mask is not None:
            temp_mask = flat_mask[batch_idxs, pt_idxs]
            m *= temp_mask.unsqueeze(1)
            closest_feats *= temp_mask.unsqueeze(3)
        # Transpose for matrix multiplication
        mp = m.permute(0, 2, 1, 3)
        e = torch.matmul(mp, closest_feats)
        e = e.view(e.size(0), e.size(1), -1)
        final = self.final_conv(e).transpose(1, 2)
        return final
