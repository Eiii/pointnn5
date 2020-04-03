import torch
import torch.nn as nn
from kaolin.models.PointNet import PointNetFeatureExtractor as Pointnet
from . import pointconv
from ..data.starcraft import parse_frame
from ..problem import sc2_loss


class SC2Scene(nn.Module):
    def __init__(self, name='SC2Scene',
                 latent_sizes = [2**5, 2**5, 2**5, 2**6, 2**6],
                 weight_hidden=[2**4, 2**4],
                 c_mid = 2**5,
                 final_hidden = [2**6, 2**6],
                 decode_hidden = [2**5, 2**5, 2**6, 2**6],
                 residual = True, norm=True, norm_type='batch', neighbors=8):
        super().__init__()
        self.name = name
        feat_size = 16
        self.make_encoder(feat_size, latent_sizes, neighbors, weight_hidden,
                          c_mid, final_hidden, residual, norm, norm_type)
        self.make_decoders(feat_size+latent_sizes[-1], decode_hidden, norm,
                           norm_type, residual)

    def make_encoder(self, feat_size, latent_sizes, neighbors, weight_hidden,
                     c_mid, final_hidden, residual, norm, norm_type):
        dim = 2
        self.pointconvs = nn.ModuleList()
        in_size = feat_size
        for ls in latent_sizes:
            a = {'neighbors': neighbors, 'c_in': in_size, 'weight_hidden': weight_hidden,
                 'c_mid': c_mid, 'final_hidden': final_hidden,
                 'c_out': ls, 'dim': dim, 'norm': norm, 'norm_type': norm_type, 'residual': residual}
            pc = pointconv.GlobalPointConv(**a)
            self.pointconvs.append(pc)
            in_size += ls

    def make_decoders(self, in_size, decode_hidden, norm, norm_type, residual):
        h_args = {'in_channels': in_size, 'feat_size': 1,
                  'layer_dims': decode_hidden, 'output_feat': 'pointwise',
                  'norm': norm, 'norm_type': norm_type, 'residual': residual}
        o_args = {'in_channels': in_size, 'feat_size': 7,
                  'layer_dims': decode_hidden, 'output_feat': 'pointwise',
                  'norm': norm, 'norm_type': norm_type, 'residual': residual}
        p_args = {'in_channels': in_size, 'feat_size': 2,
                  'layer_dims': decode_hidden, 'output_feat': 'pointwise',
                  'norm': norm, 'norm_type': norm_type, 'residual': residual}
        self.health_net = Pointnet(**h_args)
        self.shield_net = Pointnet(**h_args)
        self.ori_net = Pointnet(**o_args)
        self.pos_net = Pointnet(**p_args)

    def get_args(self, item):
        return item['pre'], item['mask']

    def assemble_frame(self, health, shield, ori, pos):
        return {'health': health.squeeze(1),
                'shields': shield.squeeze(1),
                'ori': ori,
                'pos': pos}

    def remove_pos(self, data):
        d = data[:, torch.arange(13), :]
        return d

    def forward_hack(self, pre_frame, mask, _):
        return self.forward(pre_frame, mask)

    def forward(self, pre_frame, mask):
        unit_feats = self.encode(pre_frame, mask)
        pred = self.predict(unit_feats)
        frame = self.assemble_frame(*pred)
        return frame

    def encode(self, pre_frame, mask):
        pre_pts = parse_frame(pre_frame)['pos'].transpose(1, 2)
        orig_feats = pre_frame.transpose(1, 2)
        in_feats = orig_feats
        for pc in self.pointconvs:
            pred_feats = pc(pre_pts, pre_pts, in_feats, mask)
            in_feats = torch.cat([in_feats, pred_feats], dim=2)
        return torch.cat([orig_feats, pred_feats], dim=2)

    def predict(self, unit_feats):
        healths = self.health_net(unit_feats)
        shields = self.shield_net(unit_feats)
        oris = self.ori_net(unit_feats)
        poss = self.pos_net(unit_feats)
        return healths, shields, oris, poss

class SC2SceneMultiStep(SC2Scene):
    def __init__(self, name='SC2SceneMultiStep',
                 latent_sizes = [2**5, 2**5, 2**5, 2**6, 2**6],
                 weight_hidden=[2**4, 2**4],
                 c_mid = 2**5,
                 final_hidden = [2**6, 2**6],
                 decode_hidden = [2**5, 2**5, 2**6, 2**6],
                 residual=True, norm=True, norm_type='batch', neighbors=8):
        super().__init__()

    def get_args(self, item):
        return item['pre'], item['offset'], item['mask']

    def make_decoders(self, in_size, decode_hidden, norm, norm_type, residual):
        return super().make_decoders(in_size+1, decode_hidden, norm,
                                     norm_type, residual)

    def forward(self, pre_frame, offset, mask):
        unit_feats = self.encode(pre_frame, mask)
        offset = offset.view(-1, 1, 1).repeat(1, unit_feats.size(1), 1)
        final_feats = torch.cat([unit_feats, offset.float()], dim=2)
        pred = self.predict(final_feats)
        frame = self.assemble_frame(*pred)
        return frame

class SC2SceneSample(SC2Scene):
    def __init__(self, name='SC2SceneSample',
                 latent_sizes = [2**5, 2**5, 2**5, 2**6, 2**6],
                 weight_hidden=[2**4, 2**4],
                 c_mid = 2**5,
                 final_hidden = [2**6, 2**6],
                 decode_hidden = [2**5, 2**5, 2**6, 2**6],
                 noise_size = 3,
                 residual = True, norm=True, norm_type='batch', neighbors=8):
        self.noise_size = noise_size
        super().__init__()

    def make_decoders(self, in_size, decode_hidden, norm, norm_type, residual):
        return super().make_decoders(in_size+self.noise_size, decode_hidden,
                                     norm, norm_type, residual)

    def make_training_noise(self, samples, ref):
        noise = torch.randn((1, 1, self.noise_size, samples),
                            device=ref.device)
        noise = noise.repeat(ref.size(0), ref.size(1), 1, 1)
        return noise

    def sample_predict(self, unit_feats, num_samples):
        unit_feats = unit_feats.unsqueeze(3).repeat(1, 1, 1, num_samples)
        noise = self.make_training_noise(num_samples, unit_feats)
        final_feats = torch.cat([unit_feats, noise], dim=2)
        refsize = final_feats.size()
        unrolled_input = final_feats.view(refsize[0], -1, refsize[2])
        preds = self.predict(unrolled_input)
        def group_preds(v):
            return v.view(v.size(0), v.size(1), refsize[1], refsize[3])
        return [group_preds(p) for p in preds]

    def choose_best(self, sample_preds, cheat_item, rand_chance=0.02):
        # Choose best prediction in each sample
        ls = []
        num_samples = sample_preds[0].size(3)
        for x in range(num_samples):
            args = [v.select(3, x) for v in sample_preds]
            temp_result = self.assemble_frame(*args)
            ls.append(sc2_loss(cheat_item, temp_result).sum(dim=0))
        all_ls = torch.stack(ls, dim=2)
        _, best_idxs = all_ls.min(dim=2, keepdim=True)
        rand_idxs = torch.randint(all_ls.size(2), best_idxs.size(),
                                  device=all_ls.device)
        r = torch.bernoulli(torch.tensor(rand_chance, dtype=torch.float).repeat(*rand_idxs.size()))
        r = r.bool().to(all_ls.device)
        chosen_idxs = torch.where(r, rand_idxs, best_idxs)
        def _gather(p):
            p = p.gather(3, chosen_idxs.unsqueeze(1).repeat(1, p.size(1), 1, 1))
            return p.squeeze(3)
        return [_gather(v) for v in sample_preds]

    def forward_hack(self, pre_frame, mask, cheat_item):
        unit_feats = self.encode(pre_frame, mask)
        num_samples = 5
        sample_preds = self.sample_predict(unit_feats, num_samples)
        best_preds = self.choose_best(sample_preds, cheat_item, rand_chance=1)
        result = self.assemble_frame(*best_preds)
        return result

    def forward(self, pre_frame, mask):
        unit_feats = self.encode(pre_frame, mask)
        noise = torch.randn((unit_feats.size(0), unit_feats.size(1), self.noise_size),
                            device=unit_feats.device)
        final_feats = torch.cat([unit_feats, noise], dim=2)
        pred = self.predict(final_feats)
        frame = self.assemble_frame(*pred)
        return frame
