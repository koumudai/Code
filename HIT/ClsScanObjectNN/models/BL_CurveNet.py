import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.points_utils import *
from .modules.models_utils import *


class TransitionDown(nn.Module):
    def __init__(self, n_g, r_g, k_g):
        super().__init__()
        self.n_g, self.r_g, self.k_g = n_g, r_g, k_g

    def forward(self, f, p):
        n_p, n_g, r_g, k_g = f.shape[1], self.n_g, self.r_g, self.k_g
        # print(f.shape, p.shape, n_p, n_g, r_g, k_g)
        if n_p == n_g:
            return f, p
        u_ce = farthest_point_sample(p, n_g)
        p_ce = index_points(p, u_ce)
        u_ne = query_points_ball(p, p_ce, k_g, r_g) 
        f_ne = index_points(f, u_ne)
        f_ce = torch.max(f_ne, dim=-2)[0]
        return f_ce, p_ce


class CIC(nn.Module):
    def __init__(self, n_g, r_g, k_g, d_in, d_out, ratio=2, act: Tuple[str, Dict]=('LeakyReLU', {})):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.ratio = ratio
        self.n_g = n_g
        self.r_g = r_g
        self.k_g = k_g
        d_hid = d_in // ratio

        self.dn = TransitionDown(n_g, r_g, k_g)
        if d_in != d_out:
            self.shortcut = MLPND(d_in, d_out, 1, act=('None', {}))

        self.mlp_pre = MLPND(d_in, d_hid, 1, act=act)
        self.p2f = MLPND(9, d_hid, 2, act=('None', {}))
        self.mlp = nn.Sequential(*[Act(act=act), MLPND(d_hid, d_hid, 2, act=act), Pool(name='Max', dim=2)])
        self.mlp_pst = MLPND(d_hid, d_out, 1, act=('None', {}))
        self.act = Act(act=act)

    def forward(self, f, p):
        if p.size(1) != self.n_g:
            f, p = self.dn(f, p)
        res = f
        if self.d_in != self.d_out:
            res = self.shortcut(res)
        f = self.mlp_pre(f)  # bs, c', n
        u_ne = query_points_knn(p, p, self.k_g)                     # b_s, n_p, k_g
        f_ne = index_points(f, u_ne) - f.unsqueeze(2)               # b_s, n_p, k_g, d_in
        p_ne = index_points(p, u_ne)                                # b_s, n_p, k_g, d_p
        p_ce = p.unsqueeze(2).expand(-1, -1, self.k_g, -1)          # b_s, n_p, k_g, d_p 
        p_ne = torch.cat([p_ce, p_ne, p_ne-p_ce], dim=3)            # b_s, n_p, k_g, 3*d_p
        f = self.mlp(f_ne + self.p2f(p_ne))                         # b_s, n_p, d_out
        f = self.act(self.mlp_pst(f) + res)
        return f, p


class PointHead(nn.Module):
    def __init__(self, d_in, d_out) -> None:
        super().__init__()    
        self.mlp1 = MLPND(d_in, d_in*2, 1, act=('ReLU', {}))
        self.pool1, self.pool2 = Pool('Max', 1), Pool('Avg', 1)
        self.mlp2 = nn.Sequential(MLPND(d_in*4, d_in, 0, act=('ReLU', {})), nn.Dropout(p=0.5), nn.Linear(d_in, d_out))

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(torch.cat([self.pool1(x), self.pool2(x)], dim=1))
        return x


class CurveNet(nn.Module):
    def __init__(self, n_cls=40, d_in=3, d_stem=32, k_g=20, act=('LeakyReLU', {})):
        super().__init__()
        self.stem = MLPND(d_in, d_stem, 1, act=act)
        # encoder
        self.cic11 = CIC(n_g=1024, r_g=0.05, k_g=k_g, d_in=d_stem, d_out=64, ratio=2)
        self.cic12 = CIC(n_g=1024, r_g=0.05, k_g=k_g, d_in=64, d_out=64, ratio=4)
        
        self.cic21 = CIC(n_g=1024, r_g=0.05, k_g=k_g, d_in=64, d_out=128, ratio=2)
        self.cic22 = CIC(n_g=1024, r_g=0.1, k_g=k_g, d_in=128, d_out=128, ratio=4)

        self.cic31 = CIC(n_g=256, r_g=0.1, k_g=k_g, d_in=128, d_out=256, ratio=2)
        self.cic32 = CIC(n_g=256, r_g=0.2, k_g=k_g, d_in=256, d_out=256, ratio=4)

        self.cic41 = CIC(n_g=64, r_g=0.2, k_g=k_g, d_in=256, d_out=512, ratio=2)
        self.cic42 = CIC(n_g=64, r_g=0.4, k_g=k_g, d_in=512, d_out=512, ratio=4)
        self.head = PointHead(512, n_cls)

    def forward(self, p):
        f0, p0 = p, p
        f0 = self.stem(f0)

        f1, p1 = self.cic11(f0, p0)
        f1, p1 = self.cic12(f1, p1)

        f2, p2 = self.cic21(f1, p1)
        f2, p2 = self.cic22(f2, p2)

        f3, p3 = self.cic31(f2, p2)
        f3, p3 = self.cic32(f3, p3)

        f4, p4 = self.cic41(f3, p3)
        f4, p4 = self.cic42(f4, p4)

        x = self.head(f4)

        return x 


if __name__ == '__main__':
    b_s, n_p, d_c = 2, 1024, 3
    f = torch.rand((b_s, n_p, d_c))
    p = torch.rand((b_s, n_p, d_c))
    m = CurveNet()
    print(m(p, f).shape)
