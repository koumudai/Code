import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Callable
Tensor = torch.Tensor
from .point_utils import *


# ---------------------------------------- Linear ---------------------------------------- #
class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, args: Dict = {'bias': False}) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, **{**{'bias': False}, **args})
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def __repr__(self) -> str:
        return f'Linear: {self.linear}'


# ---------------------------------------- Norm ---------------------------------------- #
class Norm(nn.Module):
    name_dict = {
        'BN0D': nn.BatchNorm1d,
        'BN1D': nn.BatchNorm1d,
        'BN2D': nn.BatchNorm2d,
        'LN0D': nn.LayerNorm,
        'LN1D': nn.LayerNorm,
        'LN2D': nn.LayerNorm,
    }

    args_dict = {
        'BN0D': {'eps': 1e-5, 'momentum': 0.1},
        'BN1D': {'eps': 1e-5, 'momentum': 0.1},
        'BN2D': {'eps': 1e-5, 'momentum': 0.1},
        'LN0D': {'eps': 1e-5},
        'LN1D': {'eps': 1e-5},
        'LN2D': {'eps': 1e-5},
    }

    def __init__(self, name: str, dim: int, d_feat: int, args: Dict = {}) -> None:
        super().__init__()
        assert name in ['BN', 'LN'] and dim in [0, 1, 2]
        self.name = f'{name}{dim}D'
        self.norm = self.name_dict[self.name](d_feat, **{**self.args_dict[self.name], **args})

    def forward(self, x: Tensor) -> Tensor:
        if self.name == 'BN1D':
            return self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        elif self.name == 'BN2D':
            return self.norm(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        else:
            return self.norm(x)
    
    def __repr__(self) -> str:
        return f'Norm({self.name}): ({self.norm})'


# ---------------------------------------- Act ---------------------------------------- #
class Act(nn.Module):
    name_dict = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'GELU': nn.GELU,
    }

    args_dict = {
        'ReLU': {'inplace': True},
        'LeakyReLU': {'negative_slope': 0.2, 'inplace': True},
        'GELU': {},
    }

    def __init__(self, name: str, args: Dict = {}) -> None:
        super().__init__()
        assert name in ['ReLU', 'LeakyReLU', 'GELU']
        self.name = name
        self.act = self.name_dict[self.name](**{**self.args_dict[self.name], **args})
    
    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)

    def __repr__(self) -> str:
        return f'Act({self.name}): {self.act}'


# ---------------------------------------- Pool ---------------------------------------- #
class Pool(nn.Module):
    def __init__(self, name: str = 'Max', dim: int = 1, keepdim: bool = False) -> None:
        super().__init__()
        assert name in ['Max', 'Avg', 'Sum']
        self.name, self.dim, self.keepdim = name, dim, keepdim
        self.pool = self._Pool(name=self.name, dim=dim, keepdim=keepdim)

    def _Pool(self, name: str, dim: int, keepdim: bool) -> Callable[[Tensor], Tensor]:
        if name == 'Max':
            return lambda x: torch.max(x, dim=dim, keepdim=keepdim)[0]
        elif name == 'Avg':
            return lambda x: torch.mean(x, dim=dim, keepdim=keepdim)
        elif name == 'Sum':
            return lambda x: torch.sum(x, dim=dim, keepdim=keepdim)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)

    def __repr__(self) -> str:
        return f'Pool({self.name}): {self.name}Pool(dim={self.dim}, keepdim={self.keepdim})'


# ---------------------------------------- MLP ---------------------------------------- #
class MLPND(nn.Module):
    def __init__(self, dim: int, d_in: int, d_out: int, norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False},
                norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        assert dim in [0, 1, 2]
        self.mlp = nn.Sequential(
            Linear(d_in, d_out, args=linear_args),
            Norm(name=norm_name, dim=dim, d_feat=d_out, args=norm_args),
            Act(name=act_name, args=act_args)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class ResMLPND(nn.Module):
    def __init__(self, dim: int, d_feat: int, ratio=1.0, norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False},
                norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        assert dim in [0, 1, 2]
        d_hid = int(round(d_feat * ratio))
        self.mlp = nn.Sequential(
            Linear(d_feat, d_hid, args=linear_args),
            Norm(name=norm_name, dim=dim, d_feat=d_hid, args=norm_args),
            Act(name=act_name, args=act_args),
            Linear(d_hid, d_feat, args=linear_args),
            Norm(name=norm_name, dim=dim, d_feat=d_feat, args=norm_args),
        )
        self.act = Act(name=act_name, args=act_args)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.mlp(x) + x)


# ---------------------------------------- Sequential Window Attention Modules ---------------------------------------- #
class SeqWinAttn(nn.Module):
    def __init__(self, win_sz: int, d_feat: int, n_head: int, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        assert d_feat % n_head == 0
        self.w_s, self.d_f, self.n_h, self.d_h = win_sz, d_feat, n_head, d_feat // n_head
        self.scale = self.d_h ** -0.5
        self.w_qkv = nn.Linear(d_feat, 3*d_feat, bias=False)
        # self.w_z = nn.Sequential(nn.Linear(3, d_feat//4, bias=False), nn.ReLU(inplace=True), nn.Linear(d_feat//4, n_head, bias=False))
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_feat, d_feat)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:                                      # (b_s, n_p, d_f), (b_s, n_p, d_c)
        b_s, n_p, n_w, w_s, n_h, d_h = x.size(0), x.size(1), x.size(1) // self.w_s, self.w_s, self.n_h, self.d_h
        q, k, v = self.w_qkv(x).reshape(b_s, n_w, w_s, 3, n_h, d_h).permute(3, 0, 1, 4, 2, 5).contiguous()  # (b_s, n_w, n_h, w_s, d_h), (b_s, n_w, n_h, w_s, d_h), (b_s, n_w, n_h, w_s, d_h)
        attn = (self.scale * q) @ k.transpose(-2, -1)                                       # (b_s, n_w, n_h, w_s, w_s)
        # z = z.reshape(b_s, n_w, w_s, -1)                                                    # (b_s, n_w, w_s, d_c)
        # p = self.w_z(z.unsqueeze(3)-z.unsqueeze(2)).permute(0, 1, 4, 2, 3).contiguous()     # (b_s, n_w, n_h, w_s, w_s)
        # attn += torch.sigmoid(p)                                                            # (b_s, n_w, n_h, w_s, w_s)
        attn = self.attn_drop(self.softmax(attn))                                           # (b_s, n_w, n_h, w_s, w_s)
        x = (attn @ v).permute(0, 1, 3, 2, 4).contiguous().reshape(b_s, n_p, -1)            # (b_s, n_p, d_f)
        x = self.proj_drop(self.proj(x))                                                    # (b_s, n_p, d_f)
        return x                                                                            # (b_s, n_p, d_f)


class FFN(nn.Module):
    def __init__(self, d_feat: int, ratio: float = 1.0, norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False},
                norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        d_hid = int(round(d_feat*ratio))
        self.mlp = nn.Sequential(
            MLPND(dim=1, d_in=d_feat, d_out=d_hid, norm_name=norm_name, act_name=act_name,
                linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            MLPND(dim=1, d_in=d_hid, d_out=d_feat, norm_name=norm_name, act_name=act_name,
                linear_args=linear_args, norm_args=norm_args, act_args=act_args),
        )

    def forward(self, x: Tensor) -> Tensor:                                                 # (b_s, n_p, d_f)
        return self.mlp(x)                                                                  # (b_s, n_p, d_f)


class SeqWinTransLayer(nn.Module):
    def __init__(self, win_sz: int, d_feat: int, n_head: int, ratio: float = 0.5, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN', act_name: str = 'ReLU',
                linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.norm1 = Norm(name=norm_name, dim=1, d_feat=d_feat, args=norm_args)
        self.norm2 = Norm(name=norm_name, dim=1, d_feat=d_feat, args=norm_args)
        self.attn = SeqWinAttn(win_sz=win_sz, d_feat=d_feat, n_head=n_head, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ffn = FFN(d_feat=d_feat, ratio=ratio, norm_name=norm_name, act_name=act_name,
                linear_args=linear_args, norm_args=norm_args, act_args=act_args)

    def forward(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:                       # (b_s, n_p, d_f), (b_s, n_p, d_c)
        x = x + self.attn(self.norm1(x), z)                                                 # (b_s, n_p, d_f)
        x = x + self.ffn(self.norm2(x))                                                     # (b_s, n_p, d_f)
        return x, z                                                                         # (b_s, n_p, d_f), (b_s, n_, d_c)


# ---------------------------------------- Sequential KNN Attention Modules ---------------------------------------- #
class SeqKNNAttn(nn.Module):
    def __init__(self, num_grp: int, d_feat: int, n_head: int, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        assert d_feat % n_head == 0
        self.n_g, self.d_f, self.n_h, self.d_h = num_grp, d_feat, n_head, d_feat // n_head
        self.scale = self.d_h ** -0.5
        self.w_qkv = nn.Linear(d_feat, 3*d_feat, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_feat, d_feat)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:                                                      # (b_s, n_p, d_f), (b_s, n_p, d_c)
        b_s, n_p, n_g, n_h, d_h = x.size(0), x.size(1), self.n_g, self.n_h, self.d_h
        p = torch.arange(n_p, dtype=torch.float32, device=z.device).reshape(1, n_p, 1).repeat([b_s, 1, 1])  # (b_s, n_p, 1)
        u_ne = knn_query(p, p, num_grp=n_g)                                                                 # (b_s, n_p, n_g)
        q, k, v = self.w_qkv(x).reshape(b_s, n_p, 3, -1).permute(2, 0, 1, 3).contiguous()                   # (b_s, n_p, d_f), (b_s, n_p, d_f), (b_s, n_p, d_f)
        q = q.reshape(b_s, n_p, 1, n_h, d_h).permute(0, 1, 3, 2, 4).contiguous()                            # (b_s, n_p, n_h, 1, d_h)
        k = gather_points(k, u_ne).reshape(b_s, n_p, n_g, n_h, d_h).permute(0, 1, 3, 2, 4).contiguous()     # (b_s, n_p, n_h, n_g, d_h)
        v = gather_points(v, u_ne).reshape(b_s, n_p, n_g, n_h, d_h).permute(0, 1, 3, 2, 4).contiguous()     # (b_s, n_p, n_h, n_g, d_h)
        attn = (self.scale * q) @ k.transpose(-2, -1)                                                       # (b_s, n_p, n_h, 1, n_g)
        attn = self.attn_drop(self.softmax(attn))                                                           # (b_s, n_p, n_h, 1, n_g)
        x = (attn @ v).reshape(b_s, n_p, -1)                                                                # (b_s, n_p, d_f)
        x = self.proj_drop(self.proj(x))                                                                    # (b_s, n_p, d_f)
        return x


class SeqKNNTransLayer(nn.Module):
    def __init__(self, num_grp: int, d_feat: int, n_head: int, ratio: float = 0.5, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN', act_name: str = 'ReLU',
                linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.norm1 = Norm(name=norm_name, dim=1, d_feat=d_feat, args=norm_args)
        self.norm2 = Norm(name=norm_name, dim=1, d_feat=d_feat, args=norm_args)
        self.attn = SeqKNNAttn(num_grp, d_feat=d_feat, n_head=n_head, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ffn = FFN(d_feat=d_feat, ratio=ratio, norm_name=norm_name, act_name=act_name,
                linear_args=linear_args, norm_args=norm_args, act_args=act_args)

    def forward(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:                       # (b_s, n_p, d_f), (b_s, n_p, d_c)
        x = x + self.attn(self.norm1(x), z)                                                 # (b_s, n_p, d_f)
        x = x + self.ffn(self.norm2(x))                                                     # (b_s, n_p, d_f)
        return x, z                                                                         # (b_s, n_p, d_f), (b_s, n_, d_c)


# ---------------------------------------- Stem Modules ---------------------------------------- #
class PointStem(nn.Module):
    def __init__(self, d_in: int = 3, d_out: int = 64, norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False},
                norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.mlp = MLPND(dim=1, d_in=d_in, d_out=d_out, norm_name=norm_name, act_name=act_name,
                linear_args=linear_args, norm_args=norm_args, act_args=act_args)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:                                  # (b_s, n_p, d_c)
        return self.mlp(z), z                                                               # (b_s, n_p, d_out), (b_s, n_p, d_c)


# ---------------------------------------- Transition Down Modules ---------------------------------------- #
class PointTransitionDown(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_qry: int, num_grp: int, n_block: int = 2, ratio: float = 1.0, norm_name: str = 'BN', act_name: str = 'ReLU',
                linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.num_qry, self.num_grp = num_qry, num_grp
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, d_in]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, d_in]))
        self.mlp = nn.Sequential(
            MLPND(dim=2, d_in=d_in*2, d_out=d_out, norm_name=norm_name, act_name=act_name,
                    linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            *[ResMLPND(dim=2, d_feat=d_out, ratio=ratio, norm_name=norm_name, act_name=act_name,
                    linear_args=linear_args, norm_args=norm_args, act_args=act_args) for _ in range(n_block)],
            Pool(name='Max', dim=2, keepdim=False)
        )

    def forward(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:                       # (b_s, n_p, d_f), (b_s, n_p, d_c)
        b_s, n_q, n_g = z.size(0), self.num_qry, self.num_grp
        u_ce = furthest_point_sampling(z, n_q)                                              # (b_s, n_q)
        x_ce = gather_points(x, u_ce)                                                       # (b_s, n_q, d_f)
        z_ce = gather_points(z, u_ce)                                                       # (b_s, n_q, d_c)
        u_ne = knn_query(z, z_ce, n_g)                                                      # (b_s, n_q, n_g)
        x_ne = gather_points(x, u_ne)                                                       # (b_s, n_q, n_g, d_f)
        # z_ne = gather_points(z, u_ne)                                                     # (b_s, n_q, n_g, d_c)
        x_mu = x_ce.unsqueeze(dim=-2)                                                       # (b_s, n_q,   1, d_f)
        x_std = torch.std((x_ne - x_mu).reshape(b_s, -1), dim=-1).reshape(b_s, 1, 1, 1)     # (b_s,   1,   1,   1)
        x_ne = self.affine_alpha * (x_ne - x_mu) / (x_std + 1e-5) + self.affine_beta        # (b_s, n_q, n_g, d_f)
        x_ne = torch.cat([x_ne, x_ce.unsqueeze(-2).expand(-1, -1, n_g, -1)], dim=-1)        # (b_s, n_q, n_g, 2 * d_f)
        x_ce = self.mlp(x_ne)                                                               # (b_s, n_q, d_out)
        return x_ce, z_ce                                                                   # (b_s, n_q, d_out), (b_s, n_q, d_c)


# ---------------------------------------- Head Modules ---------------------------------------- #
class PointClsHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False},
                norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.head = nn.Sequential(
            MLPND(dim=0, d_in=d_in, d_out=512, norm_name=norm_name, act_name=act_name,
                    linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            nn.Dropout(0.5),
            MLPND(dim=0, d_in=512, d_out=256, norm_name=norm_name, act_name=act_name,
                    linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            nn.Dropout(0.5),
            nn.Linear(256, d_out, bias=True)
        )

    def forward(self, x: Tensor) -> Tensor:                                                 # (b_s, d_in)
        return self.head(x)                                                                 # (b_s, d_out)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # FOR LINEAR DEBUG
    print(Linear(10, 12))
    print(Linear(10, 12)(torch.zeros((8, 10))).shape)
    print(Linear(10, 12)(torch.zeros((8, 5, 10))).shape)
    print(Linear(10, 12)(torch.zeros((8, 5, 6, 10))).shape)

    # FOR NORM DEBUG
    print(Norm('BN', 0, 3))
    print(Norm('BN', 1, 3))
    print(Norm('BN', 2, 3))
    print(Norm('LN', 0, 3))
    print(Norm('LN', 1, 3))
    print(Norm('LN', 2, 3))
    print(Norm('BN', 0, 3)(torch.zeros(8, 3)).shape)
    print(Norm('BN', 1, 3)(torch.zeros(8, 5, 3)).shape)
    print(Norm('BN', 2, 3)(torch.zeros(8, 5, 6, 3)).shape)
    print(Norm('LN', 0, 3)(torch.zeros(8, 3)).shape)
    print(Norm('LN', 1, 3)(torch.zeros(8, 5, 3)).shape)
    print(Norm('LN', 2, 3)(torch.zeros(8, 5, 6, 3)).shape)

    # FOR ACT DEBUG
    print(Act('ReLU'))
    print(Act('LeakyReLU', {'negative_slope': 0.1}))
    print(Act('GELU'))
    print(Act('ReLU')(torch.zeros((8, 10))).shape)
    print(Act('LeakyReLU', {'negative_slope': 0.1})(torch.zeros((8, 10))).shape)
    print(Act('GELU')(torch.zeros((8, 10))).shape)

    # FOR POOL DEBUG
    print(Pool('Max', dim=0))
    print(Pool('Avg', dim=1))
    print(Pool('Sum', dim=2))
    print(Pool('Max', dim=0)(torch.zeros(8, 10, 3)).shape)
    print(Pool('Avg', dim=1)(torch.zeros(8, 10, 3)).shape)
    print(Pool('Sum', dim=2)(torch.zeros(8, 10, 3)).shape)

    # FOR MLP DEBUG
    print(MLPND(2, 10, 20))
    print(ResMLPND(2, 10, 2.0))
    print(MLPND(2, 10, 20)(torch.zeros(8, 12, 16, 10)).shape)
    print(ResMLPND(2, 10, 2.0)(torch.zeros(8, 12, 16, 10)).shape)