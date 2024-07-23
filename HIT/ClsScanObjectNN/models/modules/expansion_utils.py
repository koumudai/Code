import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Callable
Tensor = torch.Tensor
import hilbertcurve_ops.hilbertcurve_utils as hcops


# ---------------------------------------- Expansion Operations ---------------------------------------- #
def SpaceGather(v: Tensor, idx: Tensor) -> Tensor:                                              # (b_s, n_p, d_p), (b_s, n_p)
    v = torch.gather(v, dim=1, index=idx.unsqueeze(2).expand(-1, -1, v.size(2)))                # (b_s, n_p, d_p)
    return v


class SpaceExpansion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, z: Tensor, idx_pa: Tensor) -> Tuple[Tensor, Tensor]:           # (b_s, n_p, d_f), (b_s, n_p, d_c), (b_s, n_p)
        x = SpaceGather(x, idx_pa)                                                              # (b_s, n_p, d_f)
        z = SpaceGather(z, idx_pa)                                                              # (b_s, n_p, d_c)
        return x, z                                                                  


class SpaceReverse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, z: Tensor, idx_re: Tensor) -> Tuple[Tensor, Tensor]:           # (b_s, n_p, d_f), (b_s, n_p, d_c), (b_s, n_p)
        x = SpaceGather(x, idx_re)                                                              # (b_s, n_p, d_f)
        z = SpaceGather(z, idx_re)                                                              # (b_s, n_p, d_c)
        return x, z


# ---------------------------------------- Hilbert Expansion Operations ---------------------------------------- #
class Hilbert3DGetIdx(nn.Module):
    hlb_lvl_max = 20
    def __init__(self, hlb_lvl: int = 5, hlb_shift: bool = True) -> None:
        super().__init__()
        self.hlb_max = 2 ** (self.hlb_lvl_max)
        self.hlb_bias = 2 ** (self.hlb_lvl_max - hlb_lvl) - int(self.hlb_lvl_max - hlb_lvl != 0) if hlb_shift else 0
        self.hlb_lvl = self.hlb_lvl_max + (1 if hlb_shift else 0)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:                                      # (b_s, n_p, d_c)
        b_s, n_p, _ = z.size()
        z = z.to(torch.float64)
        z_ce = z.mean(dim=1, keepdim=True)                                                      # (b_s, 1, d_c)
        z_r = torch.sqrt(torch.max(torch.sum(torch.pow(z - z_ce, 2), dim=2), dim=1)[0]).reshape(b_s, 1, 1) + 1e-6   # (b_s, 1, 1)
        z_nm = (z - z_ce + z_r) / (2 * z_r)                                                     # (b_s, n_p, d_c)
        grd_idx = (self.hlb_max * z_nm).int() + self.hlb_bias                                   # (b_s, n_p, d_c)
        hlb_ord = hcops.hlb_ord_from_grd_idx(grd_idx, 3, self.hlb_lvl).reshape(b_s, n_p)        # (b_s, n_p)
        idx_pa = torch.sort(hlb_ord, dim=1)[1]                                                  # (b_s, n_p)
        idx_re = torch.argsort(idx_pa)                                                          # (b_s, n_p)
        return idx_pa, idx_re                                                                   # (b_s, n_p), (b_s, n_p)


# ---------------------------------------- Random Expansion Operations ---------------------------------------- #
class Random3DGetIdx(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:                                      # (b_s, n_p, d_c)
        b_s, n_p, _ = z.size()
        idx_pa = torch.stack([torch.randperm(n_p) for _ in range(b_s)]).to(z.device)            # (b_s, n_p)
        idx_rg = torch.arange(n_p, device=z.device).reshape(1, n_p, 1).expand(b_s, -1, -1)      # (b_s, n_p, 1)
        idx_re = torch.sort(SpaceGather(idx_rg, idx_pa).reshape(b_s, n_p), dim=1)[1]            # (b_s, n_p)
        return idx_pa, idx_re                                                                   # (b_s, n_p), (b_s, n_p)


# ---------------------------------------- Sorting Coordinates Expansion Operations ---------------------------------------- #
def torch_lexsort(x: Tensor) -> Tensor:
    # Reference from https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850
    # x: (b_s, n_p, d_c)
    assert x.ndim == 3
    idx = torch.stack([torch.argsort(torch.unique(e, dim=-1, sorted=True, return_inverse=True)[1]) for e in x.permute(0, 2, 1)])
    return idx


class SortCoord3DGetIdx(nn.Module):
    def __init__(self, order: str = 'xyz'):
        super().__init__()
        assert order in ['xyz', 'yxz']
        self.order = order

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:                                      # (b_s, n_p, d_c)
        b_s, n_p, _ = z.size()
        if self.order == 'yxz':
            z = z[:, :, [1, 0, 2]]
        idx_pa = torch_lexsort(z)                                                               # (b_s, n_p)
        idx_rg = torch.arange(n_p, device=z.device).reshape(1, n_p, 1).expand(b_s, -1, -1)      # (b_s, n_p, 1)
        idx_re = torch.sort(SpaceGather(idx_rg, idx_pa).reshape(b_s, n_p), dim=1)[1]            # (b_s, n_p)
        return idx_pa, idx_re                                                                   # (b_s, n_p), (b_s, n_p)


# ---------------------------------------- Sorting Grids And Coordinates Expansion Operations ---------------------------------------- #
class SortGridAndCoord3DGetIdx(nn.Module):
    def __init__(self, grd_lvl=4, order='xyz'):
        super().__init__()
        assert order in ['xyz', 'yxz']
        self.grd_sz = 2 ** grd_lvl
        self.order = order

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:                                      # (b_s, n_p, d_c)
        b_s, n_p, _ = z.size()
        if self.order == 'yxz':
            z = z[:, :, [1, 0, 2]]
        z = z.to(torch.float64)
        z_ce = z.mean(dim=1, keepdim=True)                                                      # (b_s, 1, d_c)
        z_r = torch.sqrt(torch.max(torch.sum(torch.pow(z - z_ce, 2), dim=2), dim=1)[0]).reshape(b_s, 1, 1) + 1e-6   # (b_s, 1, 1)
        z_nm = (z - z_ce + z_r) / (2 * z_r)                                                     # (b_s, n_p, d_c)
        grd = torch.floor_(self.grd_sz * z_nm) / self.grd_sz                                    # (b_s, n_p)
        idx_pa = torch_lexsort(torch.cat([grd, z_nm - grd], dim=2))                             # (b_s, n_p)
        idx_rg = torch.arange(n_p, device=z.device).reshape(1, n_p, 1).expand(b_s, -1, -1)      # (b_s, n_p, 1)
        idx_re = torch.sort(SpaceGather(idx_rg, idx_pa).reshape(b_s, n_p), dim=1)[1]            # (b_s, n_p)
        return idx_pa, idx_re                                                                   # (b_s, n_p), (b_s, n_p)