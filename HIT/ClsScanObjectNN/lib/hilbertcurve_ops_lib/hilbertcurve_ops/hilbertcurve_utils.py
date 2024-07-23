import torch
import torch.nn as nn
import warnings
from torch.autograd import Function
from typing import *
Tensor = torch.Tensor

try:
    import hilbertcurve_ops._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os

    warnings.warn("Unable to load hilbertcurve_ops cpp extension. JIT Compiling.")

    _ext_src_root = f"{osp.dirname(__file__)}/_ext-src"
    _ext_sources = glob.glob(f"{_ext_src_root}/src/*.cpp") + glob.glob(f"{_ext_src_root}/src/*.cu")
    _ext_headers = glob.glob(f"{_ext_src_root}/include/*")

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5;8.0"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[f"{_ext_src_root}/include"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )


class HilbertOrderFromGridIndex(Function):
    @staticmethod
    def forward(ctx, grd_idx: Tensor, ndim: int, level: int, check: bool = False) -> Tensor:
        r"""
        Parameters:
            grd_idx : grid index. (batch_size, num_points, ndim)
            ndim    : dimension of grid index.
            level   : level of Hilbert curve.
            check   : check parameters.
        Returns:
            hlb_ord : Hilbert order. (batch_size, num_points, 1)
        """
        if check:
            assert grd_idx.dtype == torch.int32 and type(ndim) == int and type(level) == int, 'Type Error!'
            assert ndim == grd_idx.shape[2], 'Shape Error!'
            assert level > 0 and level <= 20 and ndim * level <= 60, 'Level Error!'
            assert torch.min(grd_idx) >= 0 and torch.max(grd_idx) < (1 << level), 'Grid Index Error!'

        out = _ext.distances_from_points(grd_idx.clone(), ndim, level)
        ctx.mark_non_differentiable(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


class GridIndexFromHilbertOrder(Function):
    @staticmethod
    def forward(ctx, hlb_ord: Tensor, ndim: int, level: int, check: bool = False) -> Tensor:
        r"""
        Parameters:
            hlb_ord : Hilbert order. (batch_size, num_points, 1)
            ndim    : dimension of grid index
            level   : level of Hilbert curve
            check   : check parameters
        Returns:
            grd_idx : grid index. (batch_size, num_points, ndim)
        """
        if check:
            assert hlb_ord.dtype == torch.int64 and type(ndim) == int and type(level) == int, 'Type Error!'
            assert level > 0 and level <= 20 and ndim * level <= 60, 'Level Error!'
            assert torch.min(hlb_ord) >= 0 and torch.max(hlb_ord) < (1 << (ndim * level)), 'Hilbert Order Error!'

        out = _ext.points_from_distances(hlb_ord.clone(), ndim, level)
        ctx.mark_non_differentiable(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


hlb_ord_from_grd_idx = HilbertOrderFromGridIndex.apply
grd_idx_from_hlb_ord = GridIndexFromHilbertOrder.apply