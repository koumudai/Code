import os
import torch
import torch.nn as nn
import torch.nn.functional as F
if __name__ == '__main__':
    from modules.point_utils import *
    from modules.model_utils import *
    from modules.expansion_utils import *
else:
    from models.modules.point_utils import *
    from models.modules.model_utils import *
    from models.modules.expansion_utils import *
    

# ---------------------------------------- Base Sequential Transformer ---------------------------------------- #
class BaseSeqTransBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:                                   # (b_s, n_p, d_f), (b_s, n_p, d_c)
        idx_pa_1, idx_re_1 = self.gi1(z)                                                                # (b_s, n_p), (b_s, n_p)
        idx_pa_2, idx_re_2 = self.gi2(z)                                                                # (b_s, n_p), (b_s, n_p)
        for i, (gp, st, gr) in enumerate(self.blocks):
            x, z = gp(x, z, idx_pa_1 if i % 2 == 0 else idx_pa_2)                                       # (b_s, n_p, d_f), (b_s, n_p, d_c)
            x, z = st(x, z)                                                                             # (b_s, n_p, d_f), (b_s, n_p, d_c)
            x, z = gr(x, z, idx_re_1 if i % 2 == 0 else idx_re_2)                                       # (b_s, n_p, d_f), (b_s, n_p, d_c)
        return x, z                                                                                     # (b_s, n_p, d_f), (b_s, n_p, d_c)


class BaseSeqTransBackBone(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_parms(self, num_stg: int = 4, d_b: int = 64, d_e: float = 2.0, q_b: int = 1024, q_e: float = 0.5, g_b: int = 24, g_e: float = 1.0) -> None:
        def calc(p: float, q: float, r: int) -> int:
            return int(round(p * (q ** r)))

        self.d_is = [calc(d_b, d_e, i) for i in range(num_stg)]                                         # [48, 96, 192, 384]
        self.d_os = [min(calc(d_b, d_e, i), 512) for i in range(1, num_stg+1)]                          # [96, 192, 384, 512]
        self.n_qs = [max(calc(q_b, q_e, i), 1) for i in range(1, num_stg+1)]                            # [512, 256, 128, 64]
        self.n_gs = [min(calc(g_b, g_e, i), self.n_qs[i]) for i in range(num_stg)]                      # [24, 24, 24, 24]
        print(f'd_is: {self.d_is}, d_os: {self.d_os}, n_ps: {self.n_qs}, g_ps: {self.n_gs}')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, z: Tensor) -> Tensor:                                                             # (b_s, n_p, d_c)
        assert z.size(1) == 1024
        x, z = self.stem(z)                                                                             # (b_s, n_p, d_f), (b_s, n_p, d_c)
        for dn, en in zip(self.dn_blocks, self.en_blocks):
            x, z = dn(x, z)                                                                             # (b_s, n_p', d_f'), (b_s, n_p', d_c')
            x, z = en(x, z)                                                                             # (b_s, n_p', d_f'), (b_s, n_p', d_c')
        x = self.head(self.pool(x))                                                                     # (b_s, d_cls)
        return x

# ---------------------------------------- Hilbert Sequential Window Transformer ---------------------------------------- #
class HilbertSeqWinTransBlock(BaseSeqTransBlock):
    def __init__(self, n_block: int, hlb_lvl: int, win_sz: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.gi1 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=False)
        self.gi2 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=True)
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqWinTransLayer(win_sz=win_sz, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])


class HilbertFormerV1ClsHilbertSeqWinTrans(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], hlb_lvl: List[int] = [7, 6, 5, 4], win_sz: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.hlb_lvl, self.win_sz = hlb_lvl, [min(win_sz, n_p) for win_sz, n_p in zip(win_sz, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, h_l, w_s in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.hlb_lvl, self.win_sz):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(HilbertSeqWinTransBlock(n_block=b_e, hlb_lvl=h_l, win_sz=w_s, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)

# ---------------------------------------- Hilbert Sequential Window Transformer ---------------------------------------- #
class HilbertSeqWinTransBlockV3(nn.Module):
    def __init__(self, n_block: int, hlb_lvl: int, win_sz: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        '''
        重复多次 + 旋转
        '''
        self.gi1 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=False)
        self.gi2 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=False)
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqWinTransLayer(win_sz=win_sz, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])

    def forward(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:                                   # (b_s, n_p, d_f), (b_s, n_p, d_c)
        idx_pa_1, idx_re_1 = self.gi1(z)                                                                # (b_s, n_p), (b_s, n_p)
        idx_pa_2, idx_re_2 = self.gi2(z[:, :, [1, 0, 2]])                                               # (b_s, n_p), (b_s, n_p)
        for i, (gp, st, gr) in enumerate(self.blocks):
            x, z = gp(x, z, idx_pa_1 if i % 2 == 0 else idx_pa_2)                                       # (b_s, n_p, d_f), (b_s, n_p, d_c)
            x, z = st(x, z)                                                                             # (b_s, n_p, d_f), (b_s, n_p, d_c)
            x, z = gr(x, z, idx_re_1 if i % 2 == 0 else idx_re_2)                                       # (b_s, n_p, d_f), (b_s, n_p, d_c)
        return x, z            


class HilbertFormerV1ClsHilbertSeqWinTransV3(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], hlb_lvl: List[int] = [7, 6, 5, 4], win_sz: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.hlb_lvl, self.win_sz = hlb_lvl, [min(win_sz, n_p) for win_sz, n_p in zip(win_sz, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, h_l, w_s in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.hlb_lvl, self.win_sz):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(HilbertSeqWinTransBlockV3(n_block=b_e, hlb_lvl=h_l, win_sz=w_s, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)


# ---------------------------------------- Hilbert Sequential Window Transformer ---------------------------------------- #
class HilbertSeqWinTransBlockV4(nn.Module):
    def __init__(self, n_block: int, hlb_lvl: int, win_sz: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        '''
        重复多次 + 移动窗口
        '''
        self.win_sz = win_sz
        self.gi1 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=False)
        self.gi2 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=False)
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqWinTransLayer(win_sz=win_sz, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])

    def forward(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:                                   # (b_s, n_p, d_f), (b_s, n_p, d_c)
        idx_pa_1, idx_re_1 = self.gi1(z)                                                                # (b_s, n_p), (b_s, n_p)
        idx_pa_2, idx_re_2 = self.gi2(z)                                                                # (b_s, n_p), (b_s, n_p)
        idx_pa_2 = torch.roll(idx_pa_2, self.win_sz//2, dims=1)
        idx_re_2 = torch.roll(idx_re_2, self.win_sz//2, dims=1)
        for i, (gp, st, gr) in enumerate(self.blocks):
            x, z = gp(x, z, idx_pa_1 if i % 2 == 0 else idx_pa_2)                                       # (b_s, n_p, d_f), (b_s, n_p, d_c)
            x, z = st(x, z)                                                                             # (b_s, n_p, d_f), (b_s, n_p, d_c)
            x, z = gr(x, z, idx_re_1 if i % 2 == 0 else idx_re_2)                                       # (b_s, n_p, d_f), (b_s, n_p, d_c)
        return x, z            


class HilbertFormerV1ClsHilbertSeqWinTransV4(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], hlb_lvl: List[int] = [7, 6, 5, 4], win_sz: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.hlb_lvl, self.win_sz = hlb_lvl, [min(win_sz, n_p) for win_sz, n_p in zip(win_sz, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, h_l, w_s in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.hlb_lvl, self.win_sz):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(HilbertSeqWinTransBlockV4(n_block=b_e, hlb_lvl=h_l, win_sz=w_s, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)



# ---------------------------------------- Hilbert Sequential Window Transformer ---------------------------------------- #
class MCConv1D(nn.Module):
    def __init__(self, sz: int, d_feat: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Conv1d(d_feat, d_feat, kernel_size=sz, stride=1, padding='same', bias=False),
                nn.BatchNorm1d(d_feat),
                nn.ReLU(),
                nn.Conv1d(d_feat, d_feat, kernel_size=sz, stride=1, padding='same', bias=False),
                nn.BatchNorm1d(d_feat),
                nn.ReLU(),
                nn.Conv1d(d_feat, d_feat, kernel_size=sz, stride=1, padding='same', bias=False),
                nn.BatchNorm1d(d_feat),
        )
        self.act = nn.ReLU()

    def forward(self, x, z):
        x = x.permute(0, 2, 1).contiguous()
        x = self.act(x + self.mlp(x))
        x = x.permute(0, 2, 1).contiguous()
        return x, z

class HilbertSeqWinTransBlockV5(BaseSeqTransBlock):
    def __init__(self, n_block: int, hlb_lvl: int, win_sz: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        '''
        重复多次 + Conv(3)
        '''
        self.win_sz = win_sz
        self.gi1 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=False)
        self.gi2 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=True)
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            MCConv1D(sz=3, d_feat=d_feat),
            SpaceReverse()
        ]) for _ in range(n_block)])


class HilbertFormerV1ClsHilbertSeqWinTransV5(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], hlb_lvl: List[int] = [7, 6, 5, 4], win_sz: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.hlb_lvl, self.win_sz = hlb_lvl, [min(win_sz, n_p) for win_sz, n_p in zip(win_sz, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, h_l, w_s in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.hlb_lvl, self.win_sz):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(HilbertSeqWinTransBlockV5(n_block=b_e, hlb_lvl=h_l, win_sz=w_s, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)

# ---------------------------------------- Hilbert Sequential Window Transformer ---------------------------------------- #
class HilbertSeqWinTransBlockV6(BaseSeqTransBlock):
    def __init__(self, n_block: int, hlb_lvl: int, win_sz: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        '''
        重复多次 + Conv(GS)
        '''
        self.win_sz = win_sz
        self.gi1 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=False)
        self.gi2 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=True)
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            MCConv1D(sz=win_sz, d_feat=d_feat),
            SpaceReverse()
        ]) for _ in range(n_block)])


class HilbertFormerV1ClsHilbertSeqWinTransV6(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], hlb_lvl: List[int] = [7, 6, 5, 4], win_sz: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.hlb_lvl, self.win_sz = hlb_lvl, [min(win_sz, n_p) for win_sz, n_p in zip(win_sz, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, h_l, w_s in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.hlb_lvl, self.win_sz):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(HilbertSeqWinTransBlockV6(n_block=b_e, hlb_lvl=h_l, win_sz=w_s, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)


# ---------------------------------------- Hilbert Sequential Window Transformer ---------------------------------------- #
class HilbertSeqWinTransBlockV7(BaseSeqTransBlock):
    def __init__(self, n_block: int, hlb_lvl: int, win_sz: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.gi1 = Hilbert3DGetIdx(hlb_lvl=Hilbert3DGetIdx.hlb_lvl_max - 1, hlb_shift=False)
        self.gi2 = Hilbert3DGetIdx(hlb_lvl=Hilbert3DGetIdx.hlb_lvl_max - 1, hlb_shift=True)
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqWinTransLayer(win_sz=win_sz, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])


class HilbertFormerV1ClsHilbertSeqWinTransV7(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], hlb_lvl: List[int] = [7, 6, 5, 4], win_sz: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.hlb_lvl, self.win_sz = hlb_lvl, [min(win_sz, n_p) for win_sz, n_p in zip(win_sz, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, h_l, w_s in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.hlb_lvl, self.win_sz):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(HilbertSeqWinTransBlockV7(n_block=b_e, hlb_lvl=h_l, win_sz=w_s, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)


# ---------------------------------------- Hilbert Sequential KNN Transformer ---------------------------------------- #
class HilbertSeqKNNTransBlock(BaseSeqTransBlock):
    def __init__(self, n_block: int, hlb_lvl: int, num_grp: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.gi1 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=False)
        self.gi2 = Hilbert3DGetIdx(hlb_lvl=hlb_lvl, hlb_shift=True)
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqKNNTransLayer(num_grp=num_grp, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])


class HilbertFormerV1ClsHilbertSeqKNNTrans(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], hlb_lvl: List[int] = [7, 6, 5, 4], num_grp: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.hlb_lvl, self.num_grp = hlb_lvl, [min(k_g, n_p) for k_g, n_p in zip(num_grp, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, h_l, k_g in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.hlb_lvl, self.num_grp):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(HilbertSeqKNNTransBlock(n_block=b_e, hlb_lvl=h_l, num_grp=k_g, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)


# ---------------------------------------- Random Sequential Window Transformer ---------------------------------------- #
class RandomSeqWinTransBlock(BaseSeqTransBlock):
    def __init__(self, n_block: int, win_sz: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.gi1 = Random3DGetIdx()
        self.gi2 = Random3DGetIdx()
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqWinTransLayer(win_sz=win_sz, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])


class HilbertFormerV1ClsRandomSeqWinTrans(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], win_sz: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.win_sz = [min(win_sz, n_p) for win_sz, n_p in zip(win_sz, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, w_s in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.win_sz):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(RandomSeqWinTransBlock(n_block=b_e, win_sz=w_s, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)


# ---------------------------------------- Random Sequential KNN Transformer ---------------------------------------- #
class RandomSeqKNNTransBlock(BaseSeqTransBlock):
    def __init__(self, n_block: int, num_grp: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.gi1 = Random3DGetIdx()
        self.gi2 = Random3DGetIdx()
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqKNNTransLayer(num_grp=num_grp, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])


class HilbertFormerV1ClsRandomSeqKNNTrans(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], num_grp: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.num_grp = [min(k_g, n_p) for k_g, n_p in zip(num_grp, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, k_g in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.num_grp):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(RandomSeqKNNTransBlock(n_block=b_e, num_grp=k_g, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)


# ---------------------------------------- Sorting Coordinates Sequential Window Transformer ---------------------------------------- #
class SortCoordSeqWinTransBlock(nn.Module):
    def __init__(self, n_block: int, win_sz: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.gi1 = SortCoord3DGetIdx(order='xyz')
        # self.gi2 = SortCoord3DGetIdx(order='yxz')
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqWinTransLayer(win_sz=win_sz, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])

    def forward(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:                                   # (b_s, n_p, d_f), (b_s, n_p, d_c)
        idx_pa_1, idx_re_1 = self.gi1(z)                                                                # (b_s, n_p), (b_s, n_p)
        idx_pa_2, idx_re_2 = idx_pa_1, idx_re_1                                                         # (b_s, n_p), (b_s, n_p)
        for i, (gp, st, gr) in enumerate(self.blocks):
            x, z = gp(x, z, idx_pa_1 if i % 2 == 0 else idx_pa_2)                                       # (b_s, n_p, d_f), (b_s, n_p, d_c)
            x, z = st(x, z)                                                                             # (b_s, n_p, d_f), (b_s, n_p, d_c)
            x, z = gr(x, z, idx_re_1 if i % 2 == 0 else idx_re_2)                                       # (b_s, n_p, d_f), (b_s, n_p, d_c)
        return x, z       


class HilbertFormerV1ClsSortCoordSeqWinTrans(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], win_sz: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.win_sz = [min(win_sz, n_p) for win_sz, n_p in zip(win_sz, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, w_s in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.win_sz):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(SortCoordSeqWinTransBlock(n_block=b_e, win_sz=w_s, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)


# ---------------------------------------- Sorting Coordinates Sequential Window Transformer ---------------------------------------- #
class SortCoordSeqWinTransBlockV2(BaseSeqTransBlock):
    def __init__(self, n_block: int, win_sz: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.gi1 = SortCoord3DGetIdx(order='xyz')
        self.gi2 = SortCoord3DGetIdx(order='yxz')
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqWinTransLayer(win_sz=win_sz, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])


class HilbertFormerV1ClsSortCoordSeqWinTransV2(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], win_sz: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.win_sz = [min(win_sz, n_p) for win_sz, n_p in zip(win_sz, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, w_s in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.win_sz):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(SortCoordSeqWinTransBlockV2(n_block=b_e, win_sz=w_s, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)


# ---------------------------------------- Sorting Coordinates Sequential KNN Transformer ---------------------------------------- #
class SortCoordSeqKNNTransBlock(BaseSeqTransBlock):
    def __init__(self, n_block: int, num_grp: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.gi1 = SortCoord3DGetIdx(order='xyz')
        self.gi2 = SortCoord3DGetIdx(order='yxz')
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqKNNTransLayer(num_grp=num_grp, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])


class HilbertFormerV1ClsSortCoordSeqKNNTrans(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5,
                grp_base: int = 24, grp_expand: float = 1.0, blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], num_grp: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.num_grp = [min(k_g, n_p) for k_g, n_p in zip(num_grp, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, k_g in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.num_grp):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(SortCoordSeqKNNTransBlock(n_block=b_e, num_grp=k_g, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)


# ---------------------------------------- Sorting Grids And Coordinates Sequential Window Transformer ---------------------------------------- #
class SortGridAndCoordSeqWinTransBlock(BaseSeqTransBlock):
    def __init__(self, n_block: int, grd_lvl: int, win_sz: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.gi1 = SortGridAndCoord3DGetIdx(grd_lvl=grd_lvl, order='xyz')
        self.gi2 = SortGridAndCoord3DGetIdx(grd_lvl=grd_lvl, order='yxz')
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqWinTransLayer(win_sz=win_sz, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])


class HilbertFormerV1ClsSortGridAndCoordSeqWinTrans(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5, grp_base: int = 24, grp_expand: float = 1.0,
                blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], grd_lvl: List[int] = [4, 4, 4, 4], win_sz: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.grd_lvl, self.win_sz = grd_lvl, [min(win_sz, n_p) for win_sz, n_p in zip(win_sz, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, g_l, w_s in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.grd_lvl, self.win_sz):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(SortGridAndCoordSeqWinTransBlock(n_block=b_e, grd_lvl=g_l, win_sz=w_s, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)


# ---------------------------------------- Sorting Coordinates Sequential KNN Transformer ---------------------------------------- #
class SortGridAndCoordSeqKNNTransBlock(BaseSeqTransBlock):
    def __init__(self, n_block: int, grd_lvl: int, num_grp: int, d_feat: int, n_head: int, ratio: float, attn_drop: float = 0.0, proj_drop: float = 0.0, norm_name: str = 'BN',
                act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        self.gi1 = SortGridAndCoord3DGetIdx(grd_lvl=grd_lvl, order='xyz')
        self.gi2 = SortGridAndCoord3DGetIdx(grd_lvl=grd_lvl, order='yxz')
        self.blocks = nn.ModuleList([nn.ModuleList([
            SpaceExpansion(),
            SeqKNNTransLayer(num_grp=num_grp, d_feat=d_feat, n_head=n_head, ratio=ratio, attn_drop=attn_drop, proj_drop=proj_drop,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args),
            SpaceReverse()
        ]) for _ in range(n_block)])


class HilbertFormerV1ClsSortGridAndCoordSeqKNNTrans(BaseSeqTransBackBone):
    def __init__(self, num_cls: int = 15, num_stg: int = 4, dim_stem: int = 3, dim_base: int = 48, dim_expand: float = 2.0, qry_base: int = 1024, qry_expand: float = 0.5, grp_base: int = 24, grp_expand: float = 1.0,
                blk_dns: List[int] = [1, 1, 2, 1], blk_ens: List[int] = [2, 2, 2, 2], grd_lvl: List[int] = [4, 4, 4, 4], num_grp: List[int] = [8, 8, 8, 8], ratio: float = 1.0, drop: float = 0.0,
                norm_name: str = 'BN', act_name: str = 'ReLU', linear_args: Dict = {'bias': False}, norm_args: Dict = {'eps': 1e-5, 'momentum': 0.1}, act_args: Dict = {'inplace': True}) -> None:
        super().__init__()
        # Initialize parameters
        self.num_cls, self.num_stg, self.b_ds, self.b_es = num_cls, num_stg, blk_dns, blk_ens
        self._init_parms(num_stg=num_stg, d_b=dim_base, d_e=dim_expand, q_b=qry_base, q_e=qry_expand, g_b=grp_base, g_e=grp_expand)
        self.grd_lvl, self.num_grp = grd_lvl, [min(k_g, n_p) for k_g, n_p in zip(num_grp, self.n_qs)]
        # Stem
        self.stem = PointStem(dim_stem, self.d_is[0], norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args)
        # Transition Down & Encoder
        self.dn_blocks, self.en_blocks = nn.ModuleList(), nn.ModuleList()
        for d_i, d_o, n_q, n_g, b_d, b_e, g_l, k_g in zip(self.d_is, self.d_os, self.n_qs, self.n_gs, self.b_ds, self.b_es, self.grd_lvl, self.num_grp):
            self.dn_blocks.append(PointTransitionDown(d_in=d_i, d_out=d_o, num_qry=n_q, num_grp=n_g, n_block=b_d, ratio=ratio,
                    norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
            self.en_blocks.append(SortGridAndCoordSeqKNNTransBlock(n_block=b_e, grd_lvl=g_l, num_grp=k_g, d_feat=d_o, n_head=d_o//32, ratio=ratio, attn_drop=drop,
                    proj_drop=drop, norm_name=norm_name, act_name=act_name, linear_args=linear_args, norm_args=norm_args, act_args=act_args))
        self.pool = Pool('Max', dim=1, keepdim=False)
        self.head = PointClsHead(self.d_os[-1], num_cls)
        self.apply(self._init_weights)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # print(HilbertFormerV1ClsHilbertSeqWinTrans().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
    # print(HilbertFormerV1ClsHilbertSeqKNNTrans().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
    # print(HilbertFormerV1ClsRandomSeqWinTrans().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
    # print(HilbertFormerV1ClsRandomSeqKNNTrans().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
    # print(HilbertFormerV1ClsSortCoordSeqWinTrans().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
    # print(HilbertFormerV1ClsSortCoordSeqKNNTrans().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
    # print(HilbertFormerV1ClsSortGridAndCoordSeqWinTrans().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
    # print(HilbertFormerV1ClsSortGridAndCoordSeqKNNTrans().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
    print(HilbertFormerV1ClsHilbertSeqWinTransV3().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
    print(HilbertFormerV1ClsHilbertSeqWinTransV4().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
    print(HilbertFormerV1ClsHilbertSeqWinTransV5().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
    print(HilbertFormerV1ClsHilbertSeqWinTransV6().cuda()(torch.rand((2, 1024, 3), dtype=torch.float).cuda()).shape)
