'''
CUDA_VISIBLE_DEVICES=0 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqKNNTrans.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqKNNTrans.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsRandomSeqWinTrans.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsRandomSeqWinTrans.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsRandomSeqKNNTrans.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsRandomSeqKNNTrans.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortCoordSeqWinTrans.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortCoordSeqWinTrans.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortCoordSeqKNNTrans.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortCoordSeqKNNTrans.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortGridAndCoordSeqWinTrans.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortGridAndCoordSeqWinTrans.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortGridAndCoordSeqKNNTrans.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortGridAndCoordSeqKNNTrans.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfgs/DifferentGroupSize/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_GS4.yaml

CUDA_VISIBLE_DEVICES=0 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTransV7.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTransV7.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortCoordSeqWinTrans.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortCoordSeqWinTrans.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortCoordSeqWinTransV2.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortCoordSeqWinTransV2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortGridAndCoordSeqWinTrans.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsSortGridAndCoordSeqWinTrans.txt 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTransV3.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTransV3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTransV4.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTransV4.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTransV5.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTransV5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python main.py --cfg cfgs/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTransV6.yaml > output/DifferentExpansion/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTransV6.txt 2>&1 &


# Expansin Level

CUDA_VISIBLE_DEVICES=0 nohup python main.py --cfg cfgs/DifferentExpansionLevel/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_EL4.yaml > output/DifferentExpansionLevel/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_EL4.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python main.py --cfg cfgs/DifferentExpansionLevel/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_EL5.yaml > output/DifferentExpansionLevel/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_EL5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python main.py --cfg cfgs/DifferentExpansionLevel/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_EL6.yaml > output/DifferentExpansionLevel/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_EL6.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python main.py --cfg cfgs/DifferentExpansionLevel/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_EL7.yaml > output/DifferentExpansionLevel/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_EL7.txt 2>&1 &

# Group Size

CUDA_VISIBLE_DEVICES=0 nohup python main.py --cfg cfgs/DifferentGroupSize/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_GS4.yaml > output/DifferentGroupSize/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_GS4.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python main.py --cfg cfgs/DifferentGroupSize/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_GS8.yaml > output/DifferentGroupSize/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_GS8.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python main.py --cfg cfgs/DifferentGroupSize/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_GS16.yaml > output/DifferentGroupSize/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_GS16.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python main.py --cfg cfgs/DifferentGroupSize/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_GS32.yaml > output/DifferentGroupSize/ScanObjectNN_HilbertFormerV1ClsHilbertSeqWinTrans_GS32.txt 2>&1 &

'''

use_tqdm = False

import os
import argparse
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import set_random_seed, IOStream, generate_exp_dir, load_config
import sklearn.metrics as metrics
from build import build_dataloader_from_cfg, build_model_from_cfg, build_loss_from_cfg, build_optimizer_from_cfg, build_scheduler_from_cfg


def train(cfg, io, exp_path):
    io.print(f"Let's use {str(torch.cuda.device_count())} GPUs!")
    train_loader = build_dataloader_from_cfg(cfg, 'train')
    test_loader = build_dataloader_from_cfg(cfg, 'test')
    model = nn.DataParallel(build_model_from_cfg(cfg.model).cuda())
    criterion = build_loss_from_cfg(cfg.loss).cuda()
    optimizer = build_optimizer_from_cfg(cfg.optimizer, model)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    best_test_oa, best_test_macc = 0, 0
    for epoch in range(cfg.epochs):
        # ---------------------------------------- Train ---------------------------------------- #
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred_cls = []
        train_true_cls = []
        pbar = tqdm(enumerate(train_loader), total=train_loader.__len__()) if cfg.use_tqdm else enumerate(train_loader)
        for i, (data, cls) in pbar:
            data, cls_true = data.cuda(non_blocking=True), cls.cuda(non_blocking=True)
            batch_size = data.size(0)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, cls_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            optimizer.step()
            cls_pred = logits.max(dim=-1)[1]                                # (batch_size)
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true_cls.append(cls_true.cpu().numpy())                   # (batch_size)
            train_pred_cls.append(cls_pred.detach().cpu().numpy())          # (batch_size)

        scheduler.step()
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_oa = metrics.accuracy_score(train_true_cls, train_pred_cls)
        train_macc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        io.print(f'Train {epoch}, loss: {train_loss/count:.4f}, train oa: {train_oa:.4f}, train macc: {train_macc:.4f}')

        # ---------------------------------------- Test ---------------------------------------- #
        with torch.no_grad():
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_true_cls = []
            test_pred_cls = []
            pbar = tqdm(enumerate(test_loader), total=test_loader.__len__()) if cfg.use_tqdm else enumerate(test_loader)
            for i, (data, cls) in pbar:
                data, cls_true = data.cuda(non_blocking=True), cls.cuda(non_blocking=True)
                batch_size = data.size(0)
                logits = model(data)
                loss = criterion(logits, cls_true)
                cls_pred = logits.max(dim=-1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true_cls.append(cls_true.cpu().numpy())                # (batch_size)
                test_pred_cls.append(cls_pred.detach().cpu().numpy())       # (batch_size)

            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_oa = metrics.accuracy_score(test_true_cls, test_pred_cls)
            test_macc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            if test_oa >= best_test_oa:
                best_test_oa = test_oa
                torch.save(model.state_dict(), f'{exp_path}/model_oa.pth')
            if test_macc >= best_test_macc:
                best_test_macc = test_macc
                torch.save(model.state_dict(), f'{exp_path}/model_macc.pth')

            io.print(f'Test {epoch}, loss: {test_loss/count:.4f}, test oa: {test_oa:.4f}, test macc: {test_macc:.4f}, best oa: {best_test_oa:.4f}, best macc: {best_test_macc:.4f}')


@torch.no_grad()
def test(cfg, io, exp_path):
    io.print(f"Let's use {str(torch.cuda.device_count())} GPUs!")
    test_loader = build_dataloader_from_cfg(cfg, 'test')
    model.load_state_dict(torch.load(cfg.experiment.pretrain_path))
    model = nn.DataParallel(build_model_from_cfg(cfg.model).cuda())
    model = model.eval()

    test_true_cls, test_pred_cls = [], []
    for data, cls in test_loader:
        data, cls_true = data.cuda(non_blocking=True), cls.cuda(non_blocking=True)
        logits = model(data)
        cls_pred = logits.max(dim=-1)[1]
        test_true_cls.append(cls_true.cpu().numpy())
        test_pred_cls.append(cls_pred.detach().cpu().numpy())

    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_oa = metrics.accuracy_score(test_true_cls, test_pred_cls)
    test_macc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    io.print(f'Test :: test acc: {test_oa:.4f}, test macc: {test_macc:.4f}')


if __name__ == "__main__":
    # for i in range(10):
    parser = argparse.ArgumentParser(description='Point Cloud Analysis')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    args = parser.parse_args()
    cfg = EasyDict(load_config(args.cfg))
    cfg.update({'use_tqdm': use_tqdm})
    assert torch.cuda.is_available() and cfg.mode in ['train', 'test']
    cfg.seed = cfg.get('seed', np.random.randint(1, 10000))
    set_random_seed(cfg.seed)
    exp_path = generate_exp_dir(cfg.experiment, seed=cfg.seed)
    io = IOStream(f'{exp_path}/{cfg.mode}.log')
    io.print(str(cfg))
    io.print(f'random seed is: {str(cfg.seed)}')
    io.print(f'Using GPU: {str(torch.cuda.current_device())} from {str(torch.cuda.device_count())} devices')
    if cfg.mode == 'train':
        train(cfg, io, exp_path)
    else:
        test(cfg, io, exp_path)
            