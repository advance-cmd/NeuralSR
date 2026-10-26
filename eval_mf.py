"""s194：多帧网络的评估。测试集 4 段，每段从第 0 帧（没有历史）一帧一帧跑到第 31 帧，和单帧模型 B 比，都在屏幕空间：
  PSNR：每帧各算再平均；按帧号分开：第 0 帧（多帧也还没有历史）、第 4–31 帧
  分组 PSNR：所有帧里这一组的像素合起来算。水面（hrregion = 2）、"新露出来的"（遮挡掩码 = 0，不算每段第 0 帧）
用法：python eval_mf.py（约 1 分钟）。写完先跑 python tools/check_mf.py 3
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bicubic import psnr
from dataset import SEQ,SeqDataset
from hdr import decode,encode,tonemap
from model import MFNet,SRNet
from multiframe import run_frame
from warp import backward_warp

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)),"runs")

def frames(dev):
    """按顺序取测试集的每一帧：(段号, 帧号, lr, gb, hr, velocity, jitter)，张量都在 dev 上、去掉了"第几帧"那一维"""
    ds = SeqDataset("test",frames=1,patch=None)
    for i,batch in enumerate(DataLoader(ds,batch_size=1,num_workers=4)):
        s,f = ds.clips[i]
        lr, gb, hr, velocity, jitter = [t[:, 0].to(dev) for t in batch]
        yield s, f, lr, gb, hr, velocity, jitter

def sequence_psnr(model,dev):
    """训练时用：测试集 4 段整段（128 帧），屏幕空间 PSNR 的平均"""
    model.eval()
    scores = []
    with torch.no_grad():
        for s, f, lr, gb, hr, velocity, jitter in frames(dev):
            if f == 0:
                prev = None
            out,prev,mask = run_frame(model,prev,lr,gb,velocity,jitter)
            scores.append(psnr(tonemap(decode(out)), tonemap(hr)).item())
    model.train()
    return sum(scores)/len(scores)

def compare(single, multi, dev):
    """single：单帧模型 B（SRNet）；multi：MFNet。打印一张表"""
    names = ["单帧 B", "多帧"]
    psnrs = {n: [] for n in names}
    err = {n: np.zeros(2) for n in names}  # 水面、新露出来的：平方误差加起来
    flick = {n: [] for n in names}  # 每对相邻帧"多出来的闪烁"
    count = np.zeros(2)
    nums = []  # 每一帧的帧号
    with torch.no_grad():
        for s, f, lr, gb, hr, velocity, jitter in frames(dev):
            if f == 0:
                prev = None
            out, prev, mask = run_frame(multi, prev, lr, gb, velocity, jitter)
            results = {"单帧 B": decode(single(encode(lr), gb)), "多帧": decode(out)}
            imgs = {n:tonemap(results[n]) for n in names}   #屏幕空间
            gt = tonemap(hr)
            if f > 0:  # 每段第 0 帧没有上一帧
                v2 = F.interpolate(velocity, scale_factor=2, mode="bilinear", align_corners=False)  # 这一帧的速度放大到 2K
                zero = torch.zeros(1,2,device=dev)
                gw, _ = backward_warp(prev_gt, v2, zero, zero)  # 拉过来的上一帧 GT
                d_gt = gt - gw  # GT 这两帧之间本来该变多少
                for n in names:
                    w, _ = backward_warp(prev_imgs[n], v2, zero, zero)  # 拉过来的这个方法的上一帧
                    # 多出来的变化，只算掩码 = 1 的地方，除以"掩码里的像素数 × 3 个通道"
                    flick[n].append((((imgs[n] - w) - d_gt).abs() * mask).sum().item() / (mask.sum().item() * 3))
            prev_imgs, prev_gt = imgs, gt  # 留给下一帧

            region = torch.from_numpy(np.load(os.path.join(SEQ, "test", f"{s}_{f:02d}_hrregion.npy"))).to(dev)
            # 这一帧"新露出来的"像素：掩码是 0 的地方；第 0 帧整帧都没有历史，不算进这一组
            new = (mask[0, 0] == 0) & (f > 0)
            groups = [region == 2, new]
            for g in range(2):
                count[g] += groups[g].sum().item()
            nums.append(f)
            for n in names:
                se = ((imgs[n] - gt) ** 2).mean(1)[0]  # 每个像素的平方误差：(H, W)
                psnrs[n].append(10 * torch.log10(1 / se.mean()).item())
                for g in range(2):
                    err[n][g] += se[groups[g]].sum().item()
    nums = np.array(nums)
    print("测试集 4 段 × 32 帧，每段从第 0 帧跑起；屏幕空间（× 2^EV → ACES → sRGB，不加 Bloom）")
    print("| 方法 | PSNR | 第 0 帧 | 第 4–31 帧 | 水面 | 新露出来的 | 闪烁 ×1000 |")
    print("|---|---|---|---|---|---|---|")
    for n in names:
        p = np.array(psnrs[n])
        g = 10 * np.log10(count / err[n])
        print(f"| {n} | {p.mean():.2f} | {p[nums == 0].mean():.2f} | {p[nums >= 4].mean():.2f} | {g[0]:.2f} | {g[1]:.2f} | {np.mean(flick[n]) * 1000:.2f} |")
    print(f"新露出来的像素占第 1–31 帧的 {count[1] / ((nums > 0).sum() * 1440 * 2560) * 100:.2f}%")

if __name__ == "__main__":
    dev = torch.device("cuda")
    single = SRNet(True).to(dev)
    single.load_state_dict(torch.load(os.path.join(RUNS, "hdr_B.pth")))
    multi = MFNet().to(dev)
    multi.load_state_dict(torch.load(os.path.join(RUNS, "mf.pth")))
    compare(single.eval(), multi.eval(), dev)